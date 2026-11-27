# CORE_AI/drug_database.py
"""Resolve a drug name to a SMILES string, live, via PubChem.

Design notes
------------
This module deliberately contains **no hand-written SMILES**. An earlier version
shipped a hardcoded table in which several entries were wrong (warfarin carried
an aspirin-like string; cisplatin/oxaliplatin held molecular formulas, not
SMILES). Every molecular descriptor downstream is derived from this string, so a
wrong entry silently corrupts every prediction for that drug. PubChem is the
authority; this module is only a cache in front of it.

Cache layout (drug_database.json)::

    {"hits": {"<normalised name>": "<smiles>"},
     "misses": ["<normalised name>", ...]}

`misses` matters for training: the PharmGKB/DGIdb tables contain tens of
thousands of non-drug entity names (diseases, phenotypes, star alleles like
CYP2C9*3). Without negative caching, every training run re-queries all of them.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(_HERE, "drug_database.json")
MISSING_LOG = os.path.join(_HERE, "missing_smiles.log")

# PubChem renamed its SMILES properties in 2025; CanonicalSMILES still responds
# but is the legacy spelling. Try the current name first and fall back.
_SMILES_PROPERTIES = ("SMILES", "ConnectivitySMILES", "CanonicalSMILES")
_URL = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
        "name/{name}/property/{prop}/TXT")
_CID_URL = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
            "name/{name}/cids/TXT")
# A brand name usually resolves to the marketed salt, not the active moiety:
# "Plavix" -> CID 115366 (clopidogrel bisulfate), whose parent is CID 60606
# (clopidogrel). Pharmacogenomic guidance is written about the parent, so both
# must map to the same knowledge-base entry.
_PARENT_URL = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
               "cid/{cid}/cids/TXT?cids_type=parent")

# PubChem asks for no more than 5 requests/second.
_MIN_INTERVAL = 0.25
_TIMEOUT = 10
_RETRIES = 3

_lock = threading.Lock()
_last_request = 0.0
_session = requests.Session()
_session.headers["User-Agent"] = "MedShieldAI/0.1 (research prototype)"

# Live cache. Kept module-level under the historical name because
# chemical_reader, data_loader_real, check_drugs and test_featurization all
# import it directly.
DRUG_DATABASE: dict[str, str] = {}
_MISSES: set[str] = set()
_dirty = False

# name -> PubChem CID. The CID is what makes name matching reliable: PubChem
# resolves brand names, salts and synonyms ("Plavix", "clopidogrel bisulfate")
# onto the same compound, so the knowledge base can be keyed on CID instead of
# on whatever string the user happened to type.
CID_CACHE: dict[str, int] = {}
_CID_MISSES: set[str] = set()
PARENT_CACHE: dict[str, int] = {}   # str(cid) -> parent cid


def normalise(drug_name: str) -> str:
    """Canonical cache key: lowercase, collapsed whitespace."""
    return " ".join((drug_name or "").split()).lower()


def _valid_smiles(smiles: str) -> bool:
    """Reject anything RDKit cannot parse, so bad strings never enter the cache."""
    if not smiles or len(smiles) > 2000:
        return False
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        return Chem.MolFromSmiles(smiles) is not None
    except ImportError:
        # No RDKit available: accept, and let the caller's own parse fail loudly.
        return True


def load_cache() -> None:
    """Populate DRUG_DATABASE and the miss set from disk. Safe if absent."""
    global _MISSES
    if not os.path.exists(CACHE_FILE):
        return
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] Could not read SMILES cache ({exc}); starting empty.")
        return

    global _CID_MISSES
    if isinstance(raw, dict) and "hits" in raw:
        hits = raw.get("hits") or {}
        _MISSES = set(raw.get("misses") or [])
        CID_CACHE.update({k: int(v) for k, v in (raw.get("cids") or {}).items()})
        _CID_MISSES = set(raw.get("cid_misses") or [])
        PARENT_CACHE.update({k: int(v) for k, v in (raw.get("parents") or {}).items()})
    else:
        # Legacy flat {name: smiles} file.
        hits = raw if isinstance(raw, dict) else {}

    kept = dropped = 0
    for name, smiles in hits.items():
        if isinstance(smiles, str) and _valid_smiles(smiles):
            DRUG_DATABASE[normalise(name)] = smiles
            kept += 1
        else:
            dropped += 1
    msg = (f"[cache] SMILES cache: {kept} entries, {len(_MISSES)} known misses, "
           f"{len(CID_CACHE)} CIDs")
    if dropped:
        msg += f" ({dropped} unparseable entries dropped)"
    print(msg)


def save_cache() -> None:
    """Write the cache to disk atomically."""
    global _dirty
    payload = {"hits": DRUG_DATABASE, "misses": sorted(_MISSES),
               "cids": CID_CACHE, "cid_misses": sorted(_CID_MISSES),
               "parents": PARENT_CACHE}
    tmp = CACHE_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, CACHE_FILE)
        _dirty = False
    except OSError as exc:
        print(f"[warn] Could not write SMILES cache: {exc}")


def _log_miss(name: str, detail: str = "") -> None:
    try:
        with open(MISSING_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{name}\t{detail}\n" if detail else f"{name}\n")
    except OSError:
        pass


def _throttle() -> None:
    global _last_request
    wait = _MIN_INTERVAL - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def _fetch_from_pubchem(name: str) -> Optional[str]:
    """Query PubChem. Returns a validated SMILES, or None if unresolvable.

    Only network/HTTP failures are caught here. Programming errors propagate, so
    a bug in this function can never again be mistaken for 'drug not found'.
    """
    from urllib.parse import quote

    for prop in _SMILES_PROPERTIES:
        url = _URL.format(name=quote(name, safe=""), prop=prop)
        for attempt in range(_RETRIES):
            _throttle()
            try:
                resp = _session.get(url, timeout=_TIMEOUT)
            except requests.RequestException as exc:
                if attempt == _RETRIES - 1:
                    _log_miss(name, f"network error: {exc}")
                    return None
                time.sleep(0.5 * (attempt + 1))
                continue

            if resp.status_code == 200:
                body = resp.text.strip()
                smiles = body.splitlines()[0].strip() if body else ""
                if _valid_smiles(smiles):
                    return smiles
                _log_miss(name, f"{prop} returned unparseable: {smiles[:60]!r}")
                break  # try the next property spelling
            if resp.status_code == 404:
                return None  # genuinely not a compound
            if resp.status_code == 400:
                break  # property name not supported; try the next spelling
            if resp.status_code in (429, 503):
                time.sleep(1.0 * (attempt + 1))
                continue
            _log_miss(name, f"HTTP {resp.status_code}")
            return None
    return None


def get_smiles(drug_name: str, use_network: bool = True) -> Optional[str]:
    """Return a SMILES string for `drug_name`, or None.

    Exact (normalised) match against the cache, then PubChem. Substring
    matching was removed on purpose: it previously made "warfarin sodium"
    resolve to the local warfarin entry, masking the live lookup.
    """
    global _dirty
    name = normalise(drug_name)
    if not name:
        return None

    hit = DRUG_DATABASE.get(name)
    if hit:
        return hit
    if name in _MISSES or not use_network:
        return None

    with _lock:
        hit = DRUG_DATABASE.get(name)
        if hit:
            return hit
        if name in _MISSES:
            return None

        smiles = _fetch_from_pubchem(name)
        if smiles:
            DRUG_DATABASE[name] = smiles
        else:
            _MISSES.add(name)
            _log_miss(drug_name)
        _dirty = True

        # Batch writes: training loops resolve tens of thousands of names.
        if len(DRUG_DATABASE) % 25 == 0 or len(_MISSES) % 100 == 0:
            save_cache()
        return smiles


def get_cid(drug_name: str, use_network: bool = True) -> Optional[int]:
    """Return the PubChem CID for a drug name, or None.

    Used to key the pharmacogenomic knowledge base. PubChem resolves brand
    names and salt forms onto the same CID, which is why lookups keyed on CID
    behave correctly where string matching does not.
    """
    global _dirty
    from urllib.parse import quote

    name = normalise(drug_name)
    if not name:
        return None
    if name in CID_CACHE:
        return CID_CACHE[name]
    if name in _CID_MISSES or not use_network:
        return None

    with _lock:
        if name in CID_CACHE:
            return CID_CACHE[name]
        if name in _CID_MISSES:
            return None

        cid = None
        for attempt in range(_RETRIES):
            _throttle()
            try:
                resp = _session.get(_CID_URL.format(name=quote(name, safe="")),
                                    timeout=_TIMEOUT)
            except requests.RequestException:
                if attempt == _RETRIES - 1:
                    break
                time.sleep(0.5 * (attempt + 1))
                continue
            if resp.status_code == 200:
                first = resp.text.strip().splitlines()
                if first and first[0].strip().isdigit():
                    cid = int(first[0].strip())
                break
            if resp.status_code == 404:
                break
            if resp.status_code in (429, 503):
                time.sleep(1.0 * (attempt + 1))
                continue
            break

        if cid is not None:
            CID_CACHE[name] = cid
        else:
            _CID_MISSES.add(name)
        _dirty = True
        return cid


def get_parent_cid(cid: int, use_network: bool = True) -> Optional[int]:
    """Return the parent (active moiety) CID for a CID, or None.

    Maps a marketed salt or brand form onto the compound that pharmacogenomic
    guidance is written about. Returns the CID itself when it is already a parent.
    """
    global _dirty
    if cid is None:
        return None
    key = str(cid)
    if key in PARENT_CACHE:
        return PARENT_CACHE[key]
    if not use_network:
        return None

    with _lock:
        if key in PARENT_CACHE:
            return PARENT_CACHE[key]
        parent = None
        for attempt in range(_RETRIES):
            _throttle()
            try:
                resp = _session.get(_PARENT_URL.format(cid=cid), timeout=_TIMEOUT)
            except requests.RequestException:
                if attempt == _RETRIES - 1:
                    break
                time.sleep(0.5 * (attempt + 1))
                continue
            if resp.status_code == 200:
                lines = resp.text.strip().splitlines()
                if lines and lines[0].strip().isdigit():
                    parent = int(lines[0].strip())
                break
            if resp.status_code in (429, 503):
                time.sleep(1.0 * (attempt + 1))
                continue
            break

        if parent is not None:
            PARENT_CACHE[key] = parent
            _dirty = True
        return parent


def flush() -> None:
    """Persist pending cache changes. Call at the end of a training run."""
    if _dirty:
        save_cache()


load_cache()


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or ["warfarin", "aspirin", "cisplatin", "oxaliplatin",
                             "clopidogrel", "not_a_real_drug_xyz"]
    for n in names:
        print(f"{n:28} -> {get_smiles(n)}")
    flush()
