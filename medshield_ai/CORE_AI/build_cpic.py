# CORE_AI/build_cpic.py
"""Fetch the CPIC prescribing tables and store them compactly.

Why fetch instead of hand-writing
---------------------------------
Going from a genotype to a prescribing action needs two mappings that no file in
this repository contains:

  diplotype -> phenotype      CYP2C19 *2/*2            -> Poor Metabolizer
  phenotype -> recommendation Poor Metabolizer + clopidogrel
                              -> "Avoid standard dose clopidogrel if possible"

CPIC (cpicpgx.org) publishes both, peer-reviewed and versioned, and serves them
over a public API. They are transcribed here verbatim rather than paraphrased.
Writing them from memory is not an option: an earlier version of this project
shipped a hand-written SMILES table in which 24 of 30 entries were the wrong
molecule, and a hand-written "fluorouracil -> TPMT" rule that is simply wrong.
Dosing guidance is a much worse thing to get wrong than a molecular weight.

Output: data/cpic.json

  {"drugs":       {"<rxnorm id>": "<name>"},
   "genes":       ["CYP2C19", ...],
   "guidelines":  {"<id>": {"name":..., "url":...}},
   "diplotypes":  {"CYP2C19": {"v": [["Poor Metabolizer", "Poor Metabolizer"], ...],
                              "m": {"*2/*2": 0, ...}}},   # index into "v"
   "recommendations": [ {...} ]}

The diplotype encoding matters: CPIC enumerates every allele combination, which
is 111,970 rows mapping to very few outcomes (CYP2D6 has 16,836 diplotypes over
22 distinct outcomes). Interning the (generesult, lookupkey) pairs per gene
takes it from 9.9 MB to ~4.4 MB losslessly, with every diplotype key kept so
that an unrecognised genotype can still be told apart from a known one.

Run:  python -m CORE_AI.build_cpic
      python -m CORE_AI.build_cpic --genes CYP2C19,CYP2D6   (subset, for testing)
      python -m CORE_AI.build_cpic --reencode               (recompress, no refetch)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")
OUT_JSON = os.path.join(DATA_DIR, "cpic.json")

API = "https://api.cpicpgx.org/v1"
PAGE = 1000
_MIN_INTERVAL = 0.2          # be a polite client of a public academic API
_TIMEOUT = 45
_RETRIES = 3

_session = requests.Session()
# Generic client name on purpose: no user or project identity is sent.
_session.headers["User-Agent"] = "pharmacogenomics-research-client/1.0"
_session.headers["Accept"] = "application/json"
_last = 0.0

# Fields kept from the recommendation table. The rest (ids, flags we do not
# surface) are dropped to keep the file small enough to commit.
REC_FIELDS = ("guidelineid", "drugid", "lookupkey", "implications",
              "drugrecommendation", "classification", "allelestatus",
              "activityscore", "population", "comments",
              "dosinginformation", "alternatedrugavailable",
              "otherprescribingguidance")


def _throttle():
    global _last
    wait = _MIN_INTERVAL - (time.monotonic() - _last)
    if wait > 0:
        time.sleep(wait)
    _last = time.monotonic()


def fetch(table: str, params: str = "", select: str | None = None) -> list[dict]:
    """Fetch a whole table, paginating. Raises on repeated failure."""
    rows: list[dict] = []
    offset = 0
    while True:
        q = f"{API}/{table}?limit={PAGE}&offset={offset}"
        if select:
            q += f"&select={select}"
        if params:
            q += f"&{params}"
        last_exc = None
        for attempt in range(_RETRIES):
            _throttle()
            try:
                resp = _session.get(q, timeout=_TIMEOUT)
                if resp.status_code == 200:
                    batch = resp.json()
                    break
                if resp.status_code in (429, 503):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"{table}: HTTP {resp.status_code} {resp.text[:200]}")
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(0.8 * (attempt + 1))
        else:
            raise RuntimeError(f"{table}: giving up after {_RETRIES} tries ({last_exc})")

        rows.extend(batch)
        if len(batch) < PAGE:
            return rows
        offset += PAGE
        print(f"    {table}: {len(rows)} rows...")


def encode_diplotypes(flat: dict[str, dict]) -> dict:
    """Compress {gene: {diplotype: [generesult, lookupkey]}} losslessly.

    Per gene, the distinct (generesult, lookupkey) pairs are interned and each
    diplotype stores an index into them. 111,970 rows collapse from 9.9 MB to
    ~4.4 MB because the phenotype strings repeat heavily (CYP2D6 has 16,836
    diplotypes over 22 distinct outcomes).

    Every diplotype key is kept. An earlier attempt stored only a per-gene
    default plus exceptions, which was smaller but unsafe: without the full key
    set an unrecognised diplotype is indistinguishable from one that maps to the
    default, so a typo would silently return a phenotype. Membership has to be
    checkable.
    """
    out = {}
    for gene, table in flat.items():
        values = sorted({(gr, lk) for gr, lk in table.values()})
        index = {v: i for i, v in enumerate(values)}
        out[gene] = {
            "v": [list(v) for v in values],
            "m": {dip: index[(gr, lk)] for dip, (gr, lk) in table.items()},
        }
    return out


def decode_diplotypes(encoded: dict) -> dict[str, dict]:
    """Inverse of encode_diplotypes; used by tests to prove losslessness."""
    out = {}
    for gene, blob in encoded.items():
        values = [list(v) for v in blob["v"]]
        out[gene] = {dip: values[idx] for dip, idx in (blob.get("m") or {}).items()}
    return out


def genes_with_recommendations(recs: list[dict]) -> list[str]:
    genes = set()
    for r in recs:
        genes.update((r.get("lookupkey") or {}).keys())
    return sorted(genes)


def build(gene_filter: list[str] | None = None) -> dict:
    print("[cpic] recommendations")
    recs_raw = fetch("recommendation")
    recs = [{k: r.get(k) for k in REC_FIELDS} for r in recs_raw]

    genes = gene_filter or genes_with_recommendations(recs)
    print(f"[cpic] {len(recs)} recommendations over {len(genes)} genes")

    print("[cpic] drugs")
    drugs = {d["drugid"]: d["name"] for d in fetch("drug", select="drugid,name")}

    print("[cpic] guidelines")
    guidelines = {}
    for g in fetch("guideline", select="id,name,url"):
        guidelines[str(g["id"])] = {"name": g.get("name"), "url": g.get("url")}

    print("[cpic] diplotypes (per gene)")
    diplotypes: dict[str, dict] = {}
    for gene in genes:
        rows = fetch("diplotype", params=f"genesymbol=eq.{gene}",
                     select="genesymbol,diplotype,generesult,lookupkey")
        table = {}
        for r in rows:
            dip = r.get("diplotype")
            if not dip:
                continue
            lk = (r.get("lookupkey") or {}).get(gene)
            table[dip] = [r.get("generesult"), lk]
        diplotypes[gene] = table
        print(f"    {gene:10} {len(table)} diplotypes")

    total_flat = sum(len(v) for v in diplotypes.values())
    diplotypes = encode_diplotypes(diplotypes)

    return {
        "_source": "CPIC (cpicpgx.org) public API, api.cpicpgx.org/v1",
        "_fetched": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "_note": ("Recommendation text is transcribed verbatim from CPIC. "
                  "Classification is CPIC's strength of recommendation "
                  "(Strong / Moderate / Optional / No recommendation)."),
        "_counts": {"recommendations": len(recs), "drugs": len(drugs),
                    "genes": len(diplotypes), "diplotypes": total_flat},
        "drugs": drugs,
        "genes": sorted(diplotypes),
        "guidelines": guidelines,
        "diplotypes": diplotypes,
        "recommendations": recs,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", help="comma-separated subset, for testing")
    ap.add_argument("--reencode", action="store_true",
                    help="recompress an existing cpic.json without refetching")
    args = ap.parse_args()

    if args.reencode:
        with open(OUT_JSON, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        dp = data.get("diplotypes") or {}
        if dp and "v" in next(iter(dp.values()), {}):
            print("[cpic] already in compact form; nothing to do")
            raise SystemExit(0)
        data["_counts"]["diplotypes"] = sum(len(v) for v in dp.values())
        data["diplotypes"] = encode_diplotypes(dp)
        print("[cpic] re-encoded existing tables (no network requests)")
    else:
        data = build([g.strip().upper() for g in args.genes.split(",")]
                     if args.genes else None)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"), sort_keys=False)

    size = os.path.getsize(OUT_JSON) / 1048576
    print()
    print(f"[ok] wrote {OUT_JSON}  ({size:.2f} MB)")
    for k, v in data["_counts"].items():
        print(f"     {k}: {v}")
