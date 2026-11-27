# CORE_AI/cpic.py
"""Genotype-aware prescribing guidance, from the CPIC tables.

This is the layer that turns "CYP2C19 matters for clopidogrel" into "this
patient is a poor metaboliser, so avoid clopidogrel and use prasugrel or
ticagrelor instead" -- which is what a clinician can actually act on.

All recommendation text is CPIC's, transcribed verbatim by build_cpic.py. This
module only matches a genotype to the right row; it never composes clinical
advice of its own.

Matching
--------
A genotype value may be given in any of three forms, and each resolves to the
CPIC "lookup key" that recommendations are indexed by:

  diplotype       "*2/*2"            -> Poor Metabolizer
  phenotype       "Poor Metabolizer" -> Poor Metabolizer      (used as-is)
  allele status   "*57:01 positive"  -> *57:01 positive       (HLA genes)

A recommendation may require more than one gene. Those are reported as
"incomplete" rather than matched, because a partial genotype cannot satisfy
them and silently dropping the requirement would misrepresent the guidance.

Build the data first:  python -m CORE_AI.build_cpic
"""

from __future__ import annotations

import json
import os
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
CPIC_PATH = os.path.join(_HERE, "data", "cpic.json")

# CPIC's strength of recommendation, strongest first.
CLASSIFICATION_RANK = {
    "Strong": 3, "Moderate": 2, "Optional": 1,
    "No Recommendation": 0, "n/a": 0,
}

_DATA: dict = {}
_LOADED = False
_NAME_TO_ID: dict[str, str] = {}
_LOOKUP_VALUES: dict[str, set] = {}
# gene -> phenotype name -> the lookup keys that phenotype covers.
# For CYP2D6 and DPYD, CPIC indexes recommendations by ACTIVITY SCORE, and one
# phenotype name spans several scores with materially different guidance (DPYD
# "Poor Metabolizer" is score 0.0 or 0.5: avoid entirely, or halve the dose).
# Such an input is ambiguous and is reported as such rather than resolved.
_PHENOTYPE_KEYS: dict[str, dict[str, list]] = {}


def load(path: str = CPIC_PATH) -> dict:
    global _DATA, _LOADED, _NAME_TO_ID, _LOOKUP_VALUES, _PHENOTYPE_KEYS
    if _LOADED:
        return _DATA
    _LOADED = True
    try:
        with open(path, "r", encoding="utf-8") as fh:
            _DATA = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        _DATA = {}
        print(f"[warn] CPIC tables unavailable ({exc}). "
              f"Run: python -m CORE_AI.build_cpic")
        return _DATA

    _NAME_TO_ID = {str(name).strip().lower(): did
                   for did, name in (_DATA.get("drugs") or {}).items()}

    # Every lookup-key value CPIC actually uses, per gene. Lets a caller pass a
    # phenotype directly, and lets us report valid options on a bad input.
    _LOOKUP_VALUES = {}
    for rec in _DATA.get("recommendations") or []:
        for gene, val in (rec.get("lookupkey") or {}).items():
            _LOOKUP_VALUES.setdefault(gene, set()).add(str(val))

    # Phenotype name -> lookup keys, taken from the diplotype table.
    _PHENOTYPE_KEYS = {}
    for gene, blob in (_DATA.get("diplotypes") or {}).items():
        mapping: dict[str, set] = {}
        for generesult, lookupkey in (blob.get("v") or []):
            if generesult and lookupkey not in (None, "n/a"):
                mapping.setdefault(str(generesult), set()).add(str(lookupkey))
        _PHENOTYPE_KEYS[gene] = {k: sorted(v) for k, v in mapping.items()}

    c = _DATA.get("_counts") or {}
    print(f"[cpic] {c.get('recommendations', 0)} recommendations, "
          f"{c.get('genes', 0)} genes, {c.get('diplotypes', 0)} diplotypes "
          f"(fetched {_DATA.get('_fetched')})")
    return _DATA


def is_available() -> bool:
    return bool(load().get("recommendations"))


def genes_covered() -> list[str]:
    return list(load().get("genes") or [])


def drug_covered(drug_name: str) -> bool:
    load()
    return _resolve_drug_id(drug_name) is not None


def _resolve_drug_id(drug_name: str) -> Optional[str]:
    load()
    key = " ".join((drug_name or "").split()).lower()
    if not key:
        return None
    if key in _NAME_TO_ID:
        return _NAME_TO_ID[key]
    # CPIC names are generic and lowercase; try dropping a trailing salt word.
    from CORE_AI.build_knowledge_base import canonical_drug
    canon = canonical_drug(drug_name)
    return _NAME_TO_ID.get(canon)


def _gene_blob(gene: str) -> Optional[dict]:
    return (load().get("diplotypes") or {}).get(gene.upper())


def _diplotype_value(gene: str, diplotype: str) -> Optional[tuple]:
    """(generesult, lookupkey) for a diplotype, or None if CPIC does not list it.

    Reads the interned encoding from build_cpic.encode_diplotypes. Returning
    None for an unlisted diplotype is deliberate: a typo must not resolve to a
    phenotype.
    """
    blob = _gene_blob(gene)
    if not blob:
        return None
    idx = (blob.get("m") or {}).get(diplotype)
    if idx is None:
        return None
    values = blob.get("v") or []
    return tuple(values[idx]) if 0 <= idx < len(values) else None


def valid_diplotypes(gene: str, limit: int = 0) -> list[str]:
    """Every diplotype CPIC lists for a gene."""
    blob = _gene_blob(gene)
    if not blob:
        return []
    keys = sorted((blob.get("m") or {}).keys())
    return keys[:limit] if limit else keys


def valid_phenotypes(gene: str) -> list[str]:
    load()
    return sorted(_LOOKUP_VALUES.get(gene.upper(), set()))


def resolve_genotype(gene: str, value: str) -> dict:
    """Resolve a diplotype / phenotype / allele status to a CPIC lookup key.

    Returns {"gene","input","lookup_key","phenotype","interpreted_as"} or
    {"gene","input","error", ...} when it cannot be resolved.
    """
    load()
    gene = (gene or "").upper().strip()
    raw = (value or "").strip()
    if not gene or not raw:
        return {"gene": gene, "input": raw, "error": "gene and value are required"}

    if _gene_blob(gene) is None and gene not in _LOOKUP_VALUES:
        return {"gene": gene, "input": raw,
                "error": f"CPIC has no prescribing tables for {gene}",
                "genes_available": genes_covered()}

    # 1. A diplotype, exactly as CPIC writes it.
    found = _diplotype_value(gene, raw)
    if found:
        generesult, lookup = found
        return {"gene": gene, "input": raw, "lookup_keys": [lookup],
                "lookup_key": lookup, "phenotype": generesult,
                "interpreted_as": "diplotype", "ambiguous": False}

    # 2. A diplotype with the alleles the other way round (*1/*2 vs *2/*1).
    if "/" in raw:
        a, _, b = raw.partition("/")
        flipped = f"{b.strip()}/{a.strip()}"
        found = _diplotype_value(gene, flipped)
        if found:
            generesult, lookup = found
            return {"gene": gene, "input": raw, "lookup_keys": [lookup],
                    "lookup_key": lookup, "phenotype": generesult,
                    "interpreted_as": f"diplotype (normalised to {flipped})",
                    "ambiguous": False}

    # 3. A lookup key given directly: a phenotype name for genes indexed that
    #    way, an activity score ("0.5"), or an allele status ("*57:01 positive").
    for known in _LOOKUP_VALUES.get(gene, set()):
        if known.lower() == raw.lower():
            return {"gene": gene, "input": raw, "lookup_keys": [known],
                    "lookup_key": known, "phenotype": known,
                    "interpreted_as": "phenotype or activity score",
                    "ambiguous": False}

    # 4. A phenotype NAME for a gene whose recommendations are keyed by activity
    #    score. One name can span several scores with different guidance, so all
    #    are returned and the result is flagged ambiguous.
    for name, keys in (_PHENOTYPE_KEYS.get(gene) or {}).items():
        if name.lower() == raw.lower():
            usable = [k for k in keys if k in _LOOKUP_VALUES.get(gene, set())]
            if not usable:
                break
            return {
                "gene": gene, "input": raw, "lookup_keys": usable,
                "lookup_key": usable[0] if len(usable) == 1 else None,
                "phenotype": name,
                "interpreted_as": ("phenotype name" if len(usable) == 1 else
                                   f"phenotype name spanning activity scores "
                                   f"{', '.join(usable)}"),
                "ambiguous": len(usable) > 1,
                "note": (None if len(usable) == 1 else
                         (f"CPIC indexes {gene} by activity score, and "
                          f"'{name}' covers {', '.join(usable)}, which can carry "
                          f"different recommendations. Supply the diplotype or "
                          f"activity score for a single answer.")),
            }

    return {
        "gene": gene, "input": raw,
        "error": f"'{raw}' is not a diplotype, phenotype or activity score "
                 f"CPIC recognises for {gene}",
        "example_diplotypes": valid_diplotypes(gene, limit=8),
        "valid_phenotypes": sorted(_PHENOTYPE_KEYS.get(gene) or {}),
        "valid_lookup_keys": valid_phenotypes(gene),
    }


def recommend(drug_name: str, genotypes: dict[str, str]) -> dict:
    """Return CPIC recommendations for a drug given a partial genotype.

    `genotypes` maps gene -> diplotype / phenotype / allele status.
    """
    data = load()
    if not data.get("recommendations"):
        return {"available": False,
                "error": "CPIC tables not built. Run: python -m CORE_AI.build_cpic"}

    drug_id = _resolve_drug_id(drug_name)
    if drug_id is None:
        return {"available": False, "drug": drug_name,
                "error": f"CPIC has no prescribing guideline for '{drug_name}'",
                "note": ("CPIC covers drugs with published pharmacogenomic dosing "
                         "guidance. Absence here is not evidence of safety.")}

    resolved, errors = {}, []
    for gene, value in (genotypes or {}).items():
        r = resolve_genotype(gene, value)
        if "error" in r:
            errors.append(r)
        else:
            resolved[r["gene"]] = r

    matched, incomplete = [], []
    guidelines = data.get("guidelines") or {}
    for rec in data["recommendations"]:
        if rec.get("drugid") != drug_id:
            continue
        needed = rec.get("lookupkey") or {}
        if not needed:
            continue

        missing = [g for g in needed if g not in resolved]
        if missing:
            incomplete.append({"requires_genes": sorted(needed),
                               "missing_genes": sorted(missing)})
            continue

        if not all(str(needed[g]).lower() in
                   [k.lower() for k in resolved[g]["lookup_keys"]]
                   for g in needed):
            continue

        gl = guidelines.get(str(rec.get("guidelineid")), {})
        matched.append({
            "genes": sorted(needed),
            "phenotypes": {g: resolved[g]["phenotype"] for g in needed},
            "matched_lookup_keys": {g: str(needed[g]) for g in needed},
            "activity_score": (rec.get("activityscore") or {}) or None,
            "recommendation": rec.get("drugrecommendation"),
            "implications": list((rec.get("implications") or {}).values()),
            "classification": rec.get("classification"),
            "population": (rec.get("population") or "").strip() or "general",
            "comments": (rec.get("comments") or "").strip(),
            "dosing_information": bool(rec.get("dosinginformation")),
            "alternate_drug_available": bool(rec.get("alternatedrugavailable")),
            "guideline": gl.get("name"),
            "guideline_url": gl.get("url"),
            "source": "CPIC",
        })

    matched.sort(key=lambda m: -CLASSIFICATION_RANK.get(m["classification"], 0))

    # Deduplicate the "needs more genes" notes.
    seen, needs = set(), []
    for item in incomplete:
        key = tuple(item["missing_genes"])
        if key not in seen:
            seen.add(key)
            needs.append(item)

    return {
        "available": True,
        "drug": drug_name,
        "cpic_drug": (data.get("drugs") or {}).get(drug_id),
        "resolved_genotypes": resolved,
        "ambiguous_inputs": [r for r in resolved.values() if r.get("ambiguous")],
        "recommendations": matched,
        "needs_additional_genotype": needs if not matched else [],
        "input_errors": errors,
        "strongest_classification": matched[0]["classification"] if matched else None,
    }


if __name__ == "__main__":
    import sys

    # CPIC recommendation text contains characters such as U+2265 ">=", which a
    # cp1252 Windows console cannot encode. Flask responses are UTF-8 and are
    # unaffected, but printing here would raise UnicodeEncodeError.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    cases = [
        ("clopidogrel", {"CYP2C19": "*2/*2"}),
        ("clopidogrel", {"CYP2C19": "*1/*1"}),
        ("abacavir", {"HLA-B": "*57:01 positive"}),
        ("abacavir", {"HLA-B": "*57:01 negative"}),
        ("codeine", {"CYP2D6": "Ultrarapid Metabolizer"}),
        ("azathioprine", {"TPMT": "*3A/*3A"}),
        ("capecitabine", {"DPYD": "Poor Metabolizer"}),
        ("aspirin", {"CYP2C19": "*1/*1"}),
    ]
    if len(sys.argv) >= 3:
        cases = [(sys.argv[1], {sys.argv[2]: sys.argv[3]})]

    for drug, gt in cases:
        res = recommend(drug, gt)
        print()
        print(f"{drug}  {gt}")
        if not res.get("available"):
            print(f"   {res.get('error')}")
            continue
        for e in res["input_errors"]:
            print(f"   input error: {e['error']}")
            if e.get("valid_phenotypes"):
                print(f"     phenotypes: {', '.join(e['valid_phenotypes'][:6])}")
            if e.get("example_diplotypes"):
                print(f"     e.g. {', '.join(e['example_diplotypes'][:4])}")
        for a in res.get("ambiguous_inputs") or []:
            print(f"   note: {a['note']}")
        for m in res["recommendations"]:
            score = m["matched_lookup_keys"]
            print(f"   [{m['classification']}] ({m['population']}) {score}")
            print(f"     -> {m['recommendation']}")
        if not res["recommendations"] and not res["input_errors"]:
            print(f"   no matching recommendation; "
                  f"needs {res.get('needs_additional_genotype')}")
