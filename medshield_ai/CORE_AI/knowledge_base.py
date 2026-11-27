# CORE_AI/knowledge_base.py
"""Query the curated pharmacogenomic knowledge base.

This is the part of MedShield that can be trusted, because every row traces to
a PharmGKB clinical annotation with a stated evidence level. The ML model is a
separate, clearly-labelled exploratory signal; it never overrides anything here.

Lookup order for a drug name:
  1. canonical name (lowercased, salt/hydrate suffix removed)
  2. PubChem CID, which is what makes brand names and salt forms resolve
     ("Plavix" and "clopidogrel bisulfate" both land on clopidogrel)

Build the data file first:  python -m CORE_AI.build_knowledge_base
"""

from __future__ import annotations

import json
import os
from typing import Optional

from CORE_AI.build_knowledge_base import canonical_drug
from CORE_AI.drug_database import get_cid, get_parent_cid

_HERE = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(_HERE, "data", "knowledge_base.json")

# Clinical significance of a finding. This is driven by the phenotype category,
# NOT by the evidence level: a level-1A "Efficacy" finding means the drug may
# not work, while a level-1A "Toxicity" finding means it may cause harm. Those
# warrant different words.
CATEGORY_MEANING = {
    "Toxicity": "Risk of an adverse reaction",
    "Dosage": "Dose adjustment may be required",
    "Efficacy": "Drug may be less effective or ineffective",
    "Metabolism/PK": "Altered drug metabolism or exposure",
    "Other": "Documented pharmacogenomic association",
}

LEVEL_RANK = {"1A": 6, "1B": 5, "2A": 4, "2B": 3, "3": 2, "4": 1}

_KB: dict = {}
_LOADED = False


def load(path: str = KB_PATH) -> dict:
    global _KB, _LOADED
    if _LOADED:
        return _KB
    try:
        with open(path, "r", encoding="utf-8") as fh:
            _KB = json.load(fh)
        print(f"[kb] {_KB.get('_pair_count', 0)} curated drug-gene pairs, "
              f"{_KB.get('_gene_count', 0)} genes, "
              f"levels {_KB.get('_levels_included')}")
    except (OSError, json.JSONDecodeError) as exc:
        _KB = {}
        print(f"[warn] knowledge base unavailable ({exc}). "
              f"Run: python -m CORE_AI.build_knowledge_base")
    _LOADED = True
    return _KB


def is_available() -> bool:
    return bool(load().get("drugs"))


def lookup(drug_name: str, use_network: bool = True) -> Optional[dict]:
    """Return the knowledge-base entry for a drug, or None if not annotated.

    None means "no curated pharmacogenomic annotation at the included evidence
    levels". It does NOT mean the drug is safe.
    """
    kb = load()
    drugs = kb.get("drugs") or {}
    if not drugs:
        return None

    key = canonical_drug(drug_name)
    if key in drugs:
        return _decorate(key, drugs[key], matched_on="name")

    by_cid = kb.get("by_cid") or {}
    cid = get_cid(drug_name, use_network=use_network)
    if cid is not None:
        mapped = by_cid.get(str(cid))
        if mapped and mapped in drugs:
            return _decorate(mapped, drugs[mapped], matched_on=f"PubChem CID {cid}")

        # A brand name resolves to the marketed salt; guidance is written about
        # the active moiety, so fall back to the parent compound.
        parent = get_parent_cid(cid, use_network=use_network)
        if parent is not None and parent != cid:
            mapped = by_cid.get(str(parent))
            if mapped and mapped in drugs:
                return _decorate(mapped, drugs[mapped],
                                 matched_on=f"PubChem CID {cid} -> parent {parent}")
    return None


def _decorate(key: str, entry: dict, matched_on: str) -> dict:
    findings = []
    for gene, g in entry.get("genes", {}).items():
        cats = g.get("categories") or ["Other"]
        findings.append({
            "gene": gene,
            "evidence_level": g.get("level", ""),
            "categories": cats,
            "meaning": "; ".join(CATEGORY_MEANING.get(c, c) for c in cats),
            "phenotypes": g.get("phenotypes", []),
            "variants": g.get("variants", [])[:6],
            "in_guideline": bool(g.get("in_guideline")),
            "on_drug_label": bool(g.get("on_drug_label")),
            "affects_pk": bool(g.get("pk")),
            "affects_pd": bool(g.get("pd")),
            "source": "PharmGKB curated annotation",
        })
    findings.sort(key=lambda f: -LEVEL_RANK.get(f["evidence_level"], 0))
    return {
        "canonical_name": key,
        "label": entry.get("drug", key),
        "cid": entry.get("cid"),
        "matched_on": matched_on,
        "findings": findings,
        "highest_level": findings[0]["evidence_level"] if findings else "",
        "has_toxicity_finding": any("Toxicity" in f["categories"] for f in findings),
        "guideline_backed": any(f["in_guideline"] for f in findings),
        "label_backed": any(f["on_drug_label"] for f in findings),
    }


COMBINATION_SEPARATORS = (" / ", "/", " with ", " and ", " + ", "+", ",")


def split_combination(drug_name: str) -> list[str]:
    """Split a combination product into component names.

    Prescriptions are frequently combinations ("Tylenol with codeine",
    "ezetimibe / simvastatin"). Each component carries its own pharmacogenomic
    annotations, and missing the annotated one is the dangerous failure mode.
    """
    parts = [drug_name]
    for sep in COMBINATION_SEPARATORS:
        nxt = []
        for p in parts:
            nxt.extend(p.split(sep))
        parts = nxt
    seen, out = set(), []
    for p in parts:
        c = canonical_drug(p)
        # Drop brand words that are not themselves compounds, and noise tokens.
        if c and c not in seen and len(c) > 2:
            seen.add(c)
            out.append(p.strip())
    return out


def lookup_with_components(drug_name: str, use_network: bool = True) -> dict:
    """Look up a drug, falling back to its components if it is a combination.

    Returns {"matches": [entry, ...], "combination": bool, "unmatched": [...]}.
    """
    direct = lookup(drug_name, use_network=use_network)
    if direct:
        return {"matches": [direct], "combination": False, "unmatched": []}

    components = split_combination(drug_name)
    if len(components) <= 1:
        return {"matches": [], "combination": False, "unmatched": components}

    matches, unmatched = [], []
    for comp in components:
        got = lookup(comp, use_network=use_network)
        if got:
            if not any(m["canonical_name"] == got["canonical_name"] for m in matches):
                matches.append(got)
        else:
            unmatched.append(comp)
    return {"matches": matches, "combination": True, "unmatched": unmatched}


def assess(findings: list[dict]) -> dict:
    """Turn curated findings into a risk band.

    Driven by evidence level AND phenotype category. Crucially, no findings
    yields "Unknown", never "Low": the absence of a curated annotation means
    nothing has been established, not that the drug is safe for this patient.
    """
    if not findings:
        return {
            "band": "Unknown",
            "reason": ("No curated pharmacogenomic annotation at evidence levels "
                       "1A-2B. This is not evidence of safety."),
        }

    strong = [f for f in findings if f["evidence_level"] in ("1A", "1B")]
    strong_serious = [f for f in strong
                      if "Toxicity" in f["categories"] or "Dosage" in f["categories"]]
    if strong_serious:
        genes = ", ".join(f["gene"] for f in strong_serious[:4])
        return {"band": "High",
                "reason": (f"Strong evidence (level {strong_serious[0]['evidence_level']}) "
                           f"of toxicity or dose-adjustment requirement: {genes}.")}
    if strong:
        genes = ", ".join(f["gene"] for f in strong[:4])
        return {"band": "Medium",
                "reason": f"Strong evidence of altered efficacy or metabolism: {genes}."}

    genes = ", ".join(f["gene"] for f in findings[:4])
    return {"band": "Medium" if any("Toxicity" in f["categories"] for f in findings) else "Low",
            "reason": f"Moderate evidence (level 2A/2B) for: {genes}."}


def genes_covered() -> list[str]:
    kb = load()
    return sorted({g for e in (kb.get("drugs") or {}).values() for g in e.get("genes", {})})


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or ["clopidogrel", "warfarin", "abacavir", "capecitabine",
                             "azathioprine", "irinotecan", "codeine", "aspirin"]
    for n in names:
        res = lookup(n)
        print()
        if not res:
            print(f"{n}: no curated pharmacogenomic annotation at included levels")
            continue
        print(f"{n}  ->  {res['label']} (CID {res['cid']}, matched on {res['matched_on']})")
        for f in res["findings"]:
            tags = []
            if f["in_guideline"]:
                tags.append("guideline")
            if f["on_drug_label"]:
                tags.append("drug label")
            print(f"   {f['gene']:10} {f['evidence_level']:3} "
                  f"{'/'.join(f['categories']):22} {f['meaning']}"
                  + (f"  [{', '.join(tags)}]" if tags else ""))
