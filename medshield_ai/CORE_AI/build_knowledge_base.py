# CORE_AI/build_knowledge_base.py
"""Build the pharmacogenomic knowledge base from the PharmGKB extracts.

Why this exists
---------------
Clinical pharmacogenomics is a curated-evidence field, not a prediction
problem. CPIC, PharmGKB and DPWG publish drug-gene guidance derived from
clinical studies; production decision-support tools look that guidance up.
They do not infer it from molecular descriptors, and neither should this
project: a model over six RDKit descriptors cannot know that CYP2C19 poor
metabolisers underrespond to clopidogrel.

So the knowledge base is the product, and it is assembled from two files:

  clinicalvariants.tsv  gene, drug, PharmGKB level of evidence, phenotype
                        category (Toxicity / Efficacy / Dosage / Metabolism-PK)
  relationships.tsv     gene-chemical pairs with the annotation types backing
                        them, including GuidelineAnnotation (a published dosing
                        guideline) and LabelAnnotation (named on a regulatory
                        drug label) - the two strongest clinical tiers.

Each drug is additionally resolved to a PubChem CID so lookups survive brand
names, salts and synonyms.

Evidence levels (PharmGKB): these grade how well ESTABLISHED an association is,
not how harmful it is.
  1A  in a dosing guideline or drug label, with replicated evidence
  1B  strong evidence, not yet in a guideline
  2A  moderate, in a known pharmacogene
  2B  moderate
  3   single study / preliminary
  4   case report or inference only

Run:  python -m CORE_AI.build_knowledge_base            (actionable tiers only)
      python -m CORE_AI.build_knowledge_base --all      (include levels 3 and 4)
      python -m CORE_AI.build_knowledge_base --no-cid   (skip PubChem CID pass)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd

from CORE_AI.drug_database import get_cid, flush, normalise

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")
CLINICAL_TSV = os.path.join(DATA_DIR, "clinicalvariants.tsv")
RELATIONSHIPS_TSV = os.path.join(DATA_DIR, "relationships.tsv")
OUT_JSON = os.path.join(DATA_DIR, "knowledge_base.json")

ACTIONABLE_LEVELS = ("1A", "1B", "2A", "2B")
LEVEL_RANK = {"1A": 6, "1B": 5, "2A": 4, "2B": 3, "3": 2, "4": 1, "": 0}

# Phenotype categories, worst-first. Toxicity outranks efficacy because a
# toxicity association is the one that can harm a patient who takes the drug.
CATEGORY_PRIORITY = ("Toxicity", "Dosage", "Efficacy", "Metabolism/PK", "Other")

SALT_SUFFIXES = (
    "sodium", "potassium", "calcium", "magnesium", "hydrochloride", "hcl",
    "sulfate", "sulphate", "maleate", "tartrate", "mesylate", "besylate",
    "citrate", "acetate", "phosphate", "succinate", "fumarate", "bromide",
    "chloride", "nitrate", "oxalate", "tosylate", "malate", "aspartate",
    "bisulfate", "dihydrate", "monohydrate",
)


def canonical_drug(name: str) -> str:
    """Lowercase, collapse whitespace, drop trailing salt/hydrate words."""
    parts = normalise(name).split()
    while len(parts) > 1 and parts[-1] in SALT_SUFFIXES:
        parts.pop()
    return " ".join(parts)


def _split(cell: str) -> list[str]:
    return [p.strip() for p in (cell or "").split(",") if p.strip()]


def load_clinical() -> pd.DataFrame:
    df = pd.read_csv(CLINICAL_TSV, sep="\t", dtype=str).fillna("")
    df["drug"] = df["chemicals"].apply(_split)
    df = df.explode("drug").dropna(subset=["drug"])
    df["gene"] = df["gene"].str.upper().str.strip()
    df = df[(df["gene"] != "") & (df["drug"] != "")]
    return df.rename(columns={"level of evidence": "level",
                              "type": "category",
                              "phenotypes": "phenotypes"})


def load_relationships() -> pd.DataFrame:
    """Gene-chemical pairs with their backing annotation types."""
    df = pd.read_csv(RELATIONSHIPS_TSV, sep="\t", dtype=str).fillna("")
    mask = (((df.Entity1_type == "Gene") & (df.Entity2_type == "Chemical")) |
            ((df.Entity1_type == "Chemical") & (df.Entity2_type == "Gene")))
    gc = df[mask].copy()
    is_gene_first = gc.Entity1_type == "Gene"
    gc["gene"] = gc.Entity1_name.where(is_gene_first, gc.Entity2_name).str.upper().str.strip()
    gc["drug"] = gc.Entity2_name.where(is_gene_first, gc.Entity1_name).str.strip()
    return gc[["gene", "drug", "Evidence", "Association", "PK", "PD"]]


def build(include_all_levels: bool = False, resolve_cids: bool = True) -> dict:
    clinical = load_clinical()
    rel = load_relationships()

    if not include_all_levels:
        clinical = clinical[clinical["level"].isin(ACTIONABLE_LEVELS)]

    # Index the relationship evidence by (gene, canonical drug).
    backing: dict[tuple[str, str], dict] = {}
    for r in rel.itertuples(index=False):
        key = (r.gene, canonical_drug(r.drug))
        slot = backing.setdefault(key, {"evidence": set(), "association": set(),
                                        "pk": False, "pd": False})
        slot["evidence"].update(_split(r.Evidence))
        if r.Association:
            slot["association"].add(r.Association)
        slot["pk"] = slot["pk"] or (r.PK == "PK")
        slot["pd"] = slot["pd"] or (r.PD == "PD")

    records: dict[str, dict] = {}
    for row in clinical.itertuples(index=False):
        drug_key = canonical_drug(row.drug)
        if not drug_key:
            continue
        entry = records.setdefault(drug_key, {
            "drug": row.drug.strip(), "cid": None, "genes": {}})

        gene_slot = entry["genes"].get(row.gene)
        cats = _split(row.category)
        if gene_slot is None:
            gene_slot = entry["genes"][row.gene] = {
                "level": row.level, "categories": [], "phenotypes": [],
                "variants": [], "in_guideline": False, "on_drug_label": False,
                "pk": False, "pd": False,
            }
        # Keep the strongest level seen for this pair.
        if LEVEL_RANK.get(row.level, 0) > LEVEL_RANK.get(gene_slot["level"], 0):
            gene_slot["level"] = row.level
        for c in cats:
            if c not in gene_slot["categories"]:
                gene_slot["categories"].append(c)
        for p in _split(row.phenotypes):
            if p not in gene_slot["phenotypes"]:
                gene_slot["phenotypes"].append(p)
        variant = (getattr(row, "variant", "") or "").strip()
        if variant and variant not in gene_slot["variants"]:
            gene_slot["variants"].append(variant)

        back = backing.get((row.gene, drug_key))
        if back:
            gene_slot["in_guideline"] = "GuidelineAnnotation" in back["evidence"]
            gene_slot["on_drug_label"] = "LabelAnnotation" in back["evidence"]
            gene_slot["pk"] = back["pk"]
            gene_slot["pd"] = back["pd"]

    # Sort each drug's genes worst-first so the UI can show the top concern.
    for entry in records.values():
        def sort_key(item):
            gene, g = item
            cat = min((CATEGORY_PRIORITY.index(c) for c in g["categories"]
                       if c in CATEGORY_PRIORITY), default=len(CATEGORY_PRIORITY))
            return (-LEVEL_RANK.get(g["level"], 0), cat, gene)
        entry["genes"] = dict(sorted(entry["genes"].items(), key=sort_key))

    if resolve_cids:
        print(f"[cid] resolving {len(records)} drug names against PubChem "
              f"(cached; ~0.25s each on a cold cache)")
        for i, (key, entry) in enumerate(sorted(records.items()), 1):
            entry["cid"] = get_cid(entry["drug"]) or get_cid(key)
            if i % 50 == 0:
                print(f"[cid]   {i}/{len(records)}")
        flush()

    pair_count = sum(len(e["genes"]) for e in records.values())
    by_cid = {str(e["cid"]): k for k, e in records.items() if e.get("cid")}
    return {
        "_source": "PharmGKB clinical variant annotations + relationships",
        "_levels_included": ("all" if include_all_levels else list(ACTIONABLE_LEVELS)),
        "_level_meaning": {
            "1A": "In a dosing guideline or drug label, replicated evidence",
            "1B": "Strong evidence, not yet in a guideline",
            "2A": "Moderate evidence, known pharmacogene",
            "2B": "Moderate evidence",
            "3": "Single study or preliminary",
            "4": "Case report or inference only",
        },
        "_caveat": ("Levels grade how well ESTABLISHED an association is, not how "
                    "severe it is. Phenotype category (Toxicity/Efficacy/Dosage) "
                    "carries the clinical meaning."),
        "_generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "_drug_count": len(records),
        "_pair_count": pair_count,
        "_gene_count": len({g for e in records.values() for g in e["genes"]}),
        "by_cid": by_cid,
        "drugs": records,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="include level 3 and 4 (preliminary) annotations")
    ap.add_argument("--no-cid", action="store_true",
                    help="skip the PubChem CID resolution pass")
    args = ap.parse_args()

    kb = build(include_all_levels=args.all, resolve_cids=not args.no_cid)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(kb, fh, indent=1, sort_keys=False)

    print()
    print(f"[ok] wrote {OUT_JSON}")
    print(f"     {kb['_pair_count']} drug-gene pairs | {kb['_drug_count']} drugs "
          f"| {kb['_gene_count']} genes | {len(kb['by_cid'])} with a CID")

    tally: dict[str, int] = {}
    for e in kb["drugs"].values():
        for g in e["genes"].values():
            tally[g["level"]] = tally.get(g["level"], 0) + 1
    print("     by level: " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))

    gtally: dict[str, int] = {}
    for e in kb["drugs"].values():
        for gene in e["genes"]:
            gtally[gene] = gtally.get(gene, 0) + 1
    print("     top genes: " + ", ".join(
        f"{g}({n})" for g, n in sorted(gtally.items(), key=lambda kv: -kv[1])[:10]))
