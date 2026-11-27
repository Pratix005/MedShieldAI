# tests/test_knowledge_base.py
"""Correctness tests for the curated pharmacogenomic knowledge base.

These assert facts from published pharmacogenomic guidance. If a change breaks
one of them, the knowledge base has become clinically wrong, which matters more
than any accuracy metric.

Run:  python -m pytest tests -q          (from medshield_ai/)
      python tests/test_knowledge_base.py   (no pytest needed)

Tests marked NETWORK need PubChem; they are skipped when it is unreachable.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CORE_AI import knowledge_base as kb
from CORE_AI.build_knowledge_base import canonical_drug


# --------------------------------------------------------------------------
# Established drug-gene associations. Each is in a CPIC/PharmGKB guideline
# and on the relevant regulatory label.
# --------------------------------------------------------------------------
KNOWN_PAIRS = [
    ("clopidogrel", "CYP2C19"),   # poor metabolisers underrespond
    ("warfarin", "VKORC1"),       # dose requirement
    ("warfarin", "CYP2C9"),
    ("abacavir", "HLA-B"),        # hypersensitivity, HLA-B*57:01
    ("capecitabine", "DPYD"),     # severe fluoropyrimidine toxicity
    ("azathioprine", "TPMT"),     # myelosuppression
    ("azathioprine", "NUDT15"),
    ("irinotecan", "UGT1A1"),     # neutropenia
    ("codeine", "CYP2D6"),        # ultrarapid metabolisers, morphine toxicity
    ("allopurinol", "HLA-B"),     # severe cutaneous reactions
    ("carbamazepine", "HLA-B"),   # SJS/TEN
]

# Toxicity associations: these must be categorised as toxicity, because that is
# what distinguishes "may not work" from "may cause harm".
TOXICITY_PAIRS = [
    ("abacavir", "HLA-B"),
    ("capecitabine", "DPYD"),
    ("azathioprine", "NUDT15"),
    ("carbamazepine", "HLA-B"),
]


def _findings(drug):
    res = kb.lookup(drug, use_network=False)
    return {f["gene"]: f for f in (res["findings"] if res else [])}


def test_kb_loads():
    assert kb.is_available(), "knowledge base empty; run build_knowledge_base"
    genes = kb.genes_covered()
    assert len(genes) >= 40, f"expected 40+ genes, got {len(genes)}"


def test_known_pairs_present():
    missing = [f"{d}/{g}" for d, g in KNOWN_PAIRS if g not in _findings(d)]
    assert not missing, f"established pairs absent from the KB: {missing}"


def test_toxicity_categorised():
    wrong = []
    for drug, gene in TOXICITY_PAIRS:
        f = _findings(drug).get(gene)
        if not f or "Toxicity" not in f["categories"]:
            wrong.append(f"{drug}/{gene} -> {f['categories'] if f else 'absent'}")
    assert not wrong, f"not categorised as toxicity: {wrong}"


def test_high_risk_drugs_band_high():
    """A guideline-backed toxicity or dosage finding must yield High."""
    for drug in ["abacavir", "capecitabine", "azathioprine", "warfarin", "irinotecan"]:
        res = kb.lookup(drug, use_network=False)
        assert res, f"{drug} not found"
        band = kb.assess(res["findings"])["band"]
        assert band == "High", f"{drug} banded {band}, expected High"


def test_unannotated_is_unknown_not_low():
    """The critical safety property: no evidence must never read as 'Low'."""
    verdict = kb.assess([])
    assert verdict["band"] == "Unknown", (
        f"empty findings banded {verdict['band']}; must be Unknown so that "
        f"absence of evidence is not presented as safety")
    assert "not evidence of safety" in verdict["reason"].lower()


def test_fluorouracil_not_mapped_to_tpmt():
    """Regression: the old hand-written map claimed fluorouracil -> TPMT.

    Fluorouracil is a DPYD substrate. TPMT covers the thiopurines.
    """
    f = _findings("fluorouracil")
    assert "TPMT" not in f, "fluorouracil must not be associated with TPMT"
    if f:
        assert "DPYD" in f, "fluorouracil should carry its DPYD association"


def test_canonical_drug_strips_salts():
    assert canonical_drug("Warfarin Sodium") == "warfarin"
    assert canonical_drug("clopidogrel bisulfate") == "clopidogrel"
    assert canonical_drug("  CODEINE  ") == "codeine"
    assert canonical_drug("acetaminophen") == "acetaminophen"


def test_salt_form_matches_parent_offline():
    """Salt suffixes resolve without needing the network."""
    assert kb.lookup("warfarin sodium", use_network=False), \
        "warfarin sodium should match warfarin by name canonicalisation"


def test_split_combination():
    parts = kb.split_combination("ezetimibe / simvastatin")
    assert any("simvastatin" in p.lower() for p in parts)
    assert any("ezetimibe" in p.lower() for p in parts)


def test_combination_finds_annotated_component_offline():
    res = kb.lookup_with_components("ezetimibe / simvastatin", use_network=False)
    genes = {f["gene"] for m in res["matches"] for f in m["findings"]}
    assert "SLCO1B1" in genes, (
        f"simvastatin's SLCO1B1 association must survive the combination "
        f"split; got {genes}")


def test_evidence_levels_valid():
    bad = []
    for entry in (kb.load().get("drugs") or {}).values():
        for gene, g in entry["genes"].items():
            if g.get("level") not in ("1A", "1B", "2A", "2B"):
                bad.append(f"{entry['drug']}/{gene}={g.get('level')}")
    assert not bad, f"unexpected evidence levels: {bad[:5]}"


# ---- NETWORK ----
def test_brand_name_resolves_via_cid():
    """Plavix -> clopidogrel bisulfate -> parent clopidogrel."""
    try:
        res = kb.lookup("Plavix")
    except Exception as exc:                      # pragma: no cover
        print(f"  SKIP (network): {exc}")
        return
    if res is None:
        print("  SKIP: PubChem unreachable or returned no CID")
        return
    assert res["canonical_name"] == "clopidogrel", res["canonical_name"]
    assert "CYP2C19" in {f["gene"] for f in res["findings"]}


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}\n      {exc}")
        except Exception as exc:
            failed += 1
            print(f"ERROR {name}\n      {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
