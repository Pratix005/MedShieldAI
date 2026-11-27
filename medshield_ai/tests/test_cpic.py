# tests/test_cpic.py
"""Tests for the genotype -> prescribing guidance layer.

These assert published CPIC guidance. A failure here means the tool would give
wrong dosing advice, which is the most serious way this project can be wrong.

Run:  python tests/test_cpic.py        (no pytest needed)
      python -m pytest tests -q
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from CORE_AI import cpic
from CORE_AI.build_cpic import encode_diplotypes, decode_diplotypes


def _recs(drug, genotypes):
    return cpic.recommend(drug, genotypes)


def test_tables_load():
    assert cpic.is_available(), "CPIC tables missing; run build_cpic"
    assert len(cpic.genes_covered()) >= 15


def test_abacavir_hla_b_5701_positive_is_contraindicated():
    """The single most clear-cut pharmacogenomic contraindication."""
    r = _recs("abacavir", {"HLA-B": "*57:01 positive"})
    assert r["available"], r.get("error")
    assert r["recommendations"], "no recommendation returned"
    top = r["recommendations"][0]
    assert top["classification"] == "Strong", top["classification"]
    assert "not recommended" in top["recommendation"].lower(), top["recommendation"]


def test_abacavir_hla_b_5701_negative_is_standard():
    r = _recs("abacavir", {"HLA-B": "*57:01 negative"})
    assert r["recommendations"]
    text = r["recommendations"][0]["recommendation"].lower()
    assert "standard" in text, text


def test_clopidogrel_poor_metaboliser_avoids_clopidogrel():
    r = _recs("clopidogrel", {"CYP2C19": "*2/*2"})
    assert r["recommendations"], r
    assert r["resolved_genotypes"]["CYP2C19"]["phenotype"] == "Poor Metabolizer"
    top = r["recommendations"][0]
    assert "avoid" in top["recommendation"].lower(), top["recommendation"]
    assert top["classification"] == "Strong"


def test_clopidogrel_normal_metaboliser_is_standard_dose():
    r = _recs("clopidogrel", {"CYP2C19": "*1/*1"})
    assert r["recommendations"]
    assert "standard dose" in r["recommendations"][0]["recommendation"].lower()


def test_codeine_ultrarapid_is_avoided():
    """Ultrarapid CYP2D6 metabolisers convert codeine to morphine too fast."""
    r = _recs("codeine", {"CYP2D6": "Ultrarapid Metabolizer"})
    assert r["recommendations"], r
    assert any("avoid" in m["recommendation"].lower() for m in r["recommendations"])


def test_capecitabine_dpyd_zero_activity_is_avoided():
    r = _recs("capecitabine", {"DPYD": "0.0"})
    assert r["recommendations"], r
    assert "avoid" in r["recommendations"][0]["recommendation"].lower()


def test_dpyd_phenotype_name_is_flagged_ambiguous():
    """DPYD 'Poor Metabolizer' spans activity scores 0.0 and 0.5.

    Those carry different guidance, so the input must not resolve to one of
    them silently.
    """
    r = _recs("capecitabine", {"DPYD": "Poor Metabolizer"})
    assert r["ambiguous_inputs"], "ambiguity not reported"
    note = r["ambiguous_inputs"][0]["note"]
    assert "activity score" in note.lower()
    keys = {m["matched_lookup_keys"]["DPYD"] for m in r["recommendations"]}
    assert {"0.0", "0.5"} <= keys, keys


def test_azathioprine_requires_both_tpmt_and_nudt15():
    """CPIC's thiopurine guideline is keyed on both genes.

    With only TPMT supplied, the tool must say what is missing rather than
    answer from half a genotype.
    """
    partial = _recs("azathioprine", {"TPMT": "*3A/*3A"})
    assert not partial["recommendations"]
    missing = {g for item in partial["needs_additional_genotype"]
               for g in item["missing_genes"]}
    assert "NUDT15" in missing, partial["needs_additional_genotype"]

    full = _recs("azathioprine", {"TPMT": "*3A/*3A", "NUDT15": "*1/*1"})
    assert full["recommendations"], full
    assert "alternative" in full["recommendations"][0]["recommendation"].lower()


def test_unknown_diplotype_is_rejected_not_defaulted():
    """A typo must not silently resolve to a phenotype."""
    r = cpic.resolve_genotype("CYP2C19", "*999/*999")
    assert "error" in r, r
    assert r.get("lookup_keys") is None


def test_unknown_drug_is_reported_not_guessed():
    r = _recs("not_a_real_drug_xyz", {"CYP2C19": "*1/*1"})
    assert not r["available"]
    assert "no prescribing guideline" in r["error"].lower()
    assert "not evidence of safety" in r["note"].lower()


def test_drug_without_guideline_says_so():
    """Absence of a CPIC guideline must not read as reassurance."""
    r = _recs("rivaroxaban", {"CYP2C19": "*1/*1"})
    if not r["available"]:
        assert "not evidence of safety" in r["note"].lower()


def test_salt_and_case_forms_resolve():
    for name in ["Clopidogrel", "clopidogrel bisulfate", "  clopidogrel  "]:
        r = _recs(name, {"CYP2C19": "*2/*2"})
        assert r["available"], f"{name} did not resolve"
        assert r["recommendations"], name


def test_diplotype_encoding_is_lossless():
    flat = {
        "FAKE1": {"*1/*1": ["Normal", "Normal"],
                  "*1/*2": ["Intermediate", "1.0"],
                  "*2/*2": ["Poor", "0.0"]},
        "FAKE2": {"a/a": ["X", "X"]},
    }
    assert decode_diplotypes(encode_diplotypes(flat)) == flat


def test_recommendation_text_is_never_empty_when_matched():
    for drug, gt in [("abacavir", {"HLA-B": "*57:01 positive"}),
                     ("clopidogrel", {"CYP2C19": "*2/*2"}),
                     ("codeine", {"CYP2D6": "*1/*1"})]:
        for m in _recs(drug, gt)["recommendations"]:
            assert (m["recommendation"] or "").strip(), (drug, gt)
            assert m["classification"], (drug, gt)


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
