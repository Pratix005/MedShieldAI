# MedShield AI

**Pharmacogenomic evidence lookup for prescribed drugs.**

Given a drug name, MedShield reports the curated drug–gene associations recorded
for it in PharmGKB, with each finding's evidence level, phenotype category
(toxicity / efficacy / dosage / metabolism) and whether it is backed by a
published dosing guideline or a regulatory drug label.

> **Research and educational prototype. Not a medical device and not clinical
> advice.** Findings are drug–gene associations from published literature. They
> do not account for an individual's genotype, dose, comorbidities or other
> medicines. Never change a prescription without a clinician.

---

## Why this is a lookup tool, not a prediction model

Clinical pharmacogenomics is a curated-evidence field. CPIC, PharmGKB and DPWG
publish drug–gene guidance derived from clinical studies, and production
decision-support tools look that guidance up. Nothing infers it from molecular
structure, because it is not inferable from molecular structure: that CYP2C19
poor metabolisers underrespond to clopidogrel is a clinical finding, not a
property you can read off a SMILES string.

This project previously inverted that. A small neural network over six RDKit
descriptors produced the displayed risk, and the measured result was what you
would expect:

| | held-out ROC-AUC |
|---|---|
| gene identity only (no chemistry) | 0.587 ± 0.027 |
| gene + molecular descriptors | 0.653 ± 0.083 |

Evaluated with real PharmGKB "not associated" negatives and a drug-disjoint
split, so no drug appears in both train and test. Chemistry contributes a real
but small amount (+0.066 AUC); the model heavily overfits (train 0.96 against
test 0.65); and four of its eight genes have essentially no negatives in the
data, so it learns "always associated" for them. That is not a basis for a
clinical display, so the model is reported in a clearly-labelled exploratory
panel and never sets the verdict.

The curated knowledge base is the product.

## What the knowledge base covers

Built from the PharmGKB extracts in `CORE_AI/data/`:

- **182 drug–gene pairs** across **48 genes** and 135 drugs
- Evidence levels **1A / 1B / 2A / 2B** only (1A = in a dosing guideline or drug
  label, with replicated evidence)
- Phenotype category per finding, which is what carries clinical meaning:
  *toxicity* ("may cause harm") reads differently from *efficacy* ("may not work")
- Guideline and drug-label backing flags

It covers the genes that matter most clinically — **DPYD** (fluoropyrimidine
toxicity), **HLA-B** (abacavir hypersensitivity, allopurinol and carbamazepine
severe cutaneous reactions), **UGT1A1**, **NUDT15**, **TPMT**, **CYP2D6**,
**CYP2C19**, **SLCO1B1**, **G6PD**, **MT-RNR1**, **RYR1/CACNA1S** — most of
which were absent from the eight genes the model knows.

## Genotype-specific guidance (CPIC)

Supply a genotype and the tool returns the actual prescribing recommendation for
it, quoted verbatim from CPIC:

```
Plavix + CYP2C19 *2/*2
  -> [Strong] Avoid clopidogrel if possible. Use prasugrel or ticagrelor at
     standard dose if no contraindication.

Ziagen + HLA-B *57:01 positive
  -> [Strong] Abacavir is not recommended

codeine + CYP2D6 Ultrarapid Metabolizer
  -> [Strong] Avoid codeine use because of potential for serious toxicity.
```

`CORE_AI/build_cpic.py` fetches CPIC's tables (api.cpicpgx.org) into
`data/cpic.json`: 2,159 recommendations, 19 genes, 111,970 diplotypes.
`CORE_AI/cpic.py` matches a genotype to the right row. Input may be a diplotype
(`*2/*2`), a phenotype (`Poor Metabolizer`) or an activity score (`0.5`).

Three behaviours worth knowing:

- **Multi-gene guidelines are not answered from half a genotype.** CPIC's
  thiopurine guideline is keyed on TPMT *and* NUDT15; given only TPMT, the tool
  reports NUDT15 as missing rather than guessing.
- **Ambiguous phenotype names are not silently resolved.** CPIC indexes CYP2D6
  and DPYD by activity score, and one phenotype name spans several. DPYD "Poor
  Metabolizer" is score 0.0 *or* 0.5 — avoid the drug entirely, or halve the
  dose. Both are returned, labelled, with the ambiguity stated.
- **An unrecognised diplotype is rejected, not defaulted.** A typo must not
  resolve to a phenotype.

The risk band stays **drug-level** and is reported as such (`band_scope`). A
patient with a normal genotype can correctly see a High band next to "use at
standard dose": the band says an association exists for the drug, while CPIC
says what to do for this person. Conflating them would be wrong.

## Design decisions that matter

**Absence of evidence is reported as `Unknown`, never `Low`.** A drug with no
annotation at level 1A–2B yields "No curated evidence found", with explicit text
that this is not evidence of safety. A grey band, not a green one. This is
enforced by a test.

**No drug is recommended as an alternative.** The former "Safer Drug
Alternatives" panel chose drugs whose gene list did not intersect the flagged
gene. For rivaroxaban flagged on CYP3A4 it suggested warfarin — a different
class, different indication, different monitoring, worse bleeding profile.
Choosing an alternative needs indication and patient context this project does
not hold, and it is the output a user is most likely to act on, so it was
removed rather than patched.

**Drug names resolve through PubChem, including brand names and salts.** A name
is resolved to a PubChem CID, and if that CID is a marketed salt its parent
compound is used: `Plavix` → CID 115366 (clopidogrel bisulfate) → parent CID
60606 (clopidogrel) → the CYP2C19 finding. Combination products are split and
each component looked up, so `ezetimibe / simvastatin` still surfaces
simvastatin's SLCO1B1 association.

**No chemistry is hard-coded.** An earlier version shipped a hand-written SMILES
table in which 24 of 30 entries were the wrong molecule (warfarin at 164 Da
against a true 308 Da) and four were not parseable SMILES at all. Structures now
come from PubChem, are validated with RDKit before being cached, and the cache
is keyed on canonical names with negative caching.

## Layout

```
medshield_ai/
  CORE_AI/
    knowledge_base.py        query the curated KB (lookup, combinations, banding)
    build_knowledge_base.py  build data/knowledge_base.json from the PharmGKB TSVs
    cpic.py                  genotype -> CPIC prescribing recommendation
    build_cpic.py            fetch CPIC tables into data/cpic.json
    drug_database.py         PubChem name -> SMILES / CID / parent CID, cached
    featurize.py             RDKit descriptors + gene one-hot; model domain check
    train_association.py     honest retrain: real negatives, drug-disjoint split,
                             gene-only baseline
    train_real_model.py      original training script (superseded; see above)
    data/                    PharmGKB extracts + generated knowledge_base.json
  WEB_APP/
    app.py                   Flask API; curated verdict + exploratory model panel
    templates/index.html     UI
  tests/
    test_knowledge_base.py   12 tests, including clinical correctness assertions
    test_cpic.py             15 tests asserting published CPIC guidance
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r medshield_ai/requirements.txt
```

`torch` must come from the CPU index — the default PyPI wheel on Windows pulls
several GB of CUDA libraries this project never uses:

```bash
.venv/Scripts/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0
```

Build the knowledge base, then run:

```bash
cd medshield_ai
../.venv/Scripts/python -m CORE_AI.build_knowledge_base
../.venv/Scripts/python -m CORE_AI.build_cpic
../.venv/Scripts/python tests/test_knowledge_base.py
../.venv/Scripts/python tests/test_cpic.py
cd WEB_APP && ../../.venv/Scripts/python app.py      # http://127.0.0.1:5000
```

## API

`POST /predict` with `{"drug": "Xeloda"}`:

```json
{
  "overall_risk": "High",
  "resolved": {"label": "capecitabine", "cid": 60953, "matched_on": "PubChem CID 60953"},
  "curated": {
    "available": true,
    "findings": [{
      "gene": "DPYD", "evidence_level": "1A",
      "categories": ["Toxicity"], "meaning": "Risk of an adverse reaction",
      "in_guideline": true, "on_drug_label": true
    }],
    "source": "PharmGKB clinical annotations, evidence levels 1A-2B"
  },
  "model": {"available": true, "rows": [], "caveat": "Exploratory..."},
  "disclaimer": "Research and educational prototype..."
}
```

Add a genotype to get prescribing guidance for it:

```json
{"drug": "Plavix", "genotypes": {"CYP2C19": "*2/*2"}}
```

```json
{"cpic": {"available": true, "cpic_drug": "clopidogrel",
          "recommendations": [{"classification": "Strong",
            "recommendation": "Avoid clopidogrel if possible. Use prasugrel or ticagrelor...",
            "phenotypes": {"CYP2C19": "Poor Metabolizer"}, "population": "CVI ACS PCI"}]}}
```

`GET /api/genes` returns the genes the knowledge base covers versus the eight
the model supports. `GET /api/genotype-options?gene=CYP2C19` returns the
diplotypes, phenotypes and activity scores CPIC accepts for a gene — the
notation differs per gene (star alleles for CYP2C19, allele status for HLA-B,
HGVS variants plus activity scores for DPYD).

## Known limitations

- Genotype guidance covers the **19 genes CPIC publishes recommendations for**.
  The curated knowledge base covers 48 genes; the other 29 have documented
  associations but no CPIC prescribing guideline, so no action text exists.
- CPIC populations matter: several drugs have different guidance per indication
  (clopidogrel differs for ACS/PCI versus non-ACS). All matching populations are
  returned, labelled; the tool does not pick one, because it does not know the
  indication.
- `data/cpic.json` is a dated snapshot (see `_fetched`). CPIC revises
  guidelines; re-run `build_cpic` to refresh.
- The PharmGKB extracts in `data/` are a point-in-time snapshot with no version
  stamp. A production system would pin and refresh them.
- 19 of 135 knowledge-base drugs have no PubChem CID (mostly multi-component or
  non-compound entries), so those match by name only.
- The ML model remains weak and overfit; see the table above. Improving it needs
  substructure features (Morgan fingerprints) and genes that have negatives, not
  tuning.
