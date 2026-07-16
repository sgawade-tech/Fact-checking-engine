# KG Fact-Checking Engine

A fact-checking engine that scores an arbitrary `(subject, predicate, object)`
fact with a **veracity value in `[0, 1]`** (0 = false, 1 = true), evaluated
against the **KG-2022** dataset (reified N-Triples fact-validation triples
over DBpedia entities/relations, in the SWC2017 Task 2 "Fact Validation"
format).

**Result:** 83.5% accuracy / 0.91 ROC-AUC via 5-fold cross-validation on the
training data, vs. 55–57% for majority-class / predicate-prior baselines.

## Project structure

```
kg-fact-checking-engine/
├── data/
│   ├── KG-2022-train_nt.txt      # 1,234 labelled facts (675 true / 559 false)
│   └── KG-2022-test_nt.txt       # 1,342 unlabelled facts to score
├── src/
│   ├── parse_nt.py               # reified N-Triples -> tidy DataFrame
│   ├── features.py                # Pool (structural) + LabelContext (KB evidence) features
│   ├── kge.py                     # optional TransE-style embedding classifier
│   ├── engine.py                  # FactCheckingEngine (fit / predict_veracity)
│   ├── evaluate.py                # baselines + 5-fold CV benchmarking
│   ├── ablation.py                # feature-block ablation study
│   ├── predict_test.py            # fits on full train, scores all test facts
│   ├── demo.py                    # FactChecker.check(subject, predicate, object) API
│   └── make_gerbil_answers.py     # converts predictions to a GERBIL-KBC answer file
├── notebooks/
│   └── KG_Fact_Checking_Engine.ipynb   # everything above, in one notebook (pre-executed)
└── outputs/
    ├── test_predictions.csv            # veracity score per test fact
    └── KG-2022-test-answers.nt         # ready-to-upload GERBIL-KBC submission file
```

## Setup

```bash
git clone https://github.com/sgawade-tech/Fact-checking-engine.git
cd Fact-checking-engine
pip install -r requirements.txt
```

Requires Python 3.9+. The two data files are already included under `data/`.

## Usage

Run any script from inside `src/` (or from the repo root — paths resolve
either way):

```bash
cd src

python parse_nt.py           # parse + sanity-print the two files
python evaluate.py           # baselines + 5-fold CV benchmark (prints metrics)
python ablation.py           # feature-block ablation study
python predict_test.py       # fits on full train, writes ../outputs/test_predictions.csv
python make_gerbil_answers.py  # writes ../outputs/KG-2022-test-answers.nt
python demo.py                 # a few worked fact-checking examples
```

Or check a fact interactively:

```python
from demo import FactChecker
fc = FactChecker()
fc.check("Barack_Obama", "spouse", "Michelle_Obama")
# -> {'veracity': ..., 'evidence': [...]}
```

Or open `notebooks/KG_Fact_Checking_Engine.ipynb` — it's self-contained (no
imports from `src/`) and already executed, so results are visible without
running anything.

## Method

Two complementary evidence sources are extracted from the train+test files
themselves (no external KG access was used):

1. **`Pool`** — label-free structural features (subject/object/predicate
   co-occurrence frequency), computed from the full train+test file. Safe
   to use in full since only the *existence* of triples is used, never
   their truth labels (standard transductive KG-embedding setting).
2. **`LabelContext`** — features derived only from known-labelled facts:
   does this exact triple already appear as true/false elsewhere? Does the
   same `(subject, predicate)` already have a different confirmed-true
   object? Subject/object/predicate true-rates. Built with proper
   leave-one-out counting (`collections.Counter`) so a row's own label
   never leaks into its own features.
3. **`kge.py`** (optional, off by default) — a from-scratch TransE-style
   embedding model, trained with a logistic loss directly on the real
   true/false labels. An ablation study showed it doesn't help at this
   dataset size (~1,200 facts / ~2,000 mostly-singleton entities), so it's
   disabled by default but kept as a toggle (`use_kge=True`) for larger,
   denser graphs.

These features feed a **Logistic Regression** meta-classifier with
isotonic calibration (`engine.py`), chosen over Histogram Gradient
Boosting after comparison (gradient boosting overfits badly at this
sample size — see `ablation.py` / the notebook for numbers).

## Results (5-fold stratified CV on train)

| model | accuracy | ROC-AUC | F1 | Brier |
|---|---|---|---|---|
| majority-class baseline | 54.7% | 0.500 | 70.7% | 0.248 |
| predicate-prior baseline | 57.1% | 0.597 | 67.6% | 0.239 |
| **final engine (logreg, pool+label)** | **83.5%** | **0.912** | **85.6%** | **0.116** |

**Ablation** (which evidence matters):

| feature block | accuracy | ROC-AUC |
|---|---|---|
| predicate one-hot only | 57.1% | 0.594 |
| pool (popularity) only | 66.7% | 0.745 |
| label-derived (KB evidence) only | 77.8% | 0.865 |
| **pool + label (no KGE) — shipped default** | **83.5%** | **0.912** |
| KGE only | 54.8% | 0.563 |
| pool + label + KGE | 82.1% | 0.894 |

`team` is consistently the weakest relation (~67% accuracy / 0.69 AUC) —
its false candidates are near-miss same-sport substitutions that are hard
to rule out from `(s,p,o)` structure alone.

## Submitting to GERBIL-KBC

`outputs/KG-2022-test-answers.nt` is a ready-to-upload answer file for the
[GERBIL-KBC Fact Validation task](https://gerbil-kbc.aksw.org/gerbil/config) —
one `hasTruthValue` triple per test statement, scored by ROC-AUC. Regenerate
it any time with `python src/predict_test.py && python src/make_gerbil_answers.py`.

## Limitations

- No live DBpedia/SPARQL access was used — the "knowledge base" is the
  train+test files themselves. Real entity types and a live KG could help
  further, especially for `team`.
- Benchmark numbers are a rigorous, leakage-free 5-fold CV estimate on
  train; test-set predictions can't be numerically validated without gold
  labels for test. The predicted-true rate on test (69.7%) runs
  noticeably higher than train's 54.7% base rate — flagged, not hidden;
  worth rechecking if gold labels ever become available.
- Each fact is scored independently; there's no joint constraint (e.g.
  "exactly one true object per `(subject,predicate)` group") across
  related test facts, which the data suggests would help.
