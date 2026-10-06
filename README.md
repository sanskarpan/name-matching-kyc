# Name matching across identity documents

A coding exercise: decide whether two name strings, extracted from two identity
documents, belong to the same person.

Five matchers, a generated dataset of 494 labelled pairs, a cost-sensitive
evaluation, and a written recommendation. **Standard library only** — Python
3.9+, no dependencies, nothing to install.

## Run it

```bash
python3 run.py all --test  # regenerate, print comparison, report, run tests
```

`run.py all` rewrites `data/name_pairs.csv`, `data/model.json`,
`reports/results.md` and `reports/results.json` from scratch. The pipeline
is deterministic on the same Python version: same input, byte-identical
output, fixed bootstrap seed. Floating-point serialization can differ slightly
between Python versions.

**Start with [reports/results.md](reports/results.md)** — every table is there.

For the reasoning rather than the numbers, read
[NOTES.md](NOTES.md).

## The short version

I matched names five ways, built a dataset where the labels come from who the
person *is* rather than from how similar the strings look, and measured the
matchers with an explicit review-queue cost: unnecessary review costs 1,
while missing a genuine match costs 25. Direct identity approval requires a
separate precision constraint; these triage costs do not price fraud acceptance.

The learned combiner has the best observed triage cost: 25 false positives
at the 90%-recall floor against token alignment's 54, and
0.695 precision at 99% recall against 0.310. The interesting part is not that it
wins, it's *why the second-place algorithm is worth keeping anyway*, and the one
case where the system still gets it wrong after two separate fixes.

## The five matchers

| | what it is | why it's here |
|---|---|---|
| Exact (normalised) | set equality after cleaning | the control for the effect of normalisation |
| Token-set Jaccard | order-invariant token overlap | what production pipelines actually ship |
| Phonetic + Jaro-Winkler | phonetic codes plus character similarity | the "robust to spelling" answer, and the one that's confidently wrong |
| Token-aligned scorer | hand-weighted, positional token roles | the strongest thing you can build without training one |
| Learned combiner | logistic regression over 23 features | the recommendation |

## Headline

Precision at a fixed recall, defined as the standard PR envelope
`P(R) = max{ precision(r) : r >= R }` — every value below is attained by a real
threshold, so no row can rise with recall:

| algorithm | P@R=0.90 | P@R=0.99 | PR-AUC | cost/pair | FP @ recall≥0.90 |
|---|---|---|---|---|---|
| **Learned combiner** | **0.838** | **0.695** | **0.895** | **0.174** | 25 |
| Token-aligned weighted scorer | 0.702 | 0.310 | 0.801 | 0.397 | 54 |
| Phonetic + Jaro-Winkler | 0.764 | 0.340 | 0.824 | 0.421 | 39 |
| Token-set Jaccard | 0.463 | 0.283 | 0.706 | 0.717 | 152 |
| Exact (normalised tokens) | 0.283 | 0.283 | 0.544 | 0.717 | 354 |

**Choose the learned combiner.** Its proposed bands on OOF scores are auto-approve
at 0.919 (8.1% of pairs, all correct in this sample), auto-reject below 0.053
(51.6%, all correct in this sample), and review between them. Exact cut-offs
are in the JSON report. Independently validate thresholds for the refitted
model before production. If loading fails, use token alignment to prioritise
review and disable automatic decisions.

Seven of the 25 remaining false positives are pairs of people with
byte-identical names. No name-only matcher will ever get those right, and it's
the strongest argument for adding a date of birth or a document number.

The paired bootstrap on cost between the top two comes out at +0.2227 per pair
with a 95% interval of [-0.0102, +0.4777]. It spans zero, so superiority on
cost is not established; that does not prove equivalence. The recommendation
also considers observed high-recall precision. Folds split pairs rather than
identities and thresholds are tuned on pooled scores, so this synthetic-data
evaluation does not establish generalisation to unseen customers.

## Layout

| Path | |
|---|---|
| `name_match/normalize.py` | name cleaning: diacritics, honorifics, `S/o` qualifiers, a partial transliteration lexicon |
| `name_match/string_algos.py` | Levenshtein, Damerau-OSA, Jaro, Jaro-Winkler, n-gram overlap |
| `name_match/phonetics.py` | Soundex and a Metaphone variant |
| `name_match/algorithms.py` | the five matchers, one `score(a, b) -> 0..1` interface |
| `name_match/dataset.py` | the identity model and the deterministic pair generator |
| `name_match/features.py` | 23 features for the learned arm, symmetric in both arguments |
| `name_match/model.py` | L2 logistic regression, batch gradient descent, strict artefact validation |
| `name_match/evaluate.py` | cost, PR-AUC, precision-at-recall, paired bootstrap, bands |
| `name_match/ablation.py` | leave-one-feature-out — which features are actually valuable |
| `name_match/cli.py` | `generate` / `train` / `evaluate` / `report` / `all` / `ablate` |
| `data/name_pairs.csv` | the dataset, with a commented header naming every category |
| `reports/ablation.md` | what each feature is worth (optional, ~2 min to generate) |
| `tests/` | unit, regression, documentation and complete-pipeline checks |

`python3 -m name_match.cli ablate` writes `reports/ablation.md`. It's slow
because it refits the model once per feature, so it isn't part of `all`.
