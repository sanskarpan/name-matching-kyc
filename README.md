# Name Matching Across Identity Documents

Pre-interview coding exercise. Five name-matching algorithms, a
494-pair generated labelled dataset, a cost-sensitive evaluation harness,
and a written recommendation.

**No third-party packages.** Standard library only, Python 3.9+.
351 tests, including one module that checks the numeric claims in these
documents against the code that produces them — the tables, the model
coefficients, the thresholds and the band sizes.

## Run it

```bash
python3 run.py all
```

That regenerates the dataset, trains the learned combiner, prints the full
algorithm comparison, and writes `reports/results.md` and
`reports/results.json`. About 15 seconds.

```bash
python3 run.py test        # 351 tests, 80-100s
```

See [NOTES.md](NOTES.md) for exact commands, the metric rationale, the hardest
edge case, and the final recommendation.

## What is here

| Path | What it is |
|---|---|
| `name_match/normalize.py` | Name cleaning: diacritics, honorifics, suffixes, `S/o` qualifiers, a partial transliteration lexicon |
| `name_match/string_algos.py` | Levenshtein, Damerau-OSA, Jaro, Jaro-Winkler, n-gram overlap — all from scratch |
| `name_match/phonetics.py` | Soundex and a Metaphone variant — from scratch |
| `name_match/algorithms.py` | The five matchers, all behind one `score(a, b) -> 0..1` interface |
| `name_match/dataset.py` | The identity model and the deterministic pair generator |
| `name_match/features.py` | 23 engineered features for the learned arm, symmetric in both arguments |
| `name_match/model.py` | Logistic regression with L2, batch gradient descent, JSON round-trip with strict feature validation |
| `name_match/evaluate.py` | Asymmetric cost, PR-AUC, precision-at-recall, paired bootstrap, three-band operating point |
| `name_match/cli.py` | `generate` / `train` / `evaluate` / `report` / `all` |
| `data/name_pairs.csv` | The dataset, with a commented header documenting every category |
| `data/model.json` | The trained model, coefficients readable |
| `reports/results.md` | The results report — **start here** |
| `tests/` | Primitives, dataset integrity, matcher behaviour, metrics, end-to-end gates, and a named regression test for each of the 46 defects the audit found |

## The five matchers

1. **Exact (normalised tokens)** — the control. How far does cleaning alone go?
2. **Token-set Jaccard** — the standard production baseline. Order-invariant.
3. **Phonetic + Jaro-Winkler** — the standard "robust to spelling" answer.
4. **Token-aligned weighted scorer** — hand-weighted; models order, initials,
   dropped names and joint surnames, with prefix credit falling monotonically as
   a short name's extension into a longer one grows.
5. **Learned combiner** — logistic regression over features from all of the
   above, scored out-of-fold.

## Headline

Precision at a fixed recall — the comparable metric, defined as the standard PR
envelope `P(R) = max{ precision(r) : r >= R }`, so every value is attained by a
real threshold and no row can rise with recall:

| algorithm | P@R=0.90 | P@R=0.99 | PR-AUC | cost/pair | FP @ recall≥0.90 |
|---|---|---|---|---|---|
| **Learned combiner** | **0.838** | **0.695** | **0.895** | **0.174** | 25 |
| Phonetic + Jaro-Winkler | 0.764 | 0.340 | 0.824 | 0.421 | 39 |
| Token-aligned weighted scorer | 0.702 | 0.310 | 0.801 | 0.397 | 54 |
| Token-set Jaccard | 0.463 | 0.283 | 0.706 | 0.717 | 152 |
| Exact (normalised tokens) | 0.283 | 0.283 | 0.544 | 0.717 | 354 |

Recommendation: ship the learned combiner with a three-band split at 0.919
(auto-approve) and 0.053 (auto-reject), keeping the token-aligned scorer as a
deterministic fallback. Seven of the learned model's 25 remaining false
positives are pairs of people with byte-identical names — irreducible for any
name-only matcher, and the strongest argument for adding a non-name signal.

Full numbers, thresholds, per-category failure breakdown, the cost sweep and the
confidence interval are in [reports/results.md](reports/results.md).