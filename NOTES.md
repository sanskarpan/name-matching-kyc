# Notes

## 1. Run it

From the repository root, with Python 3.9 or newer installed. No packages or
build tools are required; all implementations use the standard library.

```bash
python3 -m name_match.cli generate   # data/name_pairs.csv
python3 -m name_match.cli train      # data/model.json
python3 -m name_match.cli evaluate   # prints all five algorithms' comparison
python3 -m name_match.cli report     # reports/results.md and results.json
python3 -m unittest discover -s tests -t . -v
```

The first four steps together are `python3 -m name_match.cli all`.
For the complete submission check, including the comparison and tests:

```bash
python3 run.py all --test
```

The pipeline takes approximately 15–25 seconds and the full suite several
minutes. `run.py` resolves paths against the repository root, even when invoked
by absolute path from another directory. Dataset, model and reports are
committed; `all` regenerates them deterministically on the same Python version.
Floating-point serialization can differ slightly between runtime versions.
The bootstrap uses a fixed random seed. `requirements.txt` contains comments only, so installation is
optional. Read **[reports/results.md](reports/results.md)** for every comparison
and [reports/results.json](reports/results.json) for exact, unrounded thresholds.

Optional feature ablation, which refits the model for every feature:

```bash
python3 -m name_match.cli ablate
```

It writes `reports/ablation.md` and `reports/ablation.json`.

## 2. Implementations and dataset

Five approaches share `score(a, b) -> 0..1`: exact normalised token equality,
token-set Jaccard, phonetics plus Jaro-Winkler, a hand-weighted token alignment,
and logistic regression over 23 features. Implementing the primitives and
model from scratch makes the decisions inspectable and keeps the clean-checkout
workflow dependency-free. The trade-off is speed and responsibility for testing
those primitives. The Metaphone encoder is a simplified variant.

The dataset contains **494 pairs: 140 matches and 354 non-matches**, constructed
from **91 identities**. Labels follow the declared identity, never a similarity
score. Each CSV row includes person IDs, category, difficulty and a rationale.
The generator checks duplicate string pairs and identity/label consistency.

Positives cover initials, surname-first ordering, transliteration, dropped or
reordered middle names, relationship qualifiers, honorifics, suffixes, compound
surnames, typos and formatting. Eighteen transliteration pairs use spellings
absent from the lexicon; these test lexicon coverage, not an independent model
holdout. Of the negatives, **340 (96%) are deliberately hard**: common surnames,
partial overlap, phonetic collisions, near-duplicates, siblings, parent/child,
dropped tokens and seven pairs of different people with identical names.

## 3. The metric

A **false positive** flags a different person's name as a match. Used for direct
approval, it risks accepting the wrong identity. A **false negative** misses a
genuine match and creates customer friction or rejection. They should not be
weighted equally without specifying the downstream action.

The binary comparison models **triage into a review queue**, with illustrative
cost `C = 1*FP + 25*FN`: extra review costs 1, while missing a genuine customer
costs 25. This deliberately prioritises recall. It does **not** say fraud is
cheaper than customer friction; direct approval requires a separate precision
constraint. The sensitivity sweep uses `c_FP:c_FN = 1:1` through `1:100` and does
not cover a policy in which false positives cost more.

Cost per pair compares algorithms under this shared assumption, even when their
optimal thresholds achieve different recalls. Precision at a recall floor is a
complementary comparison independent of that cost choice:
`P(R) = max{precision(t): recall(t) >= R}`. It uses attainable thresholds and is
non-increasing because raising the floor shrinks the feasible threshold set.
PR-AUC is tie-grouped **average precision**. Accuracy alone would reward always
predicting non-match (72%); F1 does not encode the chosen business costs.

## 4. Results and recommendation

The learned arm uses five-fold out-of-fold scores over all 494 pairs.

| algorithm | P@R=0.90 | P@R=0.99 | PR-AUC | cost/pair | FP @ recall≥0.90 |
|---|---|---|---|---|---|
| **Learned combiner** | **0.838** | **0.695** | **0.895** | **0.174** | **25** |
| Token-aligned weighted scorer | 0.702 | 0.310 | 0.801 | 0.397 | 54 |
| Phonetic + Jaro-Winkler | 0.764 | 0.340 | 0.824 | 0.421 | 39 |
| Token-set Jaccard | 0.463 | 0.283 | 0.706 | 0.717 | 152 |
| Exact (normalised tokens) | 0.283 | 0.283 | 0.544 | 0.717 | 354 |

**Choose the learned combiner**: lowest observed triage cost at every tested
ratio and highest precision at 99% recall. Token alignment is second on baseline
cost; the phonetic matcher is second on high-recall precision.

The proposed three-band policy, measured on the pooled OOF scores, is:

| band | threshold (rounded) | pairs | observed correctness |
|---|---|---|---|
| auto-approve | score ≥ 0.919 | 40 (8.1%) | 100% |
| manual review | between cut-offs | 199 (40.3%) | — |
| auto-reject | score < 0.053 | 255 (51.6%) | 100% |

Use exact JSON cut-offs when reproducing these counts. Approval targets at least
99% observed precision; rejection excludes no observed positive. These are
sample-derived boundaries, not guarantees. The CLI reports the policy; it does
not deploy an identity-verification service. Before production, independently
validate thresholds for the model refitted on all rows. If the model cannot be
loaded, use token alignment for review prioritisation and disable automatic
decisions; its best approval precision misses the 99% target.

The paired cost bootstrap gives **+0.2227** (runner-up minus learned), with 95%
interval **[-0.0102, +0.4777]**. It includes zero: superiority on cost is not
established, and neither is equivalence. The other metric differences are
observations too; no significance test for them is supplied.

## 5. The edge case I'm proudest of

**"Krishna Kumar" vs "Krishnan Kumar"**, a declared father and son, is a
non-match despite the shared surname and one-character difference. Keeping
`krishna` out of the `krishnan` transliteration class prevents normalisation
from making them identical. Exact now scores **0.000**, Jaccard **0.333**.

Phonetic + Jaro-Winkler still scores **0.968** and token alignment **0.926**;
both falsely match at their evaluated binary thresholds. The learned OOF score
is about **0.839**, also a false positive at its 90%-recall threshold, but below
its approval cut-off and therefore routed to review by the proposed policy.
Name similarity alone cannot establish the family relationship or identity.

## 6. What I'd do next

1. Collect independently labelled data and evaluate identities held out from
   training; choose thresholds on validation data, then report untouched test
   results. Current folds split pairs, so identities can recur across folds.
2. Measure score calibration and approval precision uncertainty. A score of
   0.919 is not evidence of a calibrated 91.9% match probability; 40 observed
   correct approvals do not establish a 99% population precision guarantee.
3. Investigate the learned model's 1 middle-name and 2 held-out-transliteration
   misses at the 90%-recall operating point. Use the optional ablation to assess
   redundant features, then verify any changes on independent data.
4. Add another identity signal and evaluate review capacity. Identical names
   cannot be resolved by a name-only matcher or by a human reading names alone.

The generator, normaliser and scorer were developed together, so even OOF
results may be optimistic. Thresholds are selected on pooled scores, and the
row bootstrap holds them fixed and ignores shared identities. No size of that
bias is measured. Latin script is supported; unreadable scripts yield a neutral
score with an explicit undecidable flag. In production that flag must force
review regardless of the numerical threshold. OCR is outside the assignment.
