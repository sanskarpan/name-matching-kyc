# Notes

Name matching across identity documents. Five matchers, a generated dataset, a
cost-sensitive evaluation, and a recommendation.

Everything in the tables below is produced by the code in this repo. Where I
quote a number it is quoted from a run, and there is a test that fails if the
number and the code disagree.

---

## 1. Run it

Python 3.9 or newer. That is the whole dependency list — no third-party
packages anywhere.

```bash
python3 -m name_match.cli generate   # -> data/name_pairs.csv
python3 -m name_match.cli train      # -> data/model.json
python3 -m name_match.cli evaluate   # prints the comparison for all five
python3 -m name_match.cli report     # -> reports/results.md, results.json
python3 -m unittest discover -s tests -t .   # 349 tests (~4-5 min)
```

The first four in one go:

```bash
python3 -m name_match.cli all
```

The test suite is the slow part (~4-5 minutes) because it re-runs the whole
pipeline several times over.

`run.py all` and `run.py test` do the same thing and work from any directory.
`requirements.txt` exists but is empty on purpose, so `pip install -r
requirements.txt` is a clean no-op instead of an error.

Outputs:

| Path | What |
|---|---|
| `data/name_pairs.csv` | the generated dataset, with a commented header |
| `data/model.json` | the trained weights |
| `reports/results.md` | every table — **start here** |
| `reports/results.json` | the same numbers, machine-readable |
| `reports/ablation.md` | optional, see §8 |

Run `all` before the tests. Until the dataset and reports exist, eight tests
skip rather than fail — including two of the regression tests. If you want to
be sure nothing skipped, run with `-v` and check the summary says just `OK`.

Optional, and slow (~2 minutes, it refits the model once per feature):

```bash
python3 -m name_match.cli ablate
```

---

## 2. What's here

Five matchers, all behind one `score(a, b) -> 0..1` interface:

1. **Exact (normalised tokens)** — the control. Set equality after cleaning.
   Answers "how far does cleaning alone get us", without which the rest of the
   comparison means nothing. It gets 28% of pairs right.
2. **Token-set Jaccard** — what most KYC pipelines actually ship. Order
   invariant, trivial to implement. Included as the realistic baseline rather
   than a straw man. It cannot handle an initial: `{s, krishnan}` against
   `{srinivasan, krishnan}` shares one of three elements, 0.333.
3. **Phonetic + Jaro-Winkler** — the "robust to spelling variation" answer, and
   the one that is confidently wrong (§6).
4. **Token-aligned weighted scorer** — hand-weighted. Positional token roles,
   initials as partial credit, coverage as the penalty for dropped tokens, the
   family name weighted highest, phonetics discounted so they can bridge a gap
   but never manufacture a match on their own.
5. **Learned combiner** — logistic regression over 23 features drawn from all of
   the above, scored out-of-fold.

**Why five when three would have done.** The fourth and fifth aren't extra
accuracy, they're diagnostic resolution. Exact + Jaccard + phonetic can't tell
you whether a matcher is uniformly mediocre or excellent except on one failure
mode, and the per-category tables in `reports/results.md` are only actionable
because the hand-built scorer and the model fail on *different* categories.
Five is where a sixth would have duplicated a failure mode instead of exposing
a new one.

**Why from scratch.** The brief allows libraries. I didn't use any, because half
the bugs I found were in the normaliser and the metrics, and I'd rather own
those than guess what a library does at a boundary. It's also about 670 lines.
The honest cost is speed: the pipeline takes ~15 seconds, and a library version
would be well under one. At 494 rows that doesn't matter.

---

## 3. The dataset

494 pairs, 140 positives and 354 negatives, from a 91-person roster. 340 of the
negatives (96%) are deliberately hard.

The one decision everything else rests on: **labels come from identity, not from
similarity.** Two strings are a match if and only if the generator knows they
render the same person. No matcher, threshold or heuristic is consulted. This
matters more than it sounds — if labels were derived from a similarity
heuristic, every algorithm would be scored against its own opinion of what a
match is, and the whole exercise becomes circular. It's enforced mechanically:
`build_dataset` raises if any row ends up labelled inconsistently, and a test
checks that the same two strings never appear twice with different labels.

Positives, by shape:

| category | n | what it is |
|---|---|---|
| `initials` | 14 | given name reduced to an initial (`S. Kumar` / `Suresh Kumar`) |
| `surname_first` | 14 | family name written first (`Kumar Suresh`) |
| `transliteration` | 14 | two valid spellings (`Mohammed`/`Mohammad`) |
| `transliteration_heldout` | 18 | a variant the shipped lexicon has never seen |
| `middle_name` | 14 | dropped, reordered, or replaced by `S/o` / `W/o` / `D/o` |
| `honorific` | 14 | a title on one document only |
| `suffix` | 14 | `Jr.` / `II` |
| `compound_surname` | 10 | `K-Singh` written joined |
| `typo` | 14 | one or two character errors |
| `formatting` | 14 | case, spacing, punctuation only |

The `transliteration_heldout` group is the one I'd point at if you only look at
one. Eighteen of the transliteration positives use a spelling variant that is
*not* in the shipped lexicon — otherwise "generalisation" would just mean "the
lexicon worked". A test asserts this against the real normaliser rather than
against a comment, because an earlier version of the dataset claimed it in a
comment and two of the variants were in fact lexicon members.

Negatives are the interesting half:

| shape | n | why it's hard |
|---|---|---|
| `common_surname` | 137 | two unrelated people sharing a family name |
| `partial_overlap` | 73 | one person's full name, another's with a token missing |
| `phonetic` | 67 | the names sound alike |
| `near_duplicate` | 36 | share a surname *and* part of a given name |
| `identical_name` | 7 | byte-identical, different people |
| `sibling` | 6 | provably share a father and a family name |
| `sibling_reordered` | 6 | the same, one document surname-first |
| `parent_child` | 4 | declared parent/child, shared family name |
| `dropped_token` | 4 | one side is a strict subset of the other |

Negative rows are verified by shape, not by intent: a row is only filed as
`sibling` if two roster members actually declare the same father, and only as
`dropped_token` if one side's tokens really are a subset of the other's.

Those 7 identical-name pairs are the hard floor. No name-only matcher can
separate them, and all five get all seven wrong.

---

## 4. The metric

**A false positive is a fraudster's document accepted as belonging to a real
customer.** Unrecoverable: wrong decision recorded, regulatory exposure, real
account-takeover risk.

**A false negative is a genuine customer sent to a manual review queue.**
Friction, an extra document upload, a support ticket. Recoverable — nothing is
destroyed and no wrong decision is recorded.

Those aren't equally bad, so `C = c_FP * FP + c_FN * FN` with `c_FN = 25`.
The number is a judgement, not a measurement, so I swept it from 1:1 to 1:100
and the recommendation doesn't change: the learned combiner is cheapest at every
ratio in the sweep.

Accuracy is out — the set is 28% positive, so "always predict non-match" scores
72%. F1 is out — it weights the two errors equally, which is exactly the
assumption I'm rejecting.

**But I don't actually rank on the cost table**, and this is the part I'd most
want challenged. At `c_FN = 25` each algorithm's cost-optimal threshold ends up
at a slightly different recall (the learned combiner lands at 0.993, the exact
control at 1.000). Five different operating points, so the costs aren't
comparable to each other.

So the metric I rank on is **precision at a fixed recall**: every algorithm
answers the identical question, whatever its scores look like. It's the standard
PR envelope, `P(R) = max{precision(r) : r >= R}`, so every published value is
attained by a real threshold and no row can rise with recall. I originally used
linear interpolation here and it reported precisions no threshold could attain,
including a row that *increased* with recall. That's a fixed bug with a test.

---

## 5. Results

Scored out-of-fold, over the full 494 pairs.

| algorithm | P@R=0.90 | P@R=0.99 | PR-AUC | cost/pair | FP @ recall≥0.90 |
|---|---|---|---|---|---|
| **Learned combiner** | **0.838** | **0.695** | **0.895** | **0.174** | **25** |
| Token-aligned weighted scorer | 0.702 | 0.310 | 0.801 | 0.397 | 54 |
| Phonetic + Jaro-Winkler | 0.764 | 0.340 | 0.824 | 0.421 | 39 |
| Token-set Jaccard | 0.463 | 0.283 | 0.706 | 0.717 | 152 |
| Exact (normalised tokens) | 0.283 | 0.283 | 0.544 | 0.717 | 354 |

Per-category breakdown, thresholds, the cost sweep and the bootstrap are all in
`reports/results.md`.

**One thing that doesn't go my way.** The paired bootstrap on cost between the
learned combiner and the runner-up comes out at `+0.2227` per pair with a 95%
interval of `[-0.0102, +0.4777]`. That interval spans zero, so on cost alone the
two are not statistically separable on 494 rows. I'd rather say that than round
it away. What the recommendation rests on instead is the false-positive counts
at a matched recall (25 against 54) and the high-recall precision (0.695 against
0.310), both of which sit further from noise than the cost difference does.

Worth knowing: an earlier version of this dataset gave a *confidently
significant* interval. It then turned out that the `surname_first` rows all also
abbreviated the given name, so ordering was never tested on its own terms, and
the hand-built scorer couldn't be failed at it. Fixing that destroyed the
significance. The significance was partly an artefact of a category that couldn't
be failed.

---

## 6. The edge case I'm proudest of

**`"Krishna Kumar"` vs `"Krishnan Kumar"` — a father and a son, sharing a surname.**

Before any of the fixes below, the exact matcher scored it **1.000**. A false
positive on the single most dangerous category in the whole exercise: two people
in one household with a family name. The transliteration lexicon was supposed
to prevent this, and didn't.

The cause was a lexicon collision. `krishna` and `krishnan` were both listed as
members of the `krishnan` class, so normalisation collapsed them to one token
and set equality said yes.

| algorithm | before | after | what happened |
|---|---|---|---|
| Exact (normalised) | 1.000 | 0.000 | lexicon had already merged them — that *was* the bug; now correctly binary |
| Token-set Jaccard | 1.000 | 0.333 | same root cause |
| Phonetic + Jaro-Winkler | 0.906 | 0.968 | **worse, and unfixable by normalisation.** They're near-homophones, so the phonetic signal is actively wrong here |
| Token-aligned scorer | 1.000 | 0.926 | needed a second fix; see below |

Two separate bugs, one wrong answer, both in the *normalisation* layer above the
matcher. That's why this is the case I'd lead with rather than a cleverer scoring
function: the thing most worth reviewing here isn't the scorer, it's the
representation that everything downstream is built on.

**Where it still fails, honestly.** 0.926 is not low enough to be comfortable,
and it's *higher* than the 0.822 the same person scores as `"Krishnan Kumar"` vs
`"Krishnan K. Kumar"`. Nothing in the strings separates a father from a son:
`venkat`/`venkatesh` is one person and `krishna`/`krishnan` is two, and they're
structurally identical. I could have tuned it into submission by adding a
constant. I left it, because the honest version of this edge case is that
name-only matching cannot fully resolve it, and a tuned number would hide the
one thing a reviewer most needs to know.

---

## 7. What I'd ship

**The learned combiner, with the token-aligned scorer as a documented fallback.**

Ship it at the three-band operating point:

| band | threshold | pairs | accuracy |
|---|---|---|---|
| auto-approve | score ≥ 0.919 | 40 (8.1%) | 100% |
| manual review | in between | 199 (40.3%) | — |
| auto-reject | score < 0.053 | 255 (51.6%) | 100% |

The auto-reject cut-off is deliberately the *lowest score any genuine match
reached*, so a real customer is never silently thrown away on this data.

**Fallback:** if the model artefact can't be loaded, use the token-aligned
scorer. Not because it's close — it's second on cost but third on P@R=0.99 —
but because it's deterministic, has no fitted state, and can't fail the way an
artefact on disk can. Degrade to review, never to auto-approve.

**What this answer depends on:**

- **Review-queue budget is the binding constraint.** Raising the auto-approve
  cut-off above 0.919 removes no false positive (that band is already 100%
  correct) but does push genuine matches into review. A budget decision, not a
  model decision.
- **The cost ratio moves toward 1:1.** The learned combiner still wins, but the
  margin narrows — at 1:1 it's 0.067 against the phonetic matcher's 0.107, at
  1:100 it's 0.200 against the token-aligned scorer's 0.650. Note that even the
  *runner-up* changes across the sweep. At some ratio I'd start arguing the
  simpler algorithm's operational cost is worth the difference.
- **Real data has a different negative distribution.** The ranking is partly a
  property of *my* negatives, and mine are plausibly easier than production's.
  Re-fit and re-measure.
- **Name-only matching stops being sufficient.** This is the big one. 7 of the
  25 remaining false positives are the byte-identical-name pairs, and every
  algorithm gets all seven wrong. A DOB or a document number removes that entire
  error class. Given one, name matching is a supporting signal, not the decision.
- **Compound-surname recall becomes a measured problem.** It's the only category
  where the hand-built scorer beats the model: it misses 7 of the 10
  joint-surname matches where the model misses 1.

---

## 8. What I'd do next

1. **Get a second, independently specified dataset.** This is the first thing.
   I wrote the generator and the matchers, so I can't tell how much of the
   performance is real and how much is me recognising my own edge cases. I'd
   hand someone else the roster and the category list, not my code, and not look
   at the result until the design is frozen.
2. **Calibrate the model.** The 0.919 auto-approve threshold is used as if it
   meant "91.9% chance this is a match". Logistic regression on 494 rows is not
   calibrated and nothing here measures whether it is. A reliability diagram and
   an expected-calibration-error number would tell me. This is the gap I'm least
   comfortable with, because the whole three-band configuration rests on it.
3. **Fix the middle-name recall.** 4 of its 14 false negatives are middle names
   that every other matcher catches. That's a missing feature, not a capacity
   problem.
4. **Drop the features that aren't earning their place.**
   `reports/ablation.md` shows `length_ratio` carrying one of the model's
   largest weights while removing it makes out-of-fold cost slightly *better*.
   With 23 features on 494 rows that's inside the noise, but it's a feature I
   would delete. The same file shows `token_dice` with a *negative* weight is the
   single most valuable feature — a coefficient table on its own would never
   have told me that.
5. **Character-level embeddings for the held-out transliterations.** fastText
   subwords, or a small byte-level model. The held-out set is the generalisation
   test and 3 of its 18 pairs still miss.

---

## Known limits

- **The dataset is synthetic, and I wrote it.** Every caveat above follows from
  this. Labels are structural rather than guessed, but the *distribution* is mine.
- **Names are clean strings.** The brief puts the extraction layer out of scope,
  which means the dataset doesn't test it — and that's where a surprising share
  of real production errors actually live.
- **Latin script only.** The brief doesn't ask for anything else, but worth
  knowing: the normaliser folds to `[a-z0-9]`, so a name it can't read is
  reported as undecidable and scores a neutral 0.5 rather than a confident 0.0.
  That routes to human review instead of auto-rejecting a real customer.
- **The cross-validation threshold is chosen on the full dataset**, so the
  reported confidence interval is very slightly optimistic. Nested selection
  would cost more folds than a dataset this size can afford.
- **7 of the negatives are literally unsolvable.** Byte-identical names,
  different people. No name-only matcher will ever get those right.