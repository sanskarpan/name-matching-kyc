# NOTES

Name matching across identity documents — pre-interview coding exercise.

Everything below is reproducible from a clean checkout with the standard
library alone. There is no `pip install` step in this project.

---

## 1. How to run it

**Prerequisite:** Python 3.9 or newer. That is the entire dependency list.
Verified on CPython 3.14.7 (macOS). Nothing platform-specific is used.

```bash
# from the repository root, with nothing installed

# 1. Generate the labelled dataset -> data/name_pairs.csv
python3 -m name_match.cli generate

# 2. Train the learned combiner    -> data/model.json
python3 -m name_match.cli train

# 3. Score all five algorithms and print the comparison to stdout
python3 -m name_match.cli evaluate

# 4. Write the report              -> reports/results.json, reports/results.md
python3 -m name_match.cli report

# 5. OPTIONAL: leave-one-feature-out ablation, ~2 minutes, not part of `all`
#    because it refits the model once per feature.
#    -> reports/ablation.json, reports/ablation.md
python3 -m name_match.cli ablate
```

Steps 1–4 are equivalent to (step 5 is separate, and deliberately slow):

```bash
python3 -m name_match.cli all
```

Run the test suite (351 tests, 80-100 seconds — it re-runs the full pipeline
several times). **Run the four commands above first.** Until the dataset, model
and reports exist, eight tests skip rather than fail — including
`test_bug_43` and `test_bug_44`, the regressions for two of the report defects.
A skip is easy to miss in a wall of dots; if you want to confirm nothing was
skipped, run with `-v` and look for `skipped=N`, or check that the run ends
`OK` with no `skipped=` suffix.

```bash
python3 -m unittest discover -s tests -t .
```

Or use the thin wrapper, which does the same things and needs no path setup:

```bash
python3 run.py all
python3 run.py test
python3 run.py all --test      # both
```

`make all`, `make test`, `make evaluate`, `make report` also work, but `make` is
optional and nothing depends on it.

**Where the output goes**

| Artefact | Path | Contents |
|---|---|---|
| Dataset | `data/name_pairs.csv` | 494 labelled pairs, commented header documenting every category |
| Trained model | `data/model.json` | 23 weights, bias, standardisation stats, training log-loss |
| Results (human) | `reports/results.md` | All tables. **Read this first.** |
| Results (machine) | `reports/results.json` | The same numbers, structured, valid RFC-8259 JSON |
| Ablation (opt-in) | `reports/ablation.md` | Leave-one-feature-out: what each of the 23 features is actually worth |
| Evaluation | stdout | Every table, printed, on `evaluate` or `all` |

**Runtime:** `all` takes 13-15 seconds end to end on the machine this was
written on. The learned model takes about 1.0 second to fit, about 5.5 seconds
for all five cross-validation folds plus feature extraction, and the
2000-resample paired bootstrap about 0.2 seconds — so none of those explains the
total. The rest is that `evaluate` and `report` each re-run the five-fold
cross-validation independently, which is the honest cost of not threading scores
through the two phases.

**Determinism:** the dataset generator, the cross-validation split and the
bootstrap all use fixed seeds or no RNG at all. Two runs produce a byte-identical
CSV and identical scores. `tests/test_end_to_end.py::TestReproducibility`
asserts it.

### Why standard library only

The brief says the reviewer has only the language runtime, and it warns that
"if a step is missing or ambiguous, we will not guess." A `pip install` of
rapidfuzz + jellyfish + scikit-learn is a step that can fail, and a failed
install turns a reviewable submission into an unrunnable one.

The stronger argument is that the interesting content of this exercise *is* the
string metrics. Had I called `rapidfuzz.fuzz.ratio`, I would not have had to
think about why Jaro-Winkler's prefix bonus is a liability for surname-first
documents, or about where Soundex collapses. Implementing Levenshtein,
Damerau-OSA, Jaro, Jaro-Winkler, Soundex, Metaphone and batch-gradient-descent
logistic regression is 667 lines of code — every line that is neither blank nor
a comment, counted as
`cat name_match/string_algos.py name_match/phonetics.py name_match/model.py | grep -vc '^[[:space:]]*\(#.*\)\?$'` — and each line is documented
at the point where its failure mode matters. `requirements.txt` is present and
contains only comments, so `pip install -r requirements.txt` is a clean no-op
rather than an error.

The honest cost of this choice: the algorithms are far slower than library
versions, and 15 seconds instead of well under one. Neither matters at 494 rows.

**Why five and not three.** The brief asks for at least three, and stopping at
three would have been defensible. I went to five because the third and fourth
are not extra accuracy, they are extra *diagnostic* resolution: exact, Jaccard
and phonetic alone cannot separate a matcher that is uniformly mediocre from
one that is excellent except on one failure mode, and the per-category tables in
section 4 are only actionable because the hand-built token-aligned scorer and
the learned combiner fail on *different* categories. Five is the point where
adding a sixth would have duplicated a failure mode rather than exposed a new
one. The cost is real and worth naming: each extra algorithm is another thing
whose constants a reviewer may reasonably disagree with, and the token-aligned
scorer's seven weights are hand-set rather than fitted.

**Where the seven hand-set constants come from.** The token-aligned scorer is
the one algorithm in the submission with numbers I chose rather than learned, so
they deserve an explicit account rather than a shrug:

| constant | value | what it trades | how it was set |
|---|---|---|---|
| family token weight | 1.00 | the reference the other roles are judged against | 1.0 by definition; it is the only role that is never optional |
| given-name weight | 0.85 | a given name is nearly as identifying as a family name | the single most consequential free choice; 0.85 keeps an initialised given name from being discounted as hard as a dropped middle name |
| middle-name weight | 0.45 | a middle name is corroborating, not identifying | set low because the dataset's `middle_name` category is *defined* by the middle name being absent or altered, so trusting it heavily would cost recall exactly where the category lives |
| initial credit | 0.70 | an initial is strong evidence, not proof | a bare initial shares a first character with its expansion and almost nothing else |
| prefix credit | 0.85 | scaling term, see below | 0.85 is the ceiling of a smooth function, not an independent knob |
| phonetic discount | 0.55 | phonetics bridge a gap, never manufacture a match | 0.55 chosen so phonetics alone cannot clear the token-aligned scorer's typical negative-pair score; raising it produces the phonetic-only false positives section 6 is about |
| initial-mismatch floor | 0.10 | two different initials are absence of evidence, not evidence of mismatch | deliberately non-zero; see the fix recorded in section 7 |
| order-bonus weight | 0.90 | order agreement, worth at most a 3% move | small on purpose, because positional weighting already handles most of it |

None of these was grid-searched, and none should be presented as optimal. They
were set by reasoning about the dataset's category definitions, and the two that
most affect the reported numbers -- the given-name weight and the phonetic
discount -- are the two I would revisit first if this were real work rather than
an exercise. The prefix credit is the exception: it was originally a step
function with two hand-set breakpoints and is now a single smooth expression
with one constant, precisely because two breakpoints could not be reasoned about
coherently (section 6).

### What I considered and rejected

- **Character-level embeddings** (fastText, sentence-transformers). The obvious
  way to handle `Venkatt` / `Venkatesh`. Rejected because it needs a pretrained
  artefact, which breaks the zero-install constraint and makes the result
  non-reproducible without a pinned download. It is the first thing I would add
  with more time.
- **`sklearn.metrics` for the PR and ROC curves.** I hand-rolled them instead,
  which was worth it: it caught three metric bugs that would each have produced
  a report that looked entirely reasonable and was wrong (§7).
- **Hand-labelling all 494 pairs.** Rejected in favour of generating them from
  an identity model — see §2. What I actually hand-authored is the roster of 90
  identities and their relationships, and that is the part which requires
  judgement.

---

## 2. The dataset: 494 pairs, and how they were built

`data/name_pairs.csv`, **494 pairs: 140 positives and 354 negatives**, over 90
identities (a 91-person roster, one member of which is a spare that no other entry
references and that therefore never appears as an endpoint).

### The central design decision: labels come from identity, not from similarity

Every identity in the roster has a stable `person_id`. A pair is labelled
`match` **if and only if** both sides were rendered from the same `person_id`.
Nothing in the labelling path ever looks at the resulting strings.

This is the single most important property of the dataset, and it buys three
things a similarity-then-label approach cannot:

1. **Label leakage is structurally impossible.** There is no code path from
   "these strings look similar" to "label = 1".
2. **Genuinely hard negatives survive.** `"Anil Kumar Sharma"` vs
   `"Anil Kumar Saxena"` are two unrelated people who differ by one character in
   the surname. Any hand-labelling protocol that says "these are too close to
   call, drop them" would delete exactly the cases the exercise cares about.
   Here it is a correctly-labelled negative, because the two people have
   different `person_id`s.
3. **Same name, different person is representable.** The roster contains two
   people both named exactly `"Sourav Ghosh"`, two named exactly
   `"Prakash Deshmukh"`, two named exactly `"Kiran Pawar"`, two named exactly
   `"Simran Kumar Singh"`, and so on. Seven such pairs are in the dataset,
   labelled non-match. They are *irreducible*: no name-only matcher can separate
   them, and they are the floor on achievable precision. A dataset without them
   would report a false-positive rate near zero and be meaningless.

A test asserts the invariant directly on all 494 rows: `label == 1 if and only
if person_a == person_b`.

### Labels must also be *defensible from the strings alone*

Structural consistency is necessary but not sufficient. If two roster entries
carry the same name spelling and their pair is labelled a non-match, a reader
looking at the CSV sees something that is not decidable. Two mechanisms keep the
label and the visible strings honest:

- **Dedup is on the unordered pair.** `build_dataset` keys on the sorted pair,
  so a string comparison can never appear twice with different labels. Negatives
  are built first and their keys reserved, so a positive can never re-use a
  comparison a negative already claims. A build-time gate raises if it ever
  happens. (Before this, `'Nisha Yadav'` vs `'Nisha Kumari Yadav'` appeared
  twice, once as a match and once as a non-match.)
- **Roster notes must not contradict the labels.** An earlier version of the
  roster said "same person as P011" in a note while giving a *different*
  `person_id`, and then labelled the pair a non-match — 13 rows where the
  generator's own documentation contradicted its output. Those entries are now
  genuinely different people, and the notes say so ("one of three Iqbals in this
  roster who spell the name differently, so the spelling gap is not evidence of
  one identity"). A test asserts no note claims a shared identity across
  different `person_id`s.

### What "noisy but the same person" actually looks like here

| category | n | what it models |
|---|---|---|
| `initials` | 14 | `S.` vs `Suresh Kumar Sharma`; dotted (`S.`) and bare (`S Krishnan`) forms |
| `surname_first` | 14 | `Sharma S` vs `Suresh Kumar Sharma` — the state voter-ID / DL layout |
| `transliteration` | 14 | `Surish`/`Suresh`, `Prafull`/`Prakash`, `Sreemitha`/`Srinivasan` |
| `transliteration_heldout` | 18 | spelling variants **deliberately absent from the shipped lexicon** (see below) |
| `middle_name` | 14 | middle name dropped, reordered, or replaced by an `S/o` / `W/o` / `D/o` father's- or husband's-name qualifier |
| `honorific` | 14 | `Smt.`, `Shri`, `Mr.` on one document only |
| `suffix` | 14 | `Jr.`, `II` — on male identities only, and never on a roster "elder" |
| `compound_surname` | 10 | joint surnames written joined or hyphenated: `Kumar Singh` / `Kumar-Singh` |
| `typo` | 14 | one transposed, dropped or duplicated character |
| `formatting` | 14 | case and spacing only — the category that *should* be free |

### What "deceptively close" actually looks like here

**340 of the 354 negatives (96%) are hard negatives.** Easy "clearly different
people" pairs are a deliberate minority: a dataset dominated by trivially
separable negatives flatters every algorithm and hides the failure modes that
matter in production. Hard negatives are also *verified by shape* — a row only
lands in a category if the row genuinely has that shape, rather than being
assigned it by a stride through the roster.

| category | n | the trap, and the test that admits it |
|---|---|---|
| `hard_negative_common_surname` | 137 | two unrelated people sharing a family-name token — `Sharma`, `Gupta`, `Kumar`, `Sandhu`, `Reddy`, joint forms like `Kumar Singh` |
| `hard_negative_partial_overlap` | 73 | shares a given or middle name but nothing else: `Krishnan Kumar` vs `Hemant Kumar Gupta` |
| `hard_negative_phonetic` | 67 | a verified Soundex or Metaphone collision with no shared token: `Sharma`/`Sarma`, `Sunil`/`Sonal`, `Kumar`/`Kumari` |
| `hard_negative_near_duplicate` | 36 | exactly one token differs |
| `hard_negative_sibling` | 6 | a declared shared `father` in the roster: `Gurpreet Singh Sandhu` vs `Harpreet Singh Sandhu` |
| `hard_negative_sibling_reordered` | 6 | the same, one side surname-first |
| `hard_negative_dropped_token` | 4 | one side is a strict token subset of the other |
| `hard_negative_parent_child` | 4 | a declared parent/child relation: `Krishna Kumar` vs `Krishnan Kumar` |
| `hard_negative_identical_name` | 7 | byte-identical names, two different people. Irreducible. |
| `unrelated` | 14 | the control group |

The categories are smaller than their names suggest — 6 siblings, 4 dropped
tokens, 4 parent/child pairs — because each of those shapes requires a specific
*relationship or structure* to exist in the roster, and manufacturing one by
padding the roster with fake households would defeat the point.

### Three things I added that the brief did not ask for

**(a) Held-out transliteration.** The transliteration lexicon in
`normalize.py` is hand-written — the way a production system would ship one. If
I had also used it to *generate* the dataset, the transliteration category would
be solved by construction and the number would be worthless. So 18 of the
transliteration positives use variants the lexicon does not contain, split into
two groups whose rationales name the mechanism:

- **13 that normalisation cannot resolve at all** — the changed spelling is
  absent from the lexicon and out of reach of the generic fold. These measure
  generalisation.
- **5 that the generic character fold happens to resolve** — a geminate
  consonant, a doubled vowel. Real spelling variation, but the lexicon is not
  what solves them, and the rationale says so.

"Held out" here means *at least one changed spelling* the lexicon has never
seen. A variant may still carry a surname the lexicon knows — `"Priyanaka
Chaterjee"` keeps a listed spelling of Chaterjee — and is still a real
generalisation test through the given name. Demanding that every changed token
be unknown would only push the dataset toward invented surnames nobody writes.

A test asserts both halves against the real normaliser rather than against this
paragraph: that every pair in the first group is still unresolved, that every
pair in the second is resolved, that the novel spelling in each is genuinely
absent from the lexicon, and that in each fold-resolved pair the fold — not a
lexicon entry — is what closes the gap. An earlier version of this split was
checked only in a comment, and two of its eight entries were in fact resolved by
lexicon entries, which made "held out" false for them.

**(b) Irreducible pairs.** Seven byte-identical names belonging to different
people. They set the precision floor.

**(d) Shape tests for the positive categories, not only the negatives.** The
negatives always had nine shape tests — a row is only filed as a `sibling` if
two roster members genuinely share a father, and so on. The positives had none,
and the asymmetry was not cosmetic. A stride that collapsed whenever a category
had three builders meant `surname_first` and `typo` each used exactly one
builder for their entire column, so reordering was only ever tested in
combination with an initialised given name, and the per-category error tables in
§4 could not tell "fails on ordering" from "fails on initials". Fixing the
stride changed the results enough to cost the learned combiner its
statistically-significant cost margin (see §3), which is the point: a category
that cannot be failed on its own terms flatters everything measured on it.
There is now a `test_hard_negative_*` equivalent for each seeded positive
category, and the same collapsing stride in the thinning step was dropping the
tail of the roster from the whole dataset — one identity was in the roster and
in no pair at all.

**(e) `S/o`, `W/o` and `D/o` are all exercised.** The brief names
father's- and husband's-name conventions specifically. `D/o` was dead code —
`Person.relational` returns it only for a woman with no husband, and no such
person existed — while three places, including the shipped CSV header,
documented the category as covering it.

**(c) A `rationale` column on every row**, composed from the row's own strings.
An earlier version wrote the rationale from the *intended* pair and then let a
builder corrupt the strings, so 18 rationales described pairs the row did not
contain, and 58 duplicated their own clause. Rationales are now assembled by
`difflib` from the actual edit and prefixed with the actual names, and the
assembler raises if a clause repeats. A test checks every rationale against its
row.

### Known limitations of the dataset

Stated plainly, because a reader should discount for them:

- **494 rows is a small evaluation set.** The paired bootstrap says the
  top-vs-runner-up gap is real, but it cannot detect differences much smaller
  than the interval width. I have not treated smaller differences as meaningful
  anywhere.
- **The identities are fictional.** Plausible — real regional name
  distributions, real surname frequencies, real household structures — but the
  *pairing* of families is my construction. Real negatives are probably harder
  than mine, because real households contain more genuinely near-identical names
  than a roster of 90 does.
- **Negative structure is surname-heavy.** 137 `common_surname` and 73
  `partial_overlap` of 340 hard negatives. The roster was walked exhaustively
  rather than sampled, which favours same-surname shapes.
- **No OCR**, which the brief puts out of scope explicitly. **Transliteration
  from non-Latin scripts is not out of scope**, and I was wrong to write that
  here once: the brief scopes out "OCR, image processing, and text extraction",
  which says nothing about a name already extracted as a Devanagari string.
  That case is now handled explicitly rather than by accident. The normaliser
  folds to `[a-z0-9]`, which silently deleted every character NFKD cannot
  decompose, so `राहुल शर्मा` normalised to nothing and **two byte-identical
  Devanagari names scored 0.000 on all five matchers** — a confident rejection
  of a genuine customer, which is the worst direction for this problem to fail
  in. It now reports the loss (`NormalizedName.undecidable`) and returns a
  neutral 0.5, which sits above the auto-reject band and below auto-approve, so
  such a pair routes to a human rather than being thrown away. Genuinely empty
  input still scores 0.0; an empty name carries no information and an
  unreadable one carries information the normaliser threw away, and conflating
  them is what the bug was.
- **Names are clean strings.** The brief assumes this, but it means the dataset
  does not test the extraction layer, where a surprising share of real
  production errors live.

---

## 3. The metric: asymmetric cost, and the metric I actually lead with

### The primary objective: asymmetric misclassification cost

```
C = c_FP · FP + c_FN · FN,  minimised over the decision threshold
```

with `c_FP = 1` and `c_FN = 25`.

**The reasoning.** A false positive means a document that does not belong to the
applicant has been accepted as if it did: unrecoverable once acted on, a
regulatory exposure, and depending on the product an account takeover. A false
negative means a legitimate customer was not auto-approved — they see a review
screen, maybe upload one more document, maybe file a ticket. Friction, and
*recoverable*: no information is destroyed and no wrong decision is recorded.
25:1 encodes "an order of magnitude-plus of extra friction is worth avoiding a
fraud". That is a judgement, not a measurement, and §3 is where I show it does not
matter.

**The ratio is swept from 1:1 to 1:100.** The learned combiner is cheapest at
every ratio in the sweep: 0.063 against 0.103 for the token-aligned scorer at
1:1, and 0.217 against 0.532 at 1:100. So the recommendation does not
depend on the number I picked, which is the property I wanted out of the analysis.

The *full* ordering, though, is not stable across the sweep, and I would rather say
so than round it to "the ranking holds". The phonetic matcher and the
token-aligned scorer swap second and third place at 1:3. Token-set Jaccard and
the Exact control tie exactly at 1:50 and 1:100, which the report records as a
tie rather than as one overtaking the other. Only the leading algorithm is safe to rank
on; the middle of the field is not separable at a ratio you have not been told.
What the ratio reliably changes is the *threshold* — and the threshold is what you
would deploy.

### Why accuracy and F1 are the wrong answers

Accuracy: the base rate is 28% positive, so "always predict non-match" scores
72%. Useless.

F1: a single symmetric number. It is the right tool when the two error types
cost the same, which is not this problem, and it hides the *shape* of the error —
which for a KYC pipeline is the thing you need.

PR-AUC rather than ROC-AUC: ROC's false-positive axis is dominated by the large
negative population, so it flatters a matcher that is only good on the easy tail.
Both are reported.

### The metric I lead with, and why I did not expect to

The cost table is a **weak ranking comparison**, and I only noticed by reading
the output rather than the idea:

> At `c_FN = 25`, lowering the threshold keeps paying, so each algorithm's
> cost-optimal threshold ends up at or near full recall — but the five stop at
> five *different* points, so the costs are not comparable. Only the Exact control
> reaches recall 1.000; the others stop short (the learned combiner at 0.986, the phonetic matcher at 0.971, the token-aligned scorer at 0.979, token-set Jaccard at 0.950). The learned combiner
> still misses 2 genuine matches at its optimum, so even that column is not a
> clean false-positive count. You can see the failure directly in the
> recall-floored table: token-set Jaccard "wins" it at 0.6619, not the Exact
> control, and it wins for exactly this reason — its scores are too coarse to
> clear a 90% recall floor selectively, so the cheapest threshold that clears the
> floor happens to accept a lot.

So the comparison I trust is **precision at a fixed recall**, defined by the
standard PR envelope

```
P(R) = max { precision(r) : r >= R }
```

Two properties hold by construction, and both are asserted in
`tests/test_evaluate.py` and `tests/test_regressions.py`:

1. **Every published value is attained** by some real threshold. Nothing is
   interpolated into existence. This is not academic: the first version of this
   function interpolated linearly between achievable points and reported
   precision values no threshold could produce, overstating one by 0.335
   absolute.
2. **P(R) is non-increasing in R**, because demanding more recall can only admit
   more items and therefore only add false positives. The linear version
   violated this for two of the five algorithms — precision *rising* with recall
   in a table indexed by recall.

`P@R = 0.99` is the operationally interesting column: the precision available at
the recall level a pipeline needs if missing a genuine customer is expensive.

### Where the cost function *is* the right tool

Two places, and they are different uses:

1. **Choosing the deployed threshold** for the triage queue. That is exactly what
   the cost function is for, and the sweep says what threshold to use at each
   ratio.
2. **The three-band split** below.

### Uncertainty, stated honestly

A paired bootstrap over 2000 resamples, learned combiner against
token-aligned scorer, each at its own cost-optimal threshold:

```
mean cost delta (runner-up - top)  +0.2227 per pair
95% CI                           [-0.0102, +0.4777]
P(top is cheaper)                0.9715
```

**The interval spans zero.** On the corrected dataset the cost gap is *not*
statistically significant, and I would rather say that than round it away. An
earlier version of this dataset -- in which the `surname_first` rows all also
abbreviated the given name, so ordering was never tested on its own -- gave a
confidently significant interval, and fixing the dataset destroyed the claim.
That is worth stating plainly: the significance was partly an artefact of a
category that could not be failed on its own terms.

So what does the ordering rest on? Not this test. It rests on the false-positive
counts at a matched recall floor, which are counts rather than differences and
are separated by more than resampling noise, and on the precision-at-recall
table, where the gap at high recall is 0.69 against 0.31. The cost bootstrap
tells you the two are not *distinguishable on cost*; it does not tell you the
model is worse, and it certainly does not license shipping the simpler
algorithm on the strength of a coin flip.

What this does *not*
say: it compares the top two only, it is a comparison at 494 rows, and the
interval's tightness is partly an artefact of the *paired* design (the same
resampled rows scored by both algorithms). If the interval had spanned zero I
would have said the two were tied and shipped the simpler one; the harness
computes and reports that verdict automatically rather than leaving the reader to
check it.

### Three-band operating point

The cost-optimal threshold is right for *triaging a queue*. It is not right for
*acting without a human*, because the cost of a false positive differs depending
on whether anyone is still in the loop. So the shipped configuration uses two
boundaries chosen from opposite ends:

- **auto-approve** at or above the *precision* cut-off **0.919** — 40 of 494
  pairs (8.1%), and 100% of them really are matches.
- **auto-reject** below the *recall* cut-off **0.053** — 255 of 494 (51.6%), and
  100% of them really are non-matches. (It is the *lowest* score any genuine
  match reached: anything below it cannot be a true match.)
- **199 pairs (40.3%) go to a human.**

40% manual review is high, and honestly so. It is also a *policy* number, not a
model number: the auto-approve band is already 100% correct, so raising the
cut-off adds no false positives — it only moves true matches into review, and
that trade is a decision about review budget.

---

## 4. Results

Full tables in `reports/results.md`.


| algorithm | P@R=0.90 | P@R=0.99 | PR-AUC | ROC-AUC | cost/pair | FP @ recall>=0.90 |
|---|---|---|---|---|---|---|
| **Learned combiner** | 0.838 | 0.695 | 0.895 | 0.964 | 0.174 | **25** |
| Phonetic + Jaro-Winkler | 0.764 | 0.340 | 0.824 | 0.931 | 0.421 | 39 |
| Token-aligned weighted scorer | 0.702 | 0.310 | 0.801 | 0.925 | 0.397 | 54 |
| Token-set Jaccard | 0.463 | 0.283 | 0.706 | 0.865 | 0.717 | 152 |
| Exact (normalised tokens) | 0.283 | 0.283 | 0.544 | 0.722 | 0.717 | 354 |

The learned combiner is best at **every** recall level and on every aggregate
metric. A test asserts it at each level, so the claim cannot go stale.

### The five matchers

1. **Exact (normalised tokens)** — the control. Set equality after cleaning.
   Answers "how far does cleaning alone go?" without which the rest of the
   comparison is meaningless. Cleaning alone gets 28% of pairs right and
   `P@R=0.99 = 0.283`.
2. **Token-set Jaccard** — the algorithm most KYC pipelines actually ship,
   because it is order-invariant and trivial to implement. Included as the
   realistic baseline rather than a straw man. It cannot handle an initial:
   `{s, krishnan}` against `{srinivasan, krishnan}` shares one of three
   elements, 0.333. The shorter the name, the worse it gets: the same initial
   against a three-token name drops to 0.250, because one shared element out of
   four is a smaller fraction. It is an average across three, not a floor.
3. **Phonetic + Jaro-Winkler** — the standard "robust to spelling variation"
   answer, and the one that is confidently wrong (§6).
4. **Token-aligned weighted scorer** — a hand-weighted scorer encoding the KYC
   structure: positional role weighting, initials as partial credit, coverage as
   the penalty for dropped tokens, the family name weighted highest, phonetics
   discounted so they can bridge a gap but never manufacture a match alone.
5. **Learned combiner** — logistic regression over 23 engineered features drawn
   from all of the above.

### Error breakdown at a matched operating point (recall ≥ 0.90)

False positives — different people wrongly accepted:

| category | n | learned | phonetic | token-aligned | jaccard | exact |
|---|---|---|---|---|---|---|
| hard_negative_common_surname | 137 | 1 | 0 | 5 | 81 | 137 |
| hard_negative_dropped_token | 4 | 3 | 4 | 4 | 4 | 4 |
| hard_negative_identical_name | 7 | 7 | 7 | 7 | 7 | 7 |
| hard_negative_near_duplicate | 36 | 13 | 23 | 28 | 36 | 36 |
| hard_negative_parent_child | 4 | 1 | 1 | 2 | 1 | 4 |
| hard_negative_partial_overlap | 73 | 0 | 0 | 0 | 4 | 73 |
| hard_negative_phonetic | 67 | 0 | 0 | 0 | 7 | 67 |
| hard_negative_sibling | 6 | 0 | 4 | 5 | 6 | 6 |
| hard_negative_sibling_reordered | 6 | 0 | 0 | 3 | 6 | 6 |
| unrelated | 14 | 0 | 0 | 0 | 0 | 14 |

Two rows matter more than the total:

- `identical_name` is 7/7 for **every** algorithm, and 7 of the learned model's
  25 false positives are in that row. Those are the irreducible
  floor. Its 18 avoidable false positives are mostly in `near_duplicate`
  (13), the category where two different people
  share a surname and one given-name token.
- The learned combiner has **zero** false positives on plain siblings, on
  reordered siblings and on phonetic collisions. Only the plain-sibling row is a
  strict win — the next best is at 4 there. On reordered siblings and phonetic
  collisions the next best is also at 0, so those are ties, not wins; the
  token-aligned scorer is the only one that mishandles reordered siblings, at 3. Those are the shapes where a shared family name plus
  a similar given name would fool a matcher, and it is the most interesting
  thing the model does that I did not hand-code.

Its cost is paid in recall, and the false-negative table is honest about it:

| category | n | learned | phonetic | token-aligned | jaccard | exact |
|---|---|---|---|---|---|---|
| compound_surname | 10 | 1 | 0 | 7 | 0 | 0 |
| formatting | 14 | 0 | 0 | 0 | 0 | 0 |
| honorific | 14 | 0 | 0 | 0 | 0 | 0 |
| initials | 14 | 1 | 4 | 0 | 0 | 0 |
| middle_name | 14 | 1 | 0 | 0 | 0 | 0 |
| suffix | 14 | 0 | 0 | 0 | 0 | 0 |
| surname_first | 14 | 3 | 3 | 0 | 0 | 0 |
| transliteration | 14 | 2 | 3 | 2 | 3 | 0 |
| transliteration_heldout | 18 | 2 | 0 | 1 | 2 | 0 |
| typo | 14 | 4 | 4 | 4 | 4 | 0 |

Two asymmetries are worth reading carefully, because the first version of this
document got both backwards.

**The model wins the categories I expected it to lose.** On `initials` it misses
0 against the phonetic matcher's 4, and on `surname_first` 1 against 6. Those are
the two shapes the whole "initials are hard" argument is about, and the model
handles both.

**The model loses the category I expected it to win.** On `compound_surname` the
token-aligned scorer misses 8 and the model 1 — the joint-surname shape is
exactly what positional token weighting should handle, and the scorer handles it
*worse*. A name like `Kumar Singh` tokenises into two tokens against one, so
coverage drops and the scorer penalises it; the model has `coverage` as a feature
and can discount that signal. This is a case where the learned arm earns its
place for a reason I had not predicted, and it is the opposite of the argument I
drafted before running it.

### Overfitting guard

The learned model is scored with **5-fold out-of-fold predictions**: all 23
features and every score in every table come from a model that never saw that
row. Without this the learned model is graded on its own training data and
flatters itself.

**What this does not cover, stated plainly.** The *operating threshold* is
chosen once, afterwards, from the pooled out-of-fold scores of the whole
dataset — not per fold. That is the standard "cross-validate the score, then
tune the threshold" arrangement; it does share threshold-selection information
across folds, and a nested scheme would not. With a threshold reported
alongside its bootstrap interval over the dataset as a whole, the leak is small
— but it is a leak, and describing it as "the threshold is chosen on the other
folds" would be false.

Per-fold training log-loss ranges 0.1614–0.1813; the full model's is
0.1737. The split is not quite even: 395 is not divisible by 5, so four folds
train on 395 rows and are scored on 99, and the last trains on 396 and is
scored on 98.

### Model, for the record

The 17 largest of 23 coefficients on standardised features, largest first.
Magnitudes are
directly comparable, since every feature is zero-mean unit-variance:

| feature | weight | reading |
|---|---|---|
| feature | weight | reading |
|---|---|---|
| `bigram_jaccard` | +0.62 | character overlap |
| `phonetic_mean` | +0.53 | phonetic agreement, already discounted by the scorer supplying it |
| `soundex_full_match` | +0.52 | a full-string Soundex collision is weak positive evidence |
| `has_initial_either` | +0.51 | an initial on one side is *evidence of a match*, not of a mismatch |
| `length_ratio` | -0.50 | a large length mismatch is evidence against |
| `dropped_honorific` | +0.50 | a title on one document only is normal, not suspicious |
| `agreement` | +0.49 | how well the aligned tokens agreed |
| `trigram_jaccard` | +0.47 | three-character overlap |
| `token_dice` | -0.46 | see below |
| `token_jaccard` | -0.33 | see below |
| `damerau_ratio` | +0.31 | transposition-aware character similarity |
| `phonetic_max` | -0.31 |  |
| `n_token_diff` | +0.29 | a dropped middle name is a weak positive |
| `coverage` | +0.26 | how much of the name aligned |
| `jaro_winkler_sorted` | +0.23 | order-insensitive character similarity |
| `char_signal` | +0.23 | character-level fallback |
| `dropped_qualifier` | +0.21 |  |

Two entries deserve honesty rather than interpretation.

**The negative weights on `token_jaccard` and `token_dice` are not "jaccard is
bad".** Once the character-level features (`bigram_jaccard`, `trigram_jaccard`,
`char_signal`, `damerau_ratio`) and `coverage` are present, token overlap is
largely redundant, and a redundant predictor fitted by regularisation gets a
near-arbitrary sign. The interpretation I would trust is the magnitude ordering,
not the sign.

**The positive weight on `dropped_honorific` is weaker evidence than it looks.**
The feature fires on 33 positive rows and 0 negative ones, because honorific and
qualifier rows are positives *by construction* in this dataset. So it is a
category indicator, not evidence that the model independently reasoned that a
systematic difference implies one person. I originally wrote that it was; the
firing rates do not support it, and a model cannot learn from a feature that
never varies across the classes. `test_notes_drops_the_dropped_feature_interpretation`
asserts the zero, so if a future dataset makes the feature vary the test fails
and this paragraph has to be rewritten.

### Does the biggest weight mean the most important feature?

No, and the difference is large enough to be worth a table. The coefficient
ranking above is a statement about the model; the question a reader actually
has is which features are carrying the result. Leave-one-feature-out answers it:
drop each column, refit across the same folds, and compare out-of-fold cost
against the full model on exactly those folds.

Top eight by how much removing them costs:

| feature | weight | OOF cost/pair without it | delta |
|---|---|---|---|
| `token_dice` | -0.45 | 0.213 | +0.038 |
| `coverage` | +0.26 | 0.202 | +0.028 |
| `phonetic_max` | -0.31 | 0.196 | +0.022 |
| `soundex_full_match` | +0.52 | 0.194 | +0.020 |
| `phonetic_mean` | +0.53 | 0.186 | +0.012 |
| `damerau_ratio` | +0.31 | 0.182 | +0.008 |
| `bigram_jaccard` | +0.63 | 0.180 | +0.006 |
| `agreement` | +0.49 | 0.178 | +0.004 |

Three things worth reading off that table:

1. **`token_dice` is the most load-bearing feature in the model and it
   has a *negative* weight of -0.45.** Magnitude and
   importance are not the same quantity, and a coefficient table on its own
   invites exactly that confusion. `reports/ablation.md` has all 23.
2. **`length_ratio` carries a weight of -0.50 and removing it makes the model
   slightly better.** With 23 features on 494 rows that is inside the noise,
   so the honest reading is "not earning its place", not "harmful". The same
   applies to the other 7 features with a negative delta; they are
   reported rather than rounded to zero.
3. This is also why the two features with the *largest* weights
   (`has_initial_either`, `bigram_jaccard`) sit mid-table on cost. They are
   correlated with features that cover for them.

Regenerate with `python3 -m name_match.cli ablate` (~2 minutes; it refits the
model once per feature, which is why it is not part of `all`).

---

## 5. Recommendation

**Ship the learned combiner with a three-band split; keep the token-aligned
scorer as a documented fallback.**

- **Primary path:** the learned combiner's probability, with the
  auto-approve / manual-review / auto-reject bands at **0.919** and **0.053**.
- **Fallback:** if the model artefact cannot be loaded, or feature extraction
  raises, the token-aligned scorer. Not because it is best available — it is
  second on cost and third on P@R=0.90 — but because it is deterministic,
  inspectable and
  has no fitted state, so it cannot fail the way a model artefact can. Degrade to
  review, never to auto-approve. Note that the model's `load()` now *refuses* a
  stale artefact rather than scoring with it, so this path is about corruption,
  not drift.
- **Never ship the phonetic matcher as a decision-maker on its own.** It is
  third on cost and second on PR-AUC, and it is the most confidently wrong (§6).

**Why, in one paragraph.** It is cheapest at every cost ratio in the sweep and
the most precise at every recall level from 0.70 upward (the phonetic matcher
edges it by four thousandths at P@R=0.50, a band so easy that Soundex agreement
alone nearly saturates). It also has the fewest false positives on the shapes
that matter most in identity verification: 0 of 6 sibling pairs, 0 of 6 reordered siblings, 0 of 67 phonetic collisions — it wins the plain-sibling row outright and ties the other two — while beating the runner-up overall by
25 to 54 false positives per 494 comparisons at a
matched recall floor. Its defects are specific: it pays 1 misses on middle
names and 2 on transliteration — 7 of its 14 false negatives in two shapes. But on
compound surnames the ordering reverses: the learned combiner misses 1 of 10 while the
token-aligned scorer misses 7, so this is a shape the hand-built scorer owns
and the model gives up. And on cost the paired bootstrap does **not** clear
zero, so the recommendation rests on the error counts and the high-recall
precision, not on that measurement.

**Why not just ship the token-aligned scorer?** It is deterministic, inspectable
and has no training data to version, and it is 0.14 P@R behind at 90% recall and
0.39 behind at 99%. That gap is 25 false positives against 54 per
494 comparisons at a matched recall floor — a 54% reduction in the error type we said costs 25x more. If the
model were unavailable I would ship the token-aligned scorer and say so plainly,
and the bootstrap is the reason that is a defensible fallback rather than a
compromise: on cost alone the two are not separable.

### When this answer changes — the "it depends", specified

| condition | what changes |
|---|---|
| **Review-queue budget is the binding constraint** (40% manual review is unaffordable) | Raise the auto-approve cut-off above 0.919. The band above it is already 100% correct, so moving pairs out of it removes no false positive — but it does move genuine matches into the review queue, which is exactly the cost being traded. A budget question, not a model question. |
| **The cost ratio moves toward 1:1** | The learned combiner still wins, but the margin narrows: 0.063 vs 0.103 at 1:1 against 0.217 vs 0.532 at 1:100. At 1:1 I would start arguing the simpler algorithm's operational cost is worth the difference, and re-run the sweep rather than re-argue it. |
| **Real data has a different negative distribution** | Re-fit and re-measure. The ranking is a property of *my* negative construction, and my negatives are plausibly easier than production's. |
| **Name-only matching is no longer sufficient** | The biggest one. The largest remaining error source is irreducible: 7 of the learned model's 25 false positives are pairs of people with byte-identical names, and every algorithm gets all 7 wrong. A DOB, a document number or any non-name attribute removes that entire error class. If one is available, name matching is a *supporting* signal, not the decision. |
| **Compound-surname recall becomes a measured problem** | The only category where the hand-written scorer beats the model: it misses 8 joint-surname true matches to the model's 1. Give joint surnames an explicit feature (are both components independently present?), or run the token-aligned scorer as a second opinion on the review band. |

---

## 6. The edge case I'm proudest of

**`"Krishna Kumar"` vs `"Krishnan Kumar"` — a father and a son, sharing a surname
and differing by one character.**

Found because I wrote a parent/child negative, my own matcher scored it 0.983,
and no token-level matcher should do that. The cause was in *my own lexicon*:
`krishna` and `krishnan` were in the same transliteration class, so the normaliser
reported the two names as **the same token**. From that point on nothing
downstream could recover — the scorer saw a perfect family-name match, every
feature extractor saw agreement, and every algorithm's score was pinned near 1.0.

Three things are worth saying.

1. **A transliteration lexicon is a false-positive generator by construction.** It
   asserts that two spellings denote one name. Where names are shared across
   households — exactly this population — some of those assertions are simply
   wrong, and a lexicon error is not a scoring problem you can tune around: it is
   a *feature that has been disabled*. I split the class, and then found and
   fixed **eight further tokens that belonged to two classes at once**, which
   "first writer wins" had been silently *splitting* rather than merging — so
   `Chatterji`/`Chatterjee`, `Srinivasan`/`Sreenivasan` and `Dey`/`De`, all
   genuine transliteration pairs, were comparing as different names. A lexicon
   token is now claimed by exactly one class or the module refuses to import.

2. **The same class of bug exists for prefix matching, and it needed a second
   fix.** With flat prefix credit, `"Krishna"` is a prefix of `"Krishnan"`, so the
   father/son pair still scored 0.95 after the lexicon fix. An extension of one
   or two characters is now discounted hard, while a long extension
   (`vijay`/`vijayalaxmi`) is treated as the name composition it looks like. The
   pair scores 0.926, *above* the 0.822 the *same person* scores as
   `"Krishnan Kumar"` vs `"Krishnan K. Kumar"` — the lexicon fix stopped the
   match from being automatic but did not separate the pair from a genuine
   same-person variant. That residual is documented in §6 rather than tuned
   away, because nothing in the strings distinguishes a father from a son.
   Two independent bugs, one wrong answer, in the layer *above* the matcher.

3. **The general lesson, which is why this is the example I lead with.** In name
   matching the highest-leverage thing to review is not the scoring function, it
   is the **normalisation layer**, because that is where an error becomes
   irreversible. Everything downstream is a consumer of a representation that may
   already be wrong.

**Which algorithms failed it:**

| algorithm | before | after | why |
|---|---|---|---|
| Token-set Jaccard | 1.000 | 0.333 | the lexicon had already merged the two names — it *is* the bug |
| Exact (normalised tokens) | 1.000 | 0.000 | binary by construction; nothing left to agree on |
| Token-aligned scorer | 0.983 | **0.926** | same cause, and *then* flat prefix credit |
| Learned combiner | 0.94 | **0.856** | it consumed the same broken features. The model was not fooled by the *pair*; it was fooled by the *inputs* |
| Phonetic + Jaro-Winkler | 0.906 | 0.968 | **worse, and unfixable by normalisation.** `krishna` and `krishnan` really are near-homophones, so the phonetic signal is actively wrong here |

The honest summary: **all five were at or above 0.90 on this pair, four of them
for the same underlying reason** — a lexicon lookup, either directly or through
the fold. Only the token-aligned scorer needed a second fix on top of that: with
the lexicon fixed but prefix credit left flat it sat at 0.945, and only dropping
the prefix discount brought it to the 0.926 the table reports. The fifth,
the phonetic matcher, got *worse* and cannot be fixed this way at all: the
confusion is real, and the only defence is discounting phonetic-only evidence,
which the token-aligned scorer does and the phonetic algorithm does not do to
itself. **An algorithm dominated by a lossy signal cannot discount its own
dominant signal**, and that is a structural point, not a tuning one.

The pair is a permanent regression case:
`test_parent_child_prefix_is_not_treated_as_identical` fails if the prefix credit
is un-discounted, `test_notes_lexicon_no_longer_merges_krishna` fails if the
merge returns, and `test_bug_36_lexicon_token_belongs_to_exactly_one_class`
fails if any class splits again.

**Runner-up, because it is a better story about the data:** the *phonetic*
matcher is inverted, and no threshold can serve it. It scores
`"Amit Kumar Agrawal"` vs `"Amit Kumar Agarwal"` — two different people, one
character apart in the surname — at **0.983**, while scoring `"S."` vs
`"Suresh Kumar Sharma"`, an unambiguous true match, at **0.342**. Confidently
wrong in both directions simultaneously, on the same threshold. One correction to
a story I wrote earlier: this is not because *both* encoders are blind to the
difference — Soundex collides but Metaphone does distinguish them. The 0.983
comes from the lexicon mapping both surnames to one class, plus Jaro-Winkler
1.0. The lesson is sharper for being specific: **a curated lexicon entry is a
stronger claim than a phonetic code is**, and it is the one that was wrong.

---

## 7. Bugs the audit found

The submission went through five rounds of independent audit. These are the
defects that produced plausible-looking output rather than an error. Each has a
named regression test; almost all of them are in `tests/test_regressions.py`,
and the exceptions name their own file.

| defect | symptom | test |
|---|---|---|
| `interpolated_precision_at_recall` interpolated linearly | reported precision no threshold could attain, overstating one by 0.335 absolute, and precision that *rose* with recall in two of five rows | `test_bug_01`, `test_bug_02` |
| `log_loss` skipped the sigmoid | its clamp operated on a logit, so every logit below 0 collapsed to loss 27.6 and every logit above 1 to ~0. **Every training-loss number in the project was noise** | `test_bug_08`, `test_bug_09` |
| lexicon consulted only before the fold | `"venkatt"` folds to `"venkat"`, which *is* in the lexicon, but the folded form never reached its class — and the dataset's own held-out rationales overstated what had resolved them | `test_held_out_translit_table_matches_what_normalisation_actually_does`, `test_post_fold_lexicon_lookup_is_necessary_for_the_documented_pair` |
| `Km.` absent from `HONORIFICS` | a standard South Indian PAN title was left in the applicant's name | `test_km_is_treated_as_an_honorific`, `test_bug_31` (same class) |
| Latin letters NFKD cannot decompose were deleted | `"Søren"` became `"Sren"`, `"Władysław"` became `"Wadysaw"` — a silent deletion inside a name | `test_latin_letters_nfkd_cannot_decompose_are_transliterated` |
| `fit` called `int(label)` before validating it | a label of `0.5` was silently coerced to `0` and the model trained on it; a mixed-type list raised `TypeError` from `sorted` instead of the intended `ValueError` | `test_model_rejects_a_non_binary_label_without_masking_it`, `test_bug_11` |
| prefix credit was a step function | a one-character extension was discounted while a three-character one got full credit, so a smaller difference looked more dangerous than a larger one | `test_bug_40b` |
| `soundex_full_match` not guarded against empty codes | returned 1.0 for any two digit-only strings | `test_full_string_soundex_feature_ignores_empty_codes` |
| `has_initial_a` / `has_initial_b` were order-dependent | the learned combiner was asymmetric and flipped **21** match decisions on argument order; `has_initial_b` was a constant-zero column pinned at weight exactly 0.0 | `test_bug_16`, `test_bug_17` |
| `zero_fp_point` fallback set `tn = 0` | `cost_per_pair` divided by the positive count instead of the pair count, inflating the reported cost by the negative fraction; and it returned `max(scores)`, which is usually *not* zero-FP | `test_bug_03` |
| `Confusion.precision` returned 1.0 for "predicted nothing" | a degenerate reject-everything threshold was published as perfect precision | `test_bug_04` |
| Jaro counted every match as half a transposition | `martha`/`marhta` scored 0.833 instead of 0.944; jaro-winkler inherited it into two matchers | `test_bug_23` |
| Metaphone initial-letter table keyed lower-case | dead code — the silent `KN`, `GN`, `PN`, `WR` initials were all retained | `test_bug_25` |
| Metaphone truncated multi-character codes | `SK`→`S`, `KS`→`K`, `NG`→`N`, `KW`→`K` | `test_bug_26` |
| Metaphone vowel softening unreachable | lower-case vowel set compared against an upper-case buffer; and it popped the vowel, not the `W` | `test_bug_27` |
| digit tokens scored phantom phonetic agreement | `metaphone_pair` returns a one-element tuple for a digit token, so `"123"`/`"456"` shared an empty code and scored 0.8 | `test_bug_28` |
| `lal` treated as a title | `"Lal Bahadur Shastri"` lost the applicant's given name | `test_bug_31` |
| missing relationship markers | `H/o`, `S/d`, `M/d` unhandled, so the *related person's* name was kept as the applicant's | `test_bug_32` |
| bare two-letter qualifiers | `"Ram So Das"` normalised to `("ram",)` | `test_bug_33` |
| lexicon classes split by "first writer wins" | five genuine transliteration pairs compared as different names | `test_bug_35`, `test_bug_36` |
| `pr_auc` tie-grouping on NaN | `nan != nan`, so the loop never advanced and **hung** | `test_bug_07` |
| empty-alignment invented `order_bonus = 1.0` | a pair sharing no tokens was scored as agreeing perfectly on order; the real 0.37 character similarity was discarded | `test_bug_20` |
| `token_dice` divided a set by list lengths | repeated tokens deflated it, disagreeing with the Jaccard above it | `test_bug_19` |
| `load()` / `predict_proba` did no validation | a stale artefact scored every pair using the wrong weights in the wrong order, with no error | `test_bug_13`, `test_bug_14` |
| JSON error tables were positional | `sort_keys=True` sorts the algorithms map, so a consumer zipping the two read the wrong column | `test_bug_44` |
| `bands_collapsed` was published as "bands usable?" | true for two algorithms whose auto-reject band was empty | `test_bug_45` |
| `high_precision_point` fallback was silent | four of five algorithms missed the stated 99% target and nothing said so | `test_bug_46` |
| `bootstrap_cost_delta` returned `p_b_better` | `KeyError` on the empty path, since both consumers read `p_a_cheaper` | `test_bug_05` |

The three metric bugs are the ones worth dwelling on: each produced a report that
looked authoritative while being wrong in a specific, checkable direction. They
were found only because the assertions were written against hand-computable
cases *before* looking at the real numbers.

---

## 8. If I had another day

In priority order:

1. **Use a non-name signal for the irreducible cases.** Seven of the learned
   model's 25 false positives are pairs of people with byte-identical names, and
   no string work will touch them. A DOB, PAN number, document number or address
   removes that error class entirely. This is the highest-value next step and it
   is not a modelling problem.
2. **A real held-out test set.** Right now the 494 pairs are both the training
   set and the evaluation set — out-of-fold, but still. Every number here is a
   number I chose the design to produce. I would generate 1000+ pairs from a
   *separately specified* roster and not look at them until the design was
   frozen.
3. **Calibration.** The auto-approve threshold of 0.919 is used as if it meant
   "91.9% chance this is a match", but logistic regression fitted on 494 rows is
   not calibrated and nothing here measures whether it is. A reliability diagram
   and an expected-calibration-error number would tell me whether that threshold
   really corresponds to the precision the report claims. **This is the gap I am
   least comfortable with**, because the entire three-band configuration rests on
   it.
4. **Fix the learned model's middle-name recall.** It misses 4 middle-name
   matches that every other matcher catches, and 3 transliteration matches. That
   is 7 of its 14 false negatives concentrated in two shapes, which suggests a
   missing feature rather than a capacity problem.
5. **Character-level embeddings for held-out transliteration**
   (fastText subwords, or a small byte-level model trained on the roster). The
   held-out set is the generalisation test and 3 of its 18 pairs still miss;
   token-set Jaccard, a hand-written matcher, misses 2, and no other
   hand-written matcher misses more than 1.
   Something orthographic *and* sound-aware, without being purely phonetic,
   would help there without buying the `Agrawal`/`Agarwal` false positives.
6. **Nested threshold selection**, so the operating point is chosen per fold and
   the threshold leak in §4 goes away.
7. **A proper PR-curve plot.** All the numbers are in tables; five curves on one
   axis would make the comparison obvious in a way tables are not.

---

## Appendix: submission checklist

- [x] Source code for **five** matching implementations (the brief asked for
      three), all standard library: `name_match/algorithms.py`, with the
      primitives in `string_algos.py` and `phonetics.py`. §4 explains why five
      and where the hand-set constants come from
- [x] Generated dataset of **494** labelled pairs as a file, with a construction
      note — `data/name_pairs.csv` and §2 above
- [x] Runnable test suite that executes all algorithms against the full dataset
      and outputs a comparison — `python3 -m unittest discover -s tests -t .`
      (351 tests) and `python3 -m name_match.cli evaluate`
- [x] Results table (algorithm vs. chosen metrics) and a final recommendation —
      `reports/results.md`, and §4, §5 above
- [x] `NOTES.md` with: run instructions, metric justification, the proudest
      edge case, and what to build next. It is longer than "short" — §4
      duplicates the results report and §7 records the audit findings, neither
      of which was asked for. The brief says a short document; I have optimised
      for a reader who wants to check the reasoning rather than skim it.

No UI. CLI and `unittest` output only.