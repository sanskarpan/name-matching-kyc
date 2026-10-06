"""Evaluation: metrics, operating points, cost analysis and uncertainty.

Metric design is the substance of this exercise, so it is worth being explicit
about what is computed and why.

Primary metric: **asymmetric triage cost at a chosen operating point**.

A false positive is a different person's name flagged as a possible match; a
false negative is a genuine match missed. If these scores directly approved
identity, false positives would be the greater risk. The baseline here instead
models a review queue: a false positive costs extra review, while a false
negative risks excluding a genuine customer from the queue.

    C = c_FP * FP + c_FN * FN

The illustrative triage assumption is ``c_FP = 1, c_FN = 25``. It is not a
fraud-acceptance cost model. Automatic decisions use separate precision and
recall constraints in the three-band analysis. The sensitivity sweep varies
``c_FN/c_FP`` from 1 to 100; it does not test a false-positive-heavy policy.
Accuracy and F1 are reported only as supporting metrics: neither expresses
these unequal business costs. Precision at a recall floor measures the review
burden while requiring that genuine matches remain discoverable.

Secondary metrics: PR-AUC (preferred over ROC-AUC because the evaluation set is
28% positive and 72% negative, so ROC's false-positive axis is dominated by the
large negative population and flatters a model that is only good on the easy
tail), per-category error rates, a confidence interval on the cost difference,
and a three-band operating point (auto-approve / manual review / auto-reject)
which is what would actually ship.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Sequence

__all__ = [
    "ScoreSet",
    "Confusion",
    "cost_at",
    "find_operating_point",
    "pr_auc",
    "roc_auc",
    "precision_at_recall",
    "interpolated_precision_at_recall",
    "find_operating_point_at_recall",
    "high_precision_point",
    "zero_fn_point",
    "zero_fp_point",
    "band_thresholds",
    "band_report",
    "per_category_error",
    "score_all",
    "bootstrap_cost_delta",
    "three_band_split",
    "DEFAULT_COST_FN",
    "DEFAULT_COST_FP",
]

#: Illustrative triage costs: missing a genuine match costs 25 times an
#: unnecessary review. Automatic approval has a separate precision constraint.
DEFAULT_COST_FN = 25.0
DEFAULT_COST_FP = 1.0


@dataclass
class ScoreSet:
    """Scores and labels for one algorithm over the whole dataset."""

    key: str
    label: str
    scores: list[float]
    labels: list[int]
    #: Parallel to ``scores``: the dataset rows, for per-category breakdowns.
    rows: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.scores)


@dataclass
class Confusion:
    """Confusion counts at a specific threshold."""

    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int

    #: False unless the threshold met the precision target it was searched
    #: against. Set by :func:`high_precision_point` on its fallback path, so a
    #: report can distinguish "the rule was satisfied" from "the rule could not
    #: be satisfied and this is the best available".
    target_met: bool = True

    @property
    def precision(self) -> float:
        """Precision, or NaN when nothing was predicted.

        Reporting 1.0 here would be a lie with teeth: a threshold that predicts
        nothing has *undefined* precision, and reporting it as perfect is how a
        degenerate threshold comes to look like the safest one in a report.
        Callers wanting a number should check `predicted` first.
        """
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else float("nan")

    @property
    def predicted(self) -> int:
        """How many pairs this threshold actually predicted as a match."""
        return self.tp + self.fp

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        if precision != precision or recall != recall:
            return float("nan")
        return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total else 0.0

    def cost(self, cost_fp: float = DEFAULT_COST_FP,
             cost_fn: float = DEFAULT_COST_FN) -> float:
        return cost_fp * self.fp + cost_fn * self.fn

    def cost_per_pair(self, cost_fp: float = DEFAULT_COST_FP,
                      cost_fn: float = DEFAULT_COST_FN) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return self.cost(cost_fp, cost_fn) / total if total else float("inf")

    def as_row(self, cost_fp: float = DEFAULT_COST_FP,
               cost_fn: float = DEFAULT_COST_FN) -> dict[str, float]:
        return {
            "threshold": self.threshold,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "cost": self.cost(cost_fp, cost_fn),
            "cost_per_pair": self.cost_per_pair(cost_fp, cost_fn),
        }


#: Added to the maximum observed score to obtain a genuine "predict nothing"
#: threshold. Large enough to clear any finite score, small enough not to
#: collide with a legitimate neighbouring threshold.
_ABOVE_ALL_SCORES = 1e-9


def _validate_inputs(scores: Sequence[float], labels: Sequence[int]) -> None:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("scores must be finite (no NaN or infinity)")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("labels must be 0 or 1")


def _threshold_grid(scores: Sequence[float]) -> list[float]:
    """Candidate decision thresholds for a score vector.

    The smallest and largest observed scores are usable as-is: the decision rule
    is ``score >= threshold``, so the smallest observed score already means
    "accept everything" and the largest already means "accept only the top
    score". Everything between is covered by midpoints, which makes the search
    exact rather than grid-resolution dependent -- important, because the
    recommended threshold is quoted verbatim in the report.
    """
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("scores must be finite (no NaN or infinity)")
    candidates = sorted(set(scores))
    if not candidates:
        return [0.5]
    thresholds = [candidates[0]]
    for low, high in zip(candidates, candidates[1:]):
        thresholds.append((low + high) / 2.0)
    thresholds.append(candidates[-1])
    return thresholds


def cost_at(scores: Sequence[float], labels: Sequence[int], threshold: float,
            cost_fp: float = DEFAULT_COST_FP,
            cost_fn: float = DEFAULT_COST_FN) -> Confusion:
    """Confusion counts for a ``score >= threshold`` match decision.

    A NaN score compares false against every threshold and so would be silently
    classified as a non-match. That is a plausible-looking result produced by an
    upstream arithmetic fault, so NaN is rejected rather than absorbed.
    """
    _validate_inputs(scores, labels)
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels):
        predicted = 1 if score >= threshold else 0
        if predicted == 1 and label == 1:
            tp += 1
        elif predicted == 1 and label == 0:
            fp += 1
        elif predicted == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    return Confusion(threshold, tp, fp, tn, fn)


def find_operating_point(scores: Sequence[float], labels: Sequence[int],
                         cost_fp: float = DEFAULT_COST_FP,
                         cost_fn: float = DEFAULT_COST_FN) -> Confusion:
    """Threshold minimising total cost, using only midpoints of observed scores.

    Restricting candidate thresholds to score midpoints (rather than a fixed
    grid) is both exact and free of grid-resolution artefacts, which matters
    when the recommended threshold is quoted in a report.
    """
    _validate_inputs(scores, labels)
    if not scores:
        return Confusion(0.5, 0, 0, 0, 0)

    thresholds = _threshold_grid(scores)
    # Rejecting every pair can be optimal, especially when false positives
    # are costly or the scores contain no useful separation.
    thresholds.append(math.nextafter(max(scores), math.inf))

    best: Confusion | None = None
    best_key: tuple[float, int, float] | None = None

    for threshold in thresholds:
        confusion = cost_at(scores, labels, threshold, cost_fp, cost_fn)
        # Rank by total cost first. Ties break toward fewer false positives,
        # then toward the higher threshold (the more conservative decision).
        #
        # Comparing (fp, fn) lexicographically instead of comparing cost first
        # is a subtle and total failure: it ranks the "predict nothing"
        # threshold as optimal for every algorithm, because that threshold has
        # zero false positives by construction.
        key = (confusion.cost(cost_fp, cost_fn), confusion.fp, -threshold)
        if best_key is None or key < best_key:
            best_key = key
            best = confusion

    assert best is not None
    return best


def zero_fp_point(scores: Sequence[float], labels: Sequence[int]) -> Confusion:
    """Highest-recall threshold that produces no false positive at all.

    This is the threshold a bank would pick if a false positive were literally
    unacceptable (for example, auto-approving an account opening). Reported
    alongside the cost-optimal point to show the price of that constraint.

    The sweep starts above the maximum score, where the prediction is "no
    matches" and ``fp`` is necessarily zero, so a zero-FP threshold always
    exists. If the only such threshold is the degenerate one above every score,
    the returned confusion has ``fp == 0`` and ``fn == n_positive``: an
    algorithm that cannot separate a single positive is reported that way
    rather than being handed a flattering threshold it never earned.
    """
    _validate_inputs(scores, labels)
    candidates = sorted(set(scores))
    if not candidates:
        return Confusion(1.0, 0, 0, 0, 0)

    best: Confusion | None = None
    for threshold in candidates:
        confusion = cost_at(scores, labels, threshold)
        if confusion.fp == 0 and (best is None or confusion.fn < best.fn):
            best = confusion
    if best is not None:
        return best

    # No threshold in the observed score range is free of false positives. The
    # honest answer is the degenerate "reject everything" point, which does
    # achieve FP == 0 -- and it has to be built explicitly, ABOVE the maximum
    # score, because `max(scores)` itself is very often NOT zero-FP: any
    # negative whose score ties the highest-scoring positive breaks it.
    #
    # The confusion matrix must be filled in fully. Leaving `tn` at zero makes
    # `cost_per_pair` divide by the positive count instead of the pair count and
    # inflates the reported cost by the negative fraction; and `precision` would
    # be undefined, which `Confusion` now reports as NaN rather than 1.0.
    return Confusion(
        threshold=max(candidates) + _ABOVE_ALL_SCORES,
        tp=0, fp=0,
        tn=sum(1 for label in labels if label == 0),
        fn=sum(labels),
    )


# --------------------------------------------------------------------------
# Ranking metrics (hand-rolled: no numpy)
# --------------------------------------------------------------------------


def pr_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the precision-recall curve, via average precision.

    Average precision is the step-wise integral of precision over recall. It
    is the right ranking metric here because ROC-AUC's false-positive axis is
    dominated by the (large, easy) negative population, which flatters a
    matcher that is only good on the easy tail.
    """
    _validate_inputs(scores, labels)
    if not scores:
        return 0.0
    n_positive = sum(labels)
    if n_positive == 0:
        return 0.0

    # Average precision is computed one *distinct score value* at a time.
    #
    # Grouping by score is not optional here. Two of the five algorithms emit
    # large numbers of exactly-tied scores (the control emits only 0.0 and 1.0),
    # and if ties are broken by row order the metric silently becomes a
    # function of CSV ordering: an all-tied ranking scores anywhere from 0.0 to
    # 1.0 depending on which label happened to come first. With tie grouping, a
    # ranking that carries no information scores exactly the base rate, which is
    # the honest answer.
    if any(score != score for score in scores):
        raise ValueError("pr_auc received a NaN score: NaN never equals itself, so "
                         "the tie-grouping loop would not terminate")
    order = sorted(range(len(scores)), key=lambda i: -scores[i])

    true_positives = 0
    false_positives = 0
    previous_true_positives = 0
    precision_sum = 0.0
    index = 0
    while index < len(order):
        threshold = scores[order[index]]
        end = index + 1
        while end < len(order) and scores[order[end]] == threshold:
            end += 1
        for position in range(index, end):
            if labels[order[position]] == 1:
                true_positives += 1
            else:
                false_positives += 1
        retrieved = true_positives + false_positives
        # The recall gained by admitting this whole block, valued at the
        # precision achieved once the block is in.
        recall_gain = (true_positives - previous_true_positives) / n_positive
        precision_sum += recall_gain * (true_positives / retrieved if retrieved else 0.0)
        previous_true_positives = true_positives
        index = end

    return precision_sum


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC curve, computed by rank (ties contribute 0.5).

    Ranks are assigned in **ascending** score order, which is the orientation
    the Mann-Whitney formulation expects: a positive ranked above a negative
    contributes +1, below it contributes -1. Sorting descending here (the
    intuitive-looking choice) silently returns ``1 - AUC``, which is the kind of
    bug that survives review because the number still "looks like" a metric.
    """
    _validate_inputs(scores, labels)

    n_positive = sum(labels)
    n_negative = len(labels) - n_positive
    if n_positive == 0 or n_negative == 0:
        return 0.5

    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    rank_sum = 0.0
    index = 0
    while index < len(order):
        # Group equal scores so ties contribute an average rank.
        end = index
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[index]]:
            end += 1
        average_rank = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            if labels[order[position]] == 1:
                rank_sum += average_rank
        index = end + 1

    return (rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------


def bootstrap_cost_delta(scores_a: Sequence[float], labels: Sequence[int],
                         scores_b: Sequence[float], threshold_a: float,
                         threshold_b: float, iterations: int = 2000,
                         seed: int = 20260101,
                         cost_fp: float = DEFAULT_COST_FP,
                         cost_fn: float = DEFAULT_COST_FN) -> dict[str, float]:
    """Paired bootstrap on the per-pair cost difference between two algorithms.

    The comparison is *paired*: the same resampled rows are scored by both
    algorithms, which removes dataset-difficulty variance from the difference
    and makes the interval far tighter than comparing two independent
    bootstraps would.

    The intervals are reported precisely so that small differences can be
    dismissed. On a 494-row dataset, a cost difference of a couple of points is
    inside the noise, and saying so is the honest result.
    """
    n = len(labels)
    if len(scores_a) != n or len(scores_b) != n:
        raise ValueError(
            "bootstrap_cost_delta requires paired inputs: "
            f"{len(scores_a)} scores_a, {len(scores_b)} scores_b, {n} labels")
    _validate_inputs(scores_a, labels)
    _validate_inputs(scores_b, labels)
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    if n == 0:
        return {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_a_cheaper": 0.5}

    rng = random.Random(seed)
    deltas: list[float] = []
    a_better = 0

    for _ in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        labels_s = [labels[i] for i in indices]
        score_a = [scores_a[i] for i in indices]
        score_b = [scores_b[i] for i in indices]

        confusion_a = cost_at(score_a, labels_s, threshold_a, cost_fp, cost_fn)
        confusion_b = cost_at(score_b, labels_s, threshold_b, cost_fp, cost_fn)

        # Normalised by the resample size. Reporting total cost here would make
        # the number scale with `iterations` and read as "24.99 per pair" when
        # it is really a per-pair figure from a different-sized resample.
        #
        # Sign convention: delta = cost(B) - cost(A), so a *positive* delta
        # means A is the cheaper algorithm.
        delta = (
            confusion_b.cost_per_pair(cost_fp, cost_fn)
            - confusion_a.cost_per_pair(cost_fp, cost_fn)
        )
        deltas.append(delta)
        if delta > 0:
            a_better += 1

    deltas.sort()

    def percentile(fraction: float) -> float:
        position = fraction * (len(deltas) - 1)
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return deltas[low]
        weight = position - low
        return deltas[low] * (1 - weight) + deltas[high] * weight

    return {
        "mean_delta": sum(deltas) / len(deltas),
        "ci_low": percentile(0.025),
        "ci_high": percentile(0.975),
        "p_a_cheaper": a_better / len(deltas),
    }


# --------------------------------------------------------------------------
# Three-band operating point
# --------------------------------------------------------------------------


def three_band_split(scores: Sequence[float], labels: Sequence[int],
                     low: float, high: float) -> dict[str, float]:
    """Split predictions into auto-approve / manual-review / auto-reject bands.

    ``low <= score < high`` goes to manual review. This is the shape of a real
    onboarding pipeline: only a high-confidence band is auto-decided, and the
    ambiguous middle is absorbed by a human. Reporting the band sizes and the
    error rates inside each band is more useful to a reviewer than any single
    threshold.
    """
    _validate_inputs(scores, labels)
    if not (math.isfinite(low) and math.isfinite(high)) or low > high:
        raise ValueError("band boundaries must be finite and low <= high")
    bands = {
        "approve": {"n": 0, "correct": 0},
        "review": {"n": 0, "correct": 0},
        "reject": {"n": 0, "correct": 0},
    }

    for score, label in zip(scores, labels):
        if score >= high:
            band = "approve"
        elif score >= low:
            band = "review"
        else:
            band = "reject"
        bands[band]["n"] += 1
        bands[band]["correct"] += int(label == 1) if band == "approve" else int(label == 0)

    result: dict[str, float] = {}
    total = len(scores) or 1
    for name, data in bands.items():
        n = data["n"]
        result[f"{name}_n"] = n
        result[f"{name}_pct"] = 100.0 * n / total
        result[f"{name}_accuracy"] = (data["correct"] / n) if n else float("nan")
    return result


def interpolated_precision_at_recall(scores: Sequence[float], labels: Sequence[int],
                                     target_recall: float = 0.90) -> float:
    """Precision available at a target recall, using the standard PR envelope.

    Defined as

        P(R) = max { precision(r) : r >= R }

    which is the interpolation Davis & Goadrich describe and the one trec_eval
    computes. Two properties make this the right definition rather than a
    stylistic preference, and both were violated by the linear-interpolation
    version this function previously used:

    1. **Every returned value is attained.** A value here is the precision of
       an actual threshold that achieves at least the target recall. Linear
       interpolation between two adjacent achievable points invents values no
       threshold produces. Take the normalisation-only control: it emits a
       single distinct score, so its whole PR curve is the single point
       (recall 1.000, precision equal to the base rate). Under interpolation
       it could be made to claim a precision it never achieves at any recall
       at or above 0.50, because the line between "nothing above the
       threshold" and "everything below it" looks like a slope.

    2. **P(R) is non-increasing in R.** Increasing R shrinks the set of
       feasible thresholds over which the maximum is taken. Raw precision
       may rise or fall as additional positives and negatives are admitted.
       Linear interpolation is free to violate this, and did for two of the five algorithms, producing a
       precision-recall table that was not monotone in the one dimension the
       table is indexed by.

    Returns 0.0 when the target recall is unreachable. Note that "accept
    everything" always reaches recall 1.0, so on a well-formed input this is
    0.0 only when there are no positives at all.
    """
    _validate_inputs(scores, labels)
    n_positive = sum(labels)
    if n_positive == 0 or target_recall <= 0.0:
        return 0.0
    if not set(scores):
        return 0.0

    # Best precision at each achieved recall level. Thresholds that predict
    # nothing have undefined precision and are excluded: including them anchors
    # the envelope at "perfect precision" and inflates every value.
    by_recall: dict[float, float] = {}
    for threshold in _threshold_grid(scores):
        confusion = cost_at(scores, labels, threshold)
        if confusion.tp + confusion.fp == 0:
            continue
        key = round(confusion.recall, 9)
        by_recall[key] = max(by_recall.get(key, 0.0), confusion.precision)

    if not by_recall or max(by_recall) + 1e-9 < target_recall:
        return 0.0

    # Suffix maximum over recall: the best precision achievable at this recall
    # level *or any higher one*. This is the step that makes the result both
    # attainable and monotone.
    ordered = sorted(by_recall.items())
    envelope: list[tuple[float, float]] = []
    best_from_here = 0.0
    for recall, precision in reversed(ordered):
        best_from_here = max(best_from_here, precision)
        envelope.append((recall, best_from_here))
    envelope.reverse()

    # The envelope is non-increasing in recall, so the answer for target R is
    # the value at the lowest recall level that still reaches R.
    for recall, precision in envelope:
        if recall + 1e-9 >= target_recall:
            return precision
    return envelope[-1][1]


def find_operating_point_at_recall(scores: Sequence[float], labels: Sequence[int],
                                   min_recall: float = 0.90,
                                   cost_fp: float = DEFAULT_COST_FP,
                                   cost_fn: float = DEFAULT_COST_FN) -> Confusion:
    """Cheapest threshold subject to a recall floor, with recall *matched*.

    The unconstrained cost-optimal point has a degenerate shape when
    ``c_FN`` is large: the optimiser slides the threshold to the floor of the
    score distribution, achieves recall 1.0, and then the comparison is really
    just "how many false positives can each algorithm push below its worst
    true match". That is a legitimate answer, but it is a weak discriminator.

    Constraining recall fixes the degeneracy -- but only if the comparison is
    genuinely *matched*. Choosing the feasible threshold that minimises cost
    (or false positives) is not enough: an algorithm that can reach recall 0.95
    is then allowed to be less selective than one pinned at 0.90, and it wins on
    a technicality. So among the feasible thresholds this picks the one with the
    **lowest** recall (the most selective decision that still clears the floor)
    and only then minimises cost within that recall level.

    With that, every algorithm in the comparison is answering the same question:
    "at 90% recall, how many false positives do you produce?"
    """
    _validate_inputs(scores, labels)
    if not scores:
        return Confusion(0.5, 0, 0, 0, 0)
    if sum(labels) == 0:
        return cost_at(scores, labels, math.nextafter(max(scores), math.inf))

    thresholds = _threshold_grid(scores)

    feasible = [
        cost_at(scores, labels, threshold, cost_fp, cost_fn)
        for threshold in thresholds
    ]
    feasible = [c for c in feasible if c.recall + 1e-9 >= min_recall]

    if not feasible:
        # The recall floor is unreachable; return the best available point so the
        # shortfall is visible in the reported recall rather than hidden.
        return max(feasible + [cost_at(scores, labels, t) for t in thresholds],
                   key=lambda c: c.recall)

    lowest_recall = min(c.recall for c in feasible)
    matched = [c for c in feasible if c.recall <= lowest_recall + 1e-9]
    return min(matched, key=lambda c: (c.cost(cost_fp, cost_fn), c.fp, -c.threshold))


def high_precision_point(scores: Sequence[float], labels: Sequence[int],
                         target_precision: float = 0.99) -> Confusion:
    """Lowest threshold whose precision reaches ``target_precision``.

    Used as the **auto-approve** cut-off. The cost-optimal threshold minimises
    total cost but is not the threshold a production pipeline should auto-act
    on: it sits low enough to catch nearly every match, which is the right
    trade-off when a false negative costs 25x a false positive and the
    downstream consequence of a false negative is a *human reviewing a case*.
    Deciding without a human in the loop is a different decision with different
    stakes, so it gets its own, stricter cut-off.
    """
    _validate_inputs(scores, labels)
    best: Confusion | None = None
    fallback: Confusion | None = None
    for threshold in _threshold_grid(scores):
        confusion = cost_at(scores, labels, threshold)
        if confusion.tp + confusion.fp == 0:
            continue
        if fallback is None or confusion.precision > fallback.precision:
            fallback = confusion
        if confusion.precision >= target_precision and (best is None or confusion.fn < best.fn):
            best = confusion
    if best is not None:
        return best

    # No threshold reaches the target. Return the most precise one available,
    # and record that the target was missed so a report can say so instead of
    # quietly presenting the fallback as if it had complied with the rule.
    if fallback is None:
        return Confusion(0.5, 0, 0, 0, 0, target_met=False)
    return replace(fallback, target_met=False)


def zero_fn_point(scores: Sequence[float], labels: Sequence[int]) -> Confusion:
    """Highest threshold that produces no false negative at all.

    Used as the **auto-reject** cut-off, and conservative by construction: it only
    rejects below a score that no true match in the data reached, so a genuine
    match can never be auto-rejected on this dataset. That is a statement
    about separability, not about cost -- auto-rejecting a real customer is
    still friction, and the cost model in :mod:`name_match.cli` is where the
    25:1 ratio that prices it is actually set.
    """
    _validate_inputs(scores, labels)
    candidates = sorted(set(scores))
    if not candidates:
        return Confusion(1.0, 0, 0, 0, 0)

    best: Confusion | None = None
    for threshold in candidates:
        confusion = cost_at(scores, labels, threshold)
        if confusion.fn == 0 and (best is None or confusion.fp < best.fp):
            best = confusion
    # No candidate is free of false negatives: the safest possible reject
    # cut-off is the lowest score in the data, which misses nothing.
    return best or Confusion(
        threshold=candidates[0], tp=sum(labels),
        fp=0, tn=sum(1 for label in labels if label == 0), fn=0,
    )


def band_thresholds(scores: Sequence[float], labels: Sequence[int],
                    cost_fp: float = DEFAULT_COST_FP,
                    cost_fn: float = DEFAULT_COST_FN,
                    target_precision: float = 0.99) -> tuple[float, float]:
    """Derive ``(reject_below, approve_at_or_above)`` for a three-band pipeline.

    The two boundaries are chosen from opposite ends of the error profile,
    because the two bands carry different risks:

    * ``approve_at_or_above`` -- precision-driven. At least
      ``target_precision`` of everything auto-approved is a real match.
    * ``reject_below`` -- recall-driven. No real match scores below it, so
      nothing genuine is auto-rejected.

    The two can cross when the algorithm has no usable separation at all
    (zero-FN cut-off above the precision cut-off). That is a genuine, reportable
    property of the algorithm rather than a bug, and it is normalised here into
    a degenerate two-band split: reject below ``min``, approve above ``max``,
    and send everything between to review. The caller should compare the
    returned pair against the raw cut-offs (both are returned by
    :func:`band_report`) to see whether the collapse happened.
    """
    approve = high_precision_point(scores, labels, target_precision)
    reject = zero_fn_point(scores, labels)

    low, high = reject.threshold, approve.threshold
    if low > high:
        # The cut-offs genuinely cross. Swapping them silently would be a quiet
        # lie: a caller trusting the tuple order would then read the precision
        # cut-off as the reject cut-off. Normalise so the arithmetic downstream
        # stays sound, and let `band_report` surface the degeneracy.
        low, high = high, low
    return low, high


def band_report(scores: Sequence[float], labels: Sequence[int],
                cost_fp: float = DEFAULT_COST_FP,
                cost_fn: float = DEFAULT_COST_FN,
                target_precision: float = 0.99) -> dict:
    """Full three-band diagnostic, including whether the bands collapsed."""
    approve = high_precision_point(scores, labels, target_precision)
    reject = zero_fn_point(scores, labels)

    # A band is only usable if it is non-empty. An algorithm whose lowest
    # possible reject cut-off is its minimum observed score rejects nothing
    # automatically -- that is a two-band pipeline whatever the cut-offs say, and
    # the cut-off check alone does not reveal it.
    # `predicted` counts rows predicted as a match; compare against the row
    # count, not the positive count.
    n_rows = len(labels)
    approve_empty = approve.predicted == 0
    reject_empty = reject.predicted == n_rows

    raw_reject_below = reject.threshold
    raw_approve_at_or_above = approve.threshold
    collapsed = raw_reject_below > raw_approve_at_or_above

    low, high = band_thresholds(scores, labels, cost_fp, cost_fn, target_precision)
    return {
        "reject_below": low,
        "approve_at_or_above": high,
        # The raw cut-offs, before `band_thresholds` normalises a crossing pair.
        # Published so a report can label each number correctly even in the
        # degenerate case where the two are swapped.
        "raw_reject_below": raw_reject_below,
        "raw_approve_at_or_above": raw_approve_at_or_above,
        "bands_collapsed": collapsed,
        "approve_band_empty": approve_empty,
        "reject_band_empty": reject_empty,
        # A crossing pair of cut-offs is itself a reason the pipeline is not
        # usable, even when both bands are non-empty: the rendered table would
        # show band sizes computed from cut-offs printed in the opposite order,
        # which reads as a contradiction.
        "three_band_pipeline_usable": not (approve_empty or reject_empty or collapsed),
        "target_precision": target_precision,
        "approve_precision": approve.precision,
        "approve_meets_target": approve.target_met,
        "approve_coverage": approve.recall,
        "reject_fn": reject.fn,
        "reject_coverage": reject.recall,
    }


# --------------------------------------------------------------------------
# Per-category breakdown
# --------------------------------------------------------------------------


def per_category_error(score_set: ScoreSet, threshold: float) -> list[dict[str, object]]:
    """False-negative and false-positive counts grouped by dataset category.

    This is the table that says *which kinds of noise* an algorithm fails on,
    which is the actionable output. An aggregate F1 cannot distinguish a matcher
    that is uniformly mediocre from one that is perfect except on surname-first
    documents.
    """
    groups: dict[str, dict[str, int]] = {}

    for score, label, row in zip(score_set.scores, score_set.labels, score_set.rows):
        category = getattr(row, "category", "unknown")
        group = groups.setdefault(category, {"n": 0, "pos": 0, "neg": 0, "fn": 0, "fp": 0})
        group["n"] += 1
        predicted = 1 if score >= threshold else 0
        if label == 1:
            group["pos"] += 1
            if predicted == 0:
                group["fn"] += 1
        else:
            group["neg"] += 1
            if predicted == 1:
                group["fp"] += 1

    output: list[dict[str, object]] = []
    for category in sorted(groups):
        group = groups[category]
        output.append({
            "category": category,
            "n": group["n"],
            "positives": group["pos"],
            "negatives": group["neg"],
            "false_negatives": group["fn"],
            "false_positives": group["fp"],
            "fn_rate": (group["fn"] / group["pos"]) if group["pos"] else float("nan"),
            "fp_rate": (group["fp"] / group["neg"]) if group["neg"] else float("nan"),
        })
    return output


def score_all(matcher: Callable[[str, str], float],
              pairs: Iterable) -> ScoreSet:
    """Score a matcher over a list of dataset rows."""
    rows = list(pairs)
    scores = [matcher(row.name_a, row.name_b) for row in rows]
    labels = [row.label for row in rows]
    # `hasattr(matcher, "__self__")` is not the guard it looks like: every
    # bound method has `__self__`, and most objects do not carry a `.key`, so
    # passing an arbitrary bound method raised AttributeError here rather than
    # falling back. Check the attribute we actually want.
    owner = getattr(matcher, "__self__", None)
    key = getattr(owner, "key", None) or "matcher"
    return ScoreSet(
        key=key,
        label=str(owner if owner is not None else matcher),
        scores=scores,
        labels=labels,
        rows=rows,
    )


#: Accurate name for :func:`interpolated_precision_at_recall`. The original
#: name said "interpolated" because that is what the function used to do, and
#: the interpolation *was* the bug -- it reported precisions that no threshold
#: could attain. It is now a recall-floor maximum and interpolates nothing. The
#: old name is kept as an alias so existing references still resolve; new code
#: should use this one.
precision_at_recall = interpolated_precision_at_recall
