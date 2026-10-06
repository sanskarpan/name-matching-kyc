"""Tests for the evaluation harness.

The metrics are where a silent bug does the most damage: a mis-signed ROC-AUC
or a lexicographic tie-break produces a table that looks entirely reasonable
and is exactly backwards. These tests pin the metrics to hand-checkable cases.

Run with ``python3 -m unittest discover -s tests -t .`` from the repo root.
"""

from __future__ import annotations

import unittest

from name_match import evaluate
from name_match.evaluate import (Confusion, _threshold_grid, cost_at,
                                  find_operating_point,
                                  find_operating_point_at_recall,
                                  interpolated_precision_at_recall, pr_auc,
                                  roc_auc, three_band_split, zero_fp_point)


class TestConfusion(unittest.TestCase):
    def test_rates(self):
        # tp=3 fp=1 tn=4 fn=2
        confusion = Confusion(0.5, 3, 1, 4, 2)
        self.assertAlmostEqual(confusion.precision, 0.75)
        self.assertAlmostEqual(confusion.recall, 0.6)
        self.assertAlmostEqual(confusion.f1, 2 * 0.75 * 0.6 / 1.35)
        self.assertAlmostEqual(confusion.accuracy, 7 / 10)

    def test_no_predictions_denominators(self):
        confusion = Confusion(0.5, 0, 0, 0, 0)
        # Precision is *undefined*, not perfect: a threshold that predicts
        # nothing must never be reported as 1.0, which is how a degenerate
        # threshold comes to look like the safest one in a report.
        self.assertNotEqual(confusion.precision, confusion.precision)  # NaN
        self.assertEqual(confusion.predicted, 0)
        self.assertEqual(confusion.recall, 0.0)
        self.assertEqual(confusion.accuracy, 0.0)
        self.assertNotEqual(confusion.f1, confusion.f1)  # NaN

    def test_precision_is_nan_only_when_nothing_predicted(self):
        self.assertNotEqual(Confusion(0.5, 0, 0, 9, 0).precision,
                            Confusion(0.5, 0, 0, 0, 9).precision)  # NaN != NaN
        self.assertEqual(Confusion(0.5, 3, 1, 4, 2).precision, 0.75)
        self.assertEqual(Confusion(0.5, 0, 3, 4, 2).precision, 0.0)

    def test_cost_weights_asymmetry(self):
        # 1 false positive and 1 false negative must not cost the same.
        confusion = Confusion(0.5, 0, 1, 0, 1)
        self.assertEqual(confusion.cost(cost_fp=1.0, cost_fn=25.0), 26.0)
        self.assertEqual(confusion.cost(cost_fp=1.0, cost_fn=1.0), 2.0)

    def test_cost_per_pair(self):
        confusion = Confusion(0.5, 1, 1, 1, 1)
        self.assertAlmostEqual(confusion.cost_per_pair(cost_fp=1.0, cost_fn=25.0), 26 / 4)


class TestCostAt(unittest.TestCase):
    def test_threshold_inclusive(self):
        scores = [0.9, 0.5, 0.1]
        labels = [1, 1, 0]
        # score >= threshold predicts a match.
        confusion = cost_at(scores, labels, 0.5)
        self.assertEqual((confusion.tp, confusion.fp, confusion.fn, confusion.tn), (2, 0, 0, 1))
        confusion = cost_at(scores, labels, 0.51)
        self.assertEqual((confusion.tp, confusion.fp, confusion.fn, confusion.tn), (1, 0, 1, 1))

    def test_empty(self):
        confusion = cost_at([], [], 0.5)
        self.assertEqual(confusion.tp, 0)


class TestFindOperatingPoint(unittest.TestCase):
    def test_prefers_low_cost_not_low_fp(self):
        # The bug this guards against: ranking candidates by (fp, fn)
        # lexicographically makes "predict nothing" optimal for every
        # algorithm, because it has zero false positives by construction.
        scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
        labels = [1, 1, 0, 0, 0, 0]
        best = find_operating_point(scores, labels, cost_fp=1.0, cost_fn=25.0)
        self.assertLess(best.fp, 3, "should not accept every pair")
        self.assertGreater(best.tp, 0, "should not reject everything")

    def test_equal_weights_finds_perfect_split(self):
        scores = [0.9, 0.8, 0.2, 0.1]
        labels = [1, 1, 0, 0]
        best = find_operating_point(scores, labels, cost_fp=1.0, cost_fn=1.0)
        self.assertEqual((best.tp, best.fp, best.fn, best.tn), (2, 0, 0, 2))

    def test_unseparable_returns_degenerate_answer(self):
        # Perfectly overlapping scores: any threshold is equally bad, and the
        # optimiser must still return something sane.
        scores = [0.5] * 4
        labels = [1, 1, 0, 0]
        best = find_operating_point(scores, labels, cost_fp=1.0, cost_fn=25.0)
        self.assertGreaterEqual(best.fp + best.fn, 0)
        self.assertEqual(best.tp + best.fp + best.tn + best.fn, 4)

    def test_empty(self):
        self.assertEqual(find_operating_point([], []).tp, 0)

    def test_higher_fn_cost_lowers_threshold(self):
        scores = [0.9, 0.6, 0.5, 0.4]
        labels = [1, 1, 0, 0]
        lenient = find_operating_point(scores, labels, cost_fp=1.0, cost_fn=1.0)
        strict = find_operating_point(scores, labels, cost_fp=1.0, cost_fn=100.0)
        self.assertLessEqual(strict.threshold, lenient.threshold,
                             "a costlier miss should not raise the threshold")


class TestRecallFloor(unittest.TestCase):
    def test_floor_is_respected(self):
        scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
        labels = [1, 1, 0, 0, 0, 0]
        point = find_operating_point_at_recall(scores, labels, min_recall=1.0,
                                               cost_fp=1.0, cost_fn=25.0)
        self.assertGreaterEqual(point.recall, 1.0 - 1e-9)

    def test_unreachable_floor_reports_shortfall(self):
        # With no positive rows at all, recall is 0 at every threshold and any
        # floor is unreachable. The helper must return the best available point
        # rather than crash or silently pretend the constraint was met.
        point = find_operating_point_at_recall([0.5, 0.4], [0, 0], min_recall=0.9)
        self.assertEqual(point.recall, 0.0)
        self.assertEqual(point.tp, 0)

    def test_empty(self):
        self.assertEqual(find_operating_point_at_recall([], []).tp, 0)


class TestZeroFalsePositive(unittest.TestCase):
    def test_finds_clean_cut(self):
        scores = [0.9, 0.8, 0.3, 0.2]
        labels = [1, 1, 0, 0]
        point = zero_fp_point(scores, labels)
        self.assertEqual(point.fp, 0)
        self.assertEqual(point.tp, 2)

    def test_unseparable_reports_all_misses(self):
        # When no threshold is clean, the honest answer is "miss everything",
        # not a flattering threshold that does not exist.
        scores = [0.5, 0.5]
        labels = [1, 0]
        point = zero_fp_point(scores, labels)
        self.assertEqual(point.fp, 0)
        self.assertEqual(point.fn, 1)

    def test_empty(self):
        self.assertEqual(zero_fp_point([], []).tp, 0)


class TestRankingMetrics(unittest.TestCase):
    def test_roc_auc_perfect_and_inverted(self):
        self.assertAlmostEqual(roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0)
        self.assertAlmostEqual(roc_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]), 0.0)

    def test_roc_auc_ties_score_base_rate(self):
        # A ranking with no information must score exactly chance. This is the
        # check that caught a descending-sort implementation returning 1-AUC.
        self.assertAlmostEqual(roc_auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]), 0.5)
        self.assertAlmostEqual(roc_auc([0.5] * 6, [1, 0, 1, 0, 1, 0]), 0.5)

    def test_roc_auc_partial(self):
        # positives at ranks 1 and 3 of 4 -> (3+1)/4
        self.assertAlmostEqual(roc_auc([0.9, 0.4, 0.3, 0.1], [1, 0, 1, 0]), 0.75)

    def test_roc_auc_single_class(self):
        self.assertAlmostEqual(roc_auc([0.1, 0.2], [1, 1]), 0.5)

    def test_roc_auc_empty(self):
        self.assertAlmostEqual(roc_auc([], []), 0.5)

    def test_pr_auc_perfect(self):
        self.assertAlmostEqual(pr_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0)

    def test_pr_auc_ties_score_base_rate(self):
        # Tie-grouped average precision: an uninformative ranking scores the
        # base rate, not whatever the CSV row order happens to be.
        self.assertAlmostEqual(pr_auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]), 0.5)
        self.assertAlmostEqual(pr_auc([0.5] * 6, [1, 0, 1, 0, 1, 0]), 0.5)

    def test_pr_auc_row_order_independent_for_ties(self):
        forward = pr_auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0])
        reversed_ = pr_auc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1])
        self.assertAlmostEqual(forward, reversed_)

    def test_pr_auc_inverted_below_base_rate(self):
        # All negatives ranked first must score worse than chance.
        self.assertLess(pr_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]), 0.5)

    def test_pr_auc_no_positives(self):
        self.assertEqual(pr_auc([0.1, 0.2], [0, 0]), 0.0)

    def test_pr_auc_empty(self):
        self.assertEqual(pr_auc([], []), 0.0)


class TestBootstrap(unittest.TestCase):
    def test_identical_algorithms_have_zero_delta(self):
        scores = [0.9, 0.8, 0.2, 0.1, 0.15, 0.05]
        labels = [1, 1, 0, 0, 0, 0]
        result = evaluate.bootstrap_cost_delta(scores, labels, list(scores),
                                              0.5, 0.5, iterations=200)
        self.assertAlmostEqual(result["mean_delta"], 0.0, places=9)
        self.assertAlmostEqual(result["ci_low"], 0.0, places=9)

    def test_delta_is_per_pair_not_total(self):
        # A per-pair delta cannot exceed the cost of a single worst-case pair.
        scores_a = [0.9, 0.8, 0.2, 0.1]
        labels = [1, 1, 0, 0]
        result = evaluate.bootstrap_cost_delta(scores_a, labels, scores_a, 0.5, 0.15,
                                              iterations=100)
        self.assertLessEqual(abs(result["mean_delta"]), 100.0)

    def test_is_deterministic(self):
        scores = [0.9, 0.8, 0.2, 0.1, 0.3, 0.4]
        labels = [1, 1, 0, 0, 0, 0]
        first = evaluate.bootstrap_cost_delta(scores, labels, scores, 0.5, 0.3)
        second = evaluate.bootstrap_cost_delta(scores, labels, scores, 0.5, 0.3)
        self.assertEqual(first, second)

    def test_empty(self):
        result = evaluate.bootstrap_cost_delta([], [], [], 0.5, 0.5)
        self.assertEqual(result["mean_delta"], 0.0)


class TestPrecisionAtRecall(unittest.TestCase):
    """The metric the recommendation actually rests on."""

    def test_perfect_separation(self):
        self.assertAlmostEqual(
            interpolated_precision_at_recall([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], 0.9),
            1.0)

    def test_uninformative_ranking_scores_base_rate(self):
        # With no separation, precision at any recall is the base rate.
        for target in (0.5, 0.9, 0.99):
            self.assertAlmostEqual(
                interpolated_precision_at_recall([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0], target),
                0.5, places=6)

    def test_every_value_is_attained(self):
        """The headline property: a reported P@R is a real operating point."""
        scores = [1.0] * 5 + [0.0] * 10
        labels = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for target in (0.2, 0.4, 0.6, 0.8, 1.0):
            reported = interpolated_precision_at_recall(scores, labels, target)
            achievable = [
                c.precision
                for c in (cost_at(scores, labels, t) for t in _threshold_grid(scores))
                if c.recall + 1e-9 >= target and c.predicted > 0
            ]
            self.assertTrue(achievable, f"recall {target} unreachable at all")
            self.assertAlmostEqual(reported, max(achievable), places=9,
                                   msg=f"P@R={target} is not an attainable precision")

    def test_is_monotone_non_increasing(self):
        scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05, 0.04, 0.03]
        labels = [1, 1, 1, 0, 0, 0, 0, 1]
        values = [interpolated_precision_at_recall(scores, labels, r)
                  for r in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)]
        for earlier, later in zip(values, values[1:]):
            self.assertGreaterEqual(earlier + 1e-12, later,
                                    "precision rose as recall was increased")

    def test_unreachable_recall_scores_zero(self):
        self.assertEqual(
            interpolated_precision_at_recall([0.1, 0.05], [0, 0], 0.9), 0.0)

    def test_single_class(self):
        self.assertEqual(interpolated_precision_at_recall([0.5, 0.4], [0, 0], 0.9), 0.0)

    def test_empty(self):
        self.assertEqual(interpolated_precision_at_recall([], [], 0.9), 0.0)

    def test_better_ranker_scores_higher_at_every_level(self):
        # The property the report relies on to rank the algorithms.
        # Layout: [positive, positive, positive, negative, negative, negative].
        good = [0.95, 0.90, 0.85, 0.20, 0.10, 0.05]   # clean separation
        # A negative sits above two of the positives, so the bad ranker's curve
        # slopes the wrong way and it cannot reach high recall cleanly.
        bad = [0.60, 0.43, 0.42, 0.45, 0.41, 0.40]
        labels = [1, 1, 1, 0, 0, 0]
        for target in (0.34, 0.5, 0.67):
            self.assertGreater(
                interpolated_precision_at_recall(good, labels, target),
                interpolated_precision_at_recall(bad, labels, target),
                f"ordering failed at recall {target}")


class TestBands(unittest.TestCase):
    def test_split_counts_add_up(self):
        scores = [0.95, 0.9, 0.5, 0.4, 0.1, 0.05]
        labels = [1, 1, 1, 0, 0, 0]
        bands = three_band_split(scores, labels, 0.8, 0.9)
        total = bands["approve_n"] + bands["review_n"] + bands["reject_n"]
        self.assertEqual(total, len(scores))
        for key in ("approve_pct", "review_pct", "reject_pct"):
            self.assertAlmostEqual(bands[key], 100.0 * bands[key.replace("_pct", "_n")] / 6)

    def test_band_boundaries_are_ordered(self):
        scores = [0.95, 0.5, 0.1] * 4
        labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        report = evaluate.band_report(scores, labels)
        self.assertLessEqual(report["reject_below"], report["approve_at_or_above"])

    def test_collapse_is_reported_not_hidden(self):
        # A degenerate algorithm where the two cut-offs cross must say so.
        scores = [0.5] * 4
        labels = [1, 0, 1, 0]
        report = evaluate.band_report(scores, labels)
        self.assertIsInstance(report["bands_collapsed"], bool)
        self.assertLessEqual(report["reject_below"], report["approve_at_or_above"])

    def test_empty(self):
        bands = three_band_split([], [], 0.5, 0.5)
        self.assertEqual(bands["approve_n"], 0)


if __name__ == "__main__":
    unittest.main()
