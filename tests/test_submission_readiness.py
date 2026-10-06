"""Checks for submission-review findings using hand-checkable examples."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import unittest

from name_match import cli, evaluate
from name_match.algorithms import ALL_MATCHERS, LearnedCombiner
from name_match.features import FEATURE_NAMES
from name_match.model import LogisticRegression
from name_match.normalize import normalize


class TestThresholdSelection(unittest.TestCase):
    def test_reject_everything_can_be_cost_optimal(self):
        scores, labels = [0.5] * 4, [1, 0, 0, 0]
        point = evaluate.find_operating_point(scores, labels, cost_fp=25, cost_fn=1)
        self.assertEqual((point.tp, point.fp, point.tn, point.fn), (0, 0, 3, 1))
        self.assertGreater(point.threshold, max(scores))

    def test_cost_optimum_matches_exhaustive_decisions(self):
        rng = random.Random(43)
        for _ in range(50):
            scores = [rng.choice([0.0, 0.3, 0.7, 1.0]) for _ in range(10)]
            labels = [rng.randrange(2) for _ in scores]
            for cost_fp, cost_fn in [(1, 25), (25, 1), (1, 1)]:
                best = evaluate.find_operating_point(scores, labels, cost_fp, cost_fn)
                decisions = [evaluate.cost_at(scores, labels, t)
                             for t in [0.0, 0.3, 0.7, 1.0, 1.1]]
                self.assertEqual(best.cost(cost_fp, cost_fn),
                                 min(c.cost(cost_fp, cost_fn) for c in decisions))

    def test_empty_precision_target_is_not_met(self):
        point = evaluate.high_precision_point([], [])
        self.assertFalse(point.target_met)
        self.assertEqual(point.fn, 0)
        self.assertTrue(math.isnan(point.precision))

    def test_empty_zero_fp_result_has_no_misses(self):
        point = evaluate.zero_fp_point([], [])
        self.assertEqual((point.tp, point.fp, point.tn, point.fn), (0, 0, 0, 0))

    def test_no_positive_recall_result_preserves_negative_count(self):
        point = evaluate.find_operating_point_at_recall([0.9, 0.8], [0, 0])
        self.assertEqual((point.tp, point.fp, point.tn, point.fn), (0, 0, 2, 0))

    def test_custom_costs_are_preserved_in_analysis(self):
        scores = evaluate.ScoreSet("tiny", "tiny", [0.9, 0.8], [1, 0])
        result = cli.analyse([scores], cost_fp=7, cost_fn=3)["tiny"]
        for key in ("optimal", "zero_fp", "at_recall_floor"):
            row = result[key]
            self.assertEqual(row["cost"], 7 * row["fp"] + 3 * row["fn"])
            self.assertEqual(row["cost_per_pair"], row["cost"] / 2)


class TestInvalidMetricInputs(unittest.TestCase):
    def test_mismatched_or_invalid_inputs_are_rejected(self):
        functions = [evaluate.find_operating_point, evaluate.zero_fp_point,
                     evaluate.zero_fn_point, evaluate.high_precision_point,
                     evaluate.pr_auc, evaluate.roc_auc,
                     evaluate.precision_at_recall,
                     evaluate.find_operating_point_at_recall]
        for function in functions:
            for scores, labels in [([0.5], []), ([0.5], [2]),
                                   ([float("nan")], [1]),
                                   ([float("inf")], [1])]:
                with self.subTest(function=function.__name__, scores=scores, labels=labels):
                    with self.assertRaises(ValueError):
                        function(scores, labels)

    def test_cost_at_does_not_truncate_pairs(self):
        with self.assertRaises(ValueError):
            evaluate.cost_at([0.9, 0.8], [1], 0.5)

    def test_bootstrap_requires_resamples(self):
        with self.assertRaises(ValueError):
            evaluate.bootstrap_cost_delta([0.9], [1], [0.9], 0.5, 0.5, iterations=0)

    def test_band_boundaries_must_be_ordered(self):
        with self.assertRaises(ValueError):
            evaluate.three_band_split([0.9], [1], low=0.8, high=0.2)

    def test_cross_validation_rejects_empty_training_folds(self):
        for n, folds in [(0, 5), (1, 1), (4, 5), (10, 0)]:
            with self.subTest(n=n, folds=folds):
                with self.assertRaises(ValueError):
                    cli.kfold_indices(n, folds)


class TestModelAndNormalisation(unittest.TestCase):
    def test_corrupt_model_files_fail_before_prediction(self):
        model = LogisticRegression(iterations=1).fit([[0.0], [1.0]], [0, 1], ["x"])
        mutations = [("stds", []), ("stds", [0.0]), ("stds", [-1.0]),
                     ("stds", [float("inf")]), ("means", []),
                     ("bias", float("nan")), ("weights", None),
                     ("feature_names", ["x", "y"])]
        with tempfile.TemporaryDirectory() as directory:
            target = os.path.join(directory, "model.json")
            for key, value in mutations:
                payload = model.to_dict()
                payload[key] = value
                with open(target, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle)
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        LogisticRegression.load(target)

    def test_nonfinite_training_and_prediction_features_are_rejected(self):
        with self.assertRaises(ValueError):
            LogisticRegression().fit([[float("nan")]], [1], ["x"])
        model = LogisticRegression(iterations=1).fit([[0.0], [1.0]], [0, 1], ["x"])
        with self.assertRaises(ValueError):
            model.predict_proba([float("inf")])

    def test_same_width_wrong_feature_order_is_rejected(self):
        names = tuple(reversed(FEATURE_NAMES))
        model = LogisticRegression(iterations=1).fit(
            [[0.0] * len(names), [1.0] * len(names)], [0, 1], names)
        with self.assertRaises(ValueError):
            LearnedCombiner(model).score("Suresh Kumar", "Suresh Kumar")

    def test_mixed_script_token_does_not_silently_become_exact_match(self):
        left, right = "Rahulशर्मा Kumar", "Rahul Kumar"
        self.assertTrue(normalize(left).undecidable)
        for matcher in ALL_MATCHERS:
            with self.subTest(matcher=matcher.key):
                result = matcher.explain(left, right)
                self.assertEqual(result.score, 0.5)
                self.assertEqual(result.components["unsupported_script"], 1.0)

    def test_latin_diacritics_remain_supported(self):
        self.assertFalse(normalize("Sūresh Søren Kumar").undecidable)


if __name__ == "__main__":
    unittest.main()
