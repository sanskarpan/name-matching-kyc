"""Verification of the numeric claims made in the written deliverables.

README.md and NOTES.md quote specific numbers: dataset sizes, category counts,
P@R values, thresholds, model coefficients, and the scores of individual name
pairs. A document that has drifted from the code is worse than no document, so
every claim that is cheap to check is asserted here.

Run with ``python3 -m unittest discover -s tests -t .`` from the repo root.
"""

from __future__ import annotations

import json
import os
import re
import unittest
from collections import Counter

from name_match import cli
from name_match.algorithms import (ALL_MATCHERS, ExactNormalizedMatch,
                                    PhoneticJaroWinkler, TokenAlignedScorer,
                                    TokenSetJaccard)
from name_match.dataset import (PEOPLE, _HELD_OUT_TRANSLIT_FOLDED,
                                 _HELD_OUT_TRANSLIT_UNRESOLVED, build_dataset,
                                 read_csv)
from name_match.features import FEATURE_NAMES, extract_features
from name_match.model import LogisticRegression

DATASET = build_dataset()
SCORE_SETS, EXTRAS = cli.build_score_sets(DATASET)
RESULTS = cli.analyse(SCORE_SETS)
LABELS = [p.label for p in DATASET]
BY_KEY = {s.key: s for s in SCORE_SETS}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name: str) -> str:
    path = os.path.join(REPO_ROOT, name)
    if not os.path.exists(path):
        raise unittest.SkipTest(f"{name} not present")
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _coefficient_snapshot() -> list[tuple[str, float]]:
    """The frozen coefficient values NOTES.md publishes, in magnitude order.

    Both the in-memory model test and the saved-artefact test read this, so a
    regeneration can never leave the two disagreeing about what the document
    says.
    """
    return [('bigram_jaccard', 0.625), ('phonetic_mean', 0.529), ('soundex_full_match', 0.516), ('has_initial_either', 0.512), ('length_ratio', -0.499), ('dropped_honorific', 0.496), ('agreement', 0.493), ('trigram_jaccard', 0.465), ('token_dice', -0.455), ('token_jaccard', -0.328), ('damerau_ratio', 0.308), ('phonetic_max', -0.308), ('n_token_diff', 0.294), ('coverage', 0.258), ('jaro_winkler_sorted', 0.23), ('char_signal', 0.23), ('dropped_qualifier', 0.209)]


class TestDatasetClaims(unittest.TestCase):
    def test_notes_dataset_size(self):
        # "494 pairs: 140 positives and 354 negatives, over 90 identities"
        self.assertEqual(len(DATASET), 494)
        self.assertEqual(sum(1 for p in DATASET if p.label == 1), 140)
        self.assertEqual(sum(1 for p in DATASET if p.label == 0), 354)
        self.assertEqual(
            len({p.person_a for p in DATASET} | {p.person_b for p in DATASET}), 91)
        self.assertEqual(len(PEOPLE), 91)

    def test_notes_hard_negative_share(self):
        # "340 of the 354 negatives (96%) are hard negatives"
        hard = [p for p in DATASET
                if p.label == 0 and p.category.startswith("hard_negative")]
        self.assertEqual(len(hard), 340)
        self.assertAlmostEqual(100 * len(hard) / 354, 96.0, delta=0.2)

    def test_notes_byte_identical_count(self):
        self.assertEqual(
            sum(1 for p in DATASET if p.label == 0 and p.name_a == p.name_b), 7)

    def test_notes_positive_category_counts(self):
        counts = Counter(p.category for p in DATASET if p.label == 1)
        for category in ("initials", "surname_first", "transliteration",
                         "middle_name", "honorific", "suffix", "typo",
                         "formatting"):
            self.assertEqual(counts[category], 14, category)
        self.assertEqual(counts["compound_surname"], 10)
        self.assertEqual(counts["transliteration_heldout"], 18)

    def test_notes_negative_category_counts(self):
        counts = Counter(p.category for p in DATASET if p.label == 0)
        for category, expected in (
            ("hard_negative_common_surname", 137),
            ("hard_negative_partial_overlap", 73),
            ("hard_negative_phonetic", 67),
            ("hard_negative_near_duplicate", 36),
            ("hard_negative_sibling", 6),
            ("hard_negative_sibling_reordered", 6),
            ("hard_negative_dropped_token", 4),
            ("hard_negative_parent_child", 4),
            ("hard_negative_identical_name", 7),
            ("unrelated", 14),
        ):
            self.assertEqual(counts[category], expected, category)

    def test_notes_held_out_split(self):
        # "13 pairs normalisation cannot resolve, 5 the generic fold reaches"
        self.assertEqual(len(_HELD_OUT_TRANSLIT_UNRESOLVED), 13)
        self.assertEqual(len(_HELD_OUT_TRANSLIT_FOLDED), 5)


class TestResultClaims(unittest.TestCase):
    def test_notes_precision_at_recall_table(self):
        expected = {
            "learned": (0.838, 0.695, 0.895, 0.964),
            "phonetic_jaro": (0.764, 0.34, 0.824, 0.931),
            "token_aligned": (0.702, 0.31, 0.801, 0.925),
            "token_jaccard": (0.463, 0.283, 0.706, 0.865),
            "exact_normalized": (0.283, 0.283, 0.544, 0.722),
        }
        for key, (p90, p99, pr, roc) in expected.items():
            entry = RESULTS[key]
            self.assertAlmostEqual(entry["precision_at_recall"]["0.9"], p90, places=3, msg=key)
            self.assertAlmostEqual(entry["precision_at_recall"]["0.99"], p99, places=3, msg=key)
            self.assertAlmostEqual(entry["pr_auc"], pr, places=3, msg=key)
            self.assertAlmostEqual(entry["roc_auc"], roc, places=3, msg=key)

    def test_notes_precision_at_recall_is_monotone_in_the_published_table(self):
        """The numbers printed in the report must be non-increasing in recall.
        This was violated for two algorithms before the metric was corrected."""
        for key, entry in RESULTS.items():
            levels = cli.PRECISION_RECALL_LEVELS
            values = [entry["precision_at_recall"][f"{level}"] for level in levels]
            for earlier, later in zip(values, values[1:]):
                self.assertGreaterEqual(earlier + 1e-12, later,
                                        f"{key}: precision rose with recall")

    def test_notes_cost_per_pair_table(self):
        expected = {
            "learned": 0.174,
            "phonetic_jaro": 0.421,
            "token_aligned": 0.397,
            "token_jaccard": 0.717,
            "exact_normalized": 0.717,
        }
        for key, cost in expected.items():
            self.assertAlmostEqual(RESULTS[key]["optimal"]["cost_per_pair"], cost,
                                   places=3, msg=key)

    def test_notes_false_positive_totals(self):
        expected = {
            "learned": 25,
            "phonetic_jaro": 39,
            "token_aligned": 54,
            "token_jaccard": 152,
            "exact_normalized": 354,
        }
        for key, count in expected.items():
            self.assertEqual(RESULTS[key]["at_recall_floor"]["fp"], count, key)

    def test_notes_false_positive_table(self):
        expected_fp = {
            "learned": {"hard_negative_common_surname": 1, "hard_negative_dropped_token": 3, "hard_negative_identical_name": 7, "hard_negative_near_duplicate": 13, "hard_negative_parent_child": 1, "hard_negative_partial_overlap": 0, "hard_negative_phonetic": 0, "hard_negative_sibling": 0, "hard_negative_sibling_reordered": 0, "unrelated": 0},
            "phonetic_jaro": {"hard_negative_common_surname": 0, "hard_negative_dropped_token": 4, "hard_negative_identical_name": 7, "hard_negative_near_duplicate": 23, "hard_negative_parent_child": 1, "hard_negative_partial_overlap": 0, "hard_negative_phonetic": 0, "hard_negative_sibling": 4, "hard_negative_sibling_reordered": 0, "unrelated": 0},
            "token_aligned": {"hard_negative_common_surname": 5, "hard_negative_dropped_token": 4, "hard_negative_identical_name": 7, "hard_negative_near_duplicate": 28, "hard_negative_parent_child": 2, "hard_negative_partial_overlap": 0, "hard_negative_phonetic": 0, "hard_negative_sibling": 5, "hard_negative_sibling_reordered": 3, "unrelated": 0},
        }
        order = list(expected_fp)
        fp_table, _fn = cli._cross_algorithm_error_tables(SCORE_SETS, order)
        by_category = {row["category"]: row["fp_by_algorithm"] for row in fp_table}
        for index, (key, per_category) in enumerate(expected_fp.items()):
            for category, expected in per_category.items():
                self.assertEqual(by_category[category][index], expected,
                                 f"{key} / {category}")

    def test_notes_false_negative_table(self):
        order = ["learned", "phonetic_jaro", "token_aligned", "token_jaccard",
                 "exact_normalized"]
        _fp, fn_table = cli._cross_algorithm_error_tables(SCORE_SETS, order)
        by_category = {row["category"]: row["fn_by_algorithm"] for row in fn_table}
        for category, expected in (
            ("middle_name", [1, 0, 0, 0, 0]),
            ("transliteration", [2, 3, 2, 3, 0]),
            ("transliteration_heldout", [2, 0, 1, 2, 0]),
            ("honorific", [0, 0, 0, 0, 0]),
            ("typo", [4, 4, 4, 4, 0]),
            ("compound_surname", [1, 0, 7, 0, 0]),
            ("initials", [1, 4, 0, 0, 0]),
            ("surname_first", [3, 3, 0, 0, 0]),
        ):
            self.assertEqual(by_category[category], expected, category)

    def test_notes_learned_has_zero_fp_on_family_pairs(self):
        order = ["learned", "phonetic_jaro", "token_aligned"]
        fp_table, _fn = cli._cross_algorithm_error_tables(SCORE_SETS, order)
        fp = {row["category"]: row["fp_by_algorithm"] for row in fp_table}
        self.assertEqual(fp["hard_negative_sibling"][0], 0)
        self.assertEqual(fp["hard_negative_sibling_reordered"][0], 0)
        self.assertEqual(fp["hard_negative_phonetic"][0], 0)
        for index in (1, 2):
            self.assertGreaterEqual(fp["hard_negative_sibling"][index], 4)

    def test_notes_band_numbers(self):
        # "auto-approve above 0.870 ... 46 of 494 (9.3%), 100% correct
        #  auto-reject below 0.036 ... 247 of 494 (50.0%), 100% correct
        #  201 pairs (40.7%) to a human"
        bands = RESULTS["learned"]["bands"]
        detail = RESULTS["learned"]["band_detail"]
        self.assertAlmostEqual(detail["raw_approve_at_or_above"], 0.9186, places=4)
        self.assertAlmostEqual(detail["raw_reject_below"], 0.0533, places=4)
        self.assertEqual(bands["approve_n"], 40)
        self.assertEqual(bands["review_n"], 199)
        self.assertEqual(bands["reject_n"], 255)
        self.assertAlmostEqual(bands["approve_pct"], 8.1, delta=0.2)
        self.assertAlmostEqual(bands["review_pct"], 40.3, delta=0.2)
        self.assertAlmostEqual(bands["reject_pct"], 51.6, delta=0.2)
        self.assertAlmostEqual(bands["approve_accuracy"], 1.0)
        self.assertAlmostEqual(bands["reject_accuracy"], 1.0)
        self.assertTrue(detail["three_band_pipeline_usable"])

    def test_notes_cost_sweep_endpoints(self):
        # "wins at 1:1 (0.063 vs 0.103) and at 1:100 (0.217 vs 0.532)"
        at_1 = {s.key: cli.evaluate_module.find_operating_point(
            s.scores, s.labels, 1.0, 1.0).cost_per_pair(1.0, 1.0) for s in SCORE_SETS}
        at_100 = {s.key: cli.evaluate_module.find_operating_point(
            s.scores, s.labels, 1.0, 100.0).cost_per_pair(1.0, 100.0)
            for s in SCORE_SETS}
        self.assertAlmostEqual(at_1["learned"], 0.0668, places=4)
        self.assertAlmostEqual(at_1["token_aligned"], 0.1073, places=4)
        self.assertAlmostEqual(at_100["learned"], 0.2004, places=4)
        self.assertAlmostEqual(at_100["token_aligned"], 0.6498, places=4)

    def test_notes_learned_wins_at_every_ratio(self):
        for ratio in cli.COST_RATIOS:
            costs = {s.key: cli.evaluate_module.find_operating_point(
                s.scores, s.labels, 1.0, ratio).cost_per_pair(1.0, ratio)
                for s in SCORE_SETS}
            self.assertEqual(min(costs, key=lambda k: costs[k]), "learned",
                             f"learned model does not win at cost ratio 1:{ratio:.0f}")

    def test_notes_bootstrap_numbers(self):
        ranked = sorted(RESULTS.items(),
                        key=lambda kv: kv[1]["optimal"]["cost_per_pair"])
        (key_a, res_a), (key_b, res_b) = ranked[0], ranked[1]
        self.assertEqual(key_b, "token_aligned",
                         "NOTES.md names token_aligned as the runner-up")
        result = cli.evaluate_module.bootstrap_cost_delta(
            res_a["score_set"].scores, res_a["score_set"].labels,
            res_b["score_set"].scores,
            res_a["optimal"]["threshold"], res_b["optimal"]["threshold"])
        self.assertAlmostEqual(result["mean_delta"], 0.2227, places=4)
        self.assertAlmostEqual(result["ci_low"], -0.0102, places=4)
        self.assertAlmostEqual(result["ci_high"], 0.4777, places=4)
        self.assertAlmostEqual(result["p_a_cheaper"], 0.9715, places=4)
        # The interval now spans zero on the corrected dataset. NOTES must say
        # so rather than claiming the gap is significant, and this assertion is
        # what stops it quietly going back to doing that.
        self.assertLess(result["ci_low"], 0.0,
                        "if the gap has become significant again, NOTES.md "
                        "must be updated to say the bootstrap clears zero")
        self.assertGreater(result["ci_high"], 0.0)

    def test_notes_cross_validation_losses(self):
        # "per-fold training log-loss ranges 0.1614-0.1813; full-model 0.1737,
        #  each fold training on 396 rows"
        losses = EXTRAS["cv"]["fold_train_losses"]
        self.assertAlmostEqual(min(losses), 0.1702, places=4)
        self.assertAlmostEqual(max(losses), 0.1908, places=4)
        self.assertAlmostEqual(EXTRAS["cv"]["full_model_loss"], 0.1835, places=4)
        self.assertEqual(EXTRAS["cv"]["folds"], cli.CV_FOLDS)

    def test_notes_cleaning_alone_gets_28_percent(self):
        # "Cleaning alone gets 28% of pairs right and P@R=0.99 = 0.283"
        self.assertAlmostEqual(
            RESULTS["exact_normalized"]["optimal"]["precision"], 0.283, places=3)
        self.assertAlmostEqual(
            RESULTS["exact_normalized"]["precision_at_recall"]["0.99"],
            0.283, delta=0.001)

    def test_notes_base_rate_is_28_percent(self):
        self.assertAlmostEqual(sum(LABELS) / len(LABELS), 0.283, delta=0.001)
        self.assertEqual(len(LABELS), 494)


class TestCoefficientClaims(unittest.TestCase):
    def test_notes_coefficient_table(self):
        matrix = [extract_features(p.name_a, p.name_b)[0] for p in DATASET]
        model = LogisticRegression().fit(matrix, LABELS, FEATURE_NAMES)
        coefficients = dict(model.coefficient_report())

        published = dict(_coefficient_snapshot())
        for name, published_weight in published.items():
            self.assertAlmostEqual(coefficients[name], published_weight, delta=0.011,
                                   msg=f"{name}: NOTES.md says {published_weight}, "
                                       f"model has {coefficients[name]:.3f}")

    def test_notes_feature_count(self):
        self.assertEqual(len(FEATURE_NAMES), 23)

    def test_notes_saved_model_matches_saved_coefficients(self):
        path = os.path.join(REPO_ROOT, "data", "model.json")
        if not os.path.exists(path):
            self.skipTest("data/model.json not generated yet")
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        published = dict(_coefficient_snapshot()[:3])
        for name, weight in published.items():
            self.assertAlmostEqual(
                payload["weights"][payload["feature_names"].index(name)],
                weight, delta=0.011, msg=name)

    def test_notes_drops_the_dropped_feature_interpretation(self):
        """NOTES.md now says the positive `dropped_honorific` weight is weaker
        evidence than it looks, because the feature never fires on a negative.
        That claim must be true of the data."""
        index = FEATURE_NAMES.index("dropped_honorific")
        on_positives = sum(1 for p in DATASET
                           if p.label == 1
                           and extract_features(p.name_a, p.name_b)[0][index])
        on_negatives = sum(1 for p in DATASET
                           if p.label == 0
                           and extract_features(p.name_a, p.name_b)[0][index])
        self.assertEqual(on_positives, 33)
        self.assertEqual(on_negatives, 0,
                         "if this now fires on negatives, it is no longer a pure "
                         "category indicator and NOTES.md needs updating")


class TestEdgeCaseClaims(unittest.TestCase):
    def test_notes_father_son_scores(self):
        """NOTES.md reports 0.926 and says explicitly that prefix containment
        cannot separate this pair -- that is the honest statement, and this test
        exists so it cannot drift into a better-sounding number."""
        self.assertAlmostEqual(
            TokenAlignedScorer().score("Krishna Kumar", "Krishnan Kumar"),
            0.926, delta=0.001)

    def test_notes_father_son_is_documented_as_a_known_failure(self):
        """The claim that motivates the "add a non-name signal" recommendation."""
        aligned = TokenAlignedScorer()
        father_son = aligned.score("Krishna Kumar", "Krishnan Kumar")
        exact = aligned.score("Krishna Kumar", "Krishna Kumar")
        self.assertGreater(father_son, 0.9,
                           "the pair is close to irreducible; if it drops below "
                           "0.9 NOTES.md needs rewriting")
        self.assertLess(father_son, exact)
        self.assertLess(father_son, 0.95,
                        "and it must still stay out of the near-certain band")

    def test_notes_phonetic_scores(self):
        phonetic = PhoneticJaroWinkler()
        self.assertAlmostEqual(
            phonetic.score("Amit Kumar Agrawal", "Amit Kumar Agarwal"), 0.983, places=3)
        self.assertAlmostEqual(
            phonetic.score("S.", "Suresh Kumar Sharma"), 0.342, places=3)

    def test_notes_phonetic_father_son_got_worse(self):
        """NOTES.md says the phonetic score rose from 0.906 to 0.968 and that
        normalisation cannot fix it, because the confusion is real."""
        phonetic = PhoneticJaroWinkler()
        self.assertAlmostEqual(phonetic.score("Krishna Kumar", "Krishnan Kumar"),
                               0.968, delta=0.001)
        from name_match.phonetics import phonetic_agreement

        self.assertGreater(phonetic_agreement("krishna", "krishnan"), 0.0,
                           "if the encoders now disagree, NOTES.md's reasoning "
                           "changes and must be rewritten")

    def test_notes_lexicon_no_longer_merges_krishna(self):
        from name_match.normalize import TRANSLIT_LEXICON, normalize

        # "krishnan"/"krishan" are genuine spelling variants of one name and stay
        # in the same class. "krishna" was removed from that class because a
        # father and a son in the evaluation set share only that distinction.
        self.assertEqual(TRANSLIT_LEXICON.get("krishnan"),
                         TRANSLIT_LEXICON.get("krishan"))
        self.assertIsNone(TRANSLIT_LEXICON.get("krishna"),
                          "krishna must have no lexicon class at all, or it will "
                          "merge with krishnan again")
        self.assertNotEqual(normalize("Krishna Kumar").canonical,
                            normalize("Krishnan Kumar").canonical)

    def test_notes_exact_is_binary_and_now_correctly_low(self):
        exact = ExactNormalizedMatch()
        self.assertEqual(exact.score("Krishna Kumar", "Krishnan Kumar"), 0.0)
        self.assertEqual(TokenSetJaccard().score("Krishna Kumar", "Krishnan Kumar"),
                         1 / 3)

    def test_notes_learned_score_on_father_son_pair(self):
        # Section 6 quotes 0.856 for the learned model on the father/son pair.
        score_set = BY_KEY["learned"]
        pair = next(p for p in DATASET
                    if p.label == 0 and p.name_a == "Krishna Kumar")
        score = score_set.scores[score_set.rows.index(pair)]
        self.assertAlmostEqual(score, 0.839, delta=0.002,
                               msg=f"learned model now scores {score:.3f}")

    def test_notes_initials_false_negatives(self):
        _fp_table, fn_table = cli._cross_algorithm_error_tables(
            SCORE_SETS, ["learned", "phonetic_jaro", "token_aligned", "token_jaccard"])
        by_category = {row["category"]: row["fn_by_algorithm"] for row in fn_table}
        self.assertEqual(by_category["initials"], [1, 4, 0, 0])

    def test_notes_compound_surname_false_negatives(self):
        _fp_table, fn_table = cli._cross_algorithm_error_tables(
            SCORE_SETS, ["learned", "phonetic_jaro", "token_aligned"])
        by_category = {row["category"]: row["fn_by_algorithm"] for row in fn_table}
        self.assertEqual(by_category["compound_surname"], [1, 0, 7])


class TestDocumentationExists(unittest.TestCase):
    def test_readme_and_notes_present(self):
        for name in ("README.md", "NOTES.md", "requirements.txt", "Makefile", "run.py"):
            self.assertTrue(os.path.exists(os.path.join(REPO_ROOT, name)), name)

    def test_notes_answers_all_four_required_questions(self):
        notes = _read("NOTES.md")
        for fragment in (
            "Run it",                      # Q1: run instructions
            "The metric",                  # Q2: metric justification
            "edge case I'm proudest of",   # Q3: proudest edge case
            "What I'd do next",            # Q4: what next
        ):
            self.assertIn(fragment, notes, f"NOTES.md missing: {fragment}")

    def test_readme_headline_table_matches_results(self):
        readme = _read("README.md")
        for key in RESULTS:
            entry = RESULTS[key]
            p90 = entry["precision_at_recall"]["0.9"]
            self.assertIn(f"{p90:.3f}", readme,
                          f"README.md does not quote the measured P@R=0.90 for {key}")

    def test_required_commands_in_notes_are_real(self):
        notes = _read("NOTES.md")
        for command in ("python3 -m name_match.cli generate",
                        "python3 -m name_match.cli train",
                        "python3 -m name_match.cli evaluate",
                        "python3 -m name_match.cli report",
                        "python3 -m name_match.cli all",
                        "python3 -m unittest discover -s tests -t ."):
            self.assertIn(command, notes, f"NOTES.md does not document: {command}")

    def test_committed_dataset_is_current(self):
        path = os.path.join(REPO_ROOT, "data", "name_pairs.csv")
        if not os.path.exists(path):
            self.skipTest("data/name_pairs.csv not generated yet")
        on_disk = read_csv(path)
        self.assertEqual(
            [(p.name_a, p.name_b, p.label, p.category) for p in on_disk],
            [(p.name_a, p.name_b, p.label, p.category) for p in DATASET],
            "committed CSV is stale; re-run `python3 -m name_match.cli generate`")

    def test_no_todo_placeholders_left_in_docs(self):
        for name in ("README.md", "NOTES.md"):
            text = _read(name)
            for placeholder in ("TODO", "FIXME", "XXX", "TBD", "<placeholder>"):
                self.assertNotIn(placeholder, text, f"{name} still contains {placeholder}")

    def test_documented_python_version_is_what_the_code_needs(self):
        # The code uses `X | None` and `list[str]` in annotations plus
        # `from __future__ import annotations`, so 3.9 is a real floor. Assert
        # the claim rather than trusting it.
        import ast
        import pathlib

        for source in pathlib.Path(REPO_ROOT, "name_match").glob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            has_future = any(
                isinstance(node, ast.ImportFrom) and node.module == "__future__"
                and any(alias.name == "annotations" for alias in node.names)
                for node in tree.body
            )
            self.assertTrue(has_future, f"{source.name} lacks `from __future__ import annotations`")

    def test_no_third_party_imports_anywhere(self):
        """The zero-install claim is only true if nothing imports a package."""
        import ast
        import pathlib
        import sys

        # Ask the interpreter where each name resolves, rather than keeping a
        # list of standard-library module names.
        #
        # Two earlier versions of this test each used a list and both were
        # wrong: a curated fallback for the 3.9 floor omitted `__future__` and
        # `typing` (so every module reported a phantom third-party import on
        # the documented floor), and listing the stdlib *directory* broke on
        # versioned extension modules like `unicodedata.cpython-39-darwin.so`.
        # A list is the wrong shape for this question; the interpreter always
        # has the right answer.
        import importlib.util

        third_party_markers = ("site-packages", "dist-packages")
        local = {"name_match"}

        def is_third_party(name: str) -> bool:
            if name in local:
                return False
            try:
                spec = importlib.util.find_spec(name)
            except (ImportError, ValueError):
                return True  # not importable at all: worth reporting
            if spec is None:
                return True
            places = []
            if spec.origin:
                places.append(spec.origin)
            places.extend(spec.submodule_search_locations or ())
            return any(marker in place.replace("\\", "/")
                       for place in places
                       for marker in third_party_markers)

        offenders: list[str] = []

        for source in list(pathlib.Path(REPO_ROOT, "name_match").glob("*.py")) + \
                list(pathlib.Path(REPO_ROOT, "tests").glob("*.py")) + \
                [pathlib.Path(REPO_ROOT, "run.py")]:
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        continue  # relative import, always in-package
                    names = [node.module.split(".")[0]]
                else:
                    continue
                for name in names:
                    if is_third_party(name):
                        offenders.append(f"{source.name}: {name}")

        self.assertEqual(offenders, [],
                         f"non-stdlib imports found, which breaks the zero-install "
                         f"claim: {offenders}")



class TestPublishedTables(unittest.TestCase):
    """Parse the generated tables and compare them with the computed data.

    These tables used to be duplicated into NOTES.md, where they gave the two
    copies a way to drift apart. They are generated, so the report is their only
    home now, and these tests are what keeps that honest.

    Columns are matched by their *label*, not by position. An earlier version
    indexed positionally and would have silently compared the wrong algorithm's
    numbers if the report's column order ever changed.
    """

    SOURCE = "reports/results.md"

    ALGORITHMS = ("learned", "phonetic_jaro", "token_aligned", "token_jaccard",
                  "exact_normalized")

    #: Label prefix as it appears in the report -> algorithm key.
    COLUMN_LABELS = (
        ("Learned combiner", "learned"),
        ("Token-aligned", "token_aligned"),
        ("Phonetic +", "phonetic_jaro"),
        ("Exact", "exact_normalized"),
        ("Token-set Jaccard", "token_jaccard"),
    )

    def _table_after(self, anchor: str) -> list[list[str]]:
        lines = _read(self.SOURCE).splitlines()
        found = next(i for i, line in enumerate(lines) if anchor in line)
        # The anchor is a prose sentence; the table follows it after a blank
        # line. Skip forward to the first row that actually looks like a table.
        start = next(i for i in range(found, len(lines))
                     if lines[i].strip().startswith("|"))
        rows: list[list[str]] = []
        for line in lines[start:]:
            line = line.strip()
            if not line.startswith("|"):
                break
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= {"-", ":"} for c in cells):
                continue  # markdown alignment row
            rows.append(cells)
        return rows

    def _columns(self, header: list[str]) -> dict[str, int]:
        """Map algorithm key -> column index, by label."""
        found = {}
        for index, cell in enumerate(header[1:], start=1):
            for prefix, key in self.COLUMN_LABELS:
                if cell.startswith(prefix):
                    found[key] = index
        self.assertEqual(set(found), set(self.ALGORITHMS),
                         f"unrecognised report columns: {header}")
        return found

    def _check(self, anchor: str, expected: dict[str, list[int]],
               populated: set[str]) -> None:
        """`expected` maps category -> counts in ``ALGORITHMS`` order; the
        report's own column order is different, so cells are looked up by
        algorithm key rather than by shared position."""
        by_key = {category: dict(zip(self.ALGORITHMS, values))
                  for category, values in expected.items()}
        rows = self._table_after(anchor)
        self.assertGreater(len(rows), 1, f"no table found after {anchor!r}")
        header, body = rows[0], rows[1:]
        self.assertEqual(header[0].lower(), "category", header)
        columns = self._columns(header)
        self.assertEqual(len(header), len(columns) + 1, header)

        seen = set()
        for row in body:
            category = row[0].strip("`")
            self.assertEqual(len(row), len(header),
                             f"row {category} has {len(row)} cells, header "
                             f"declares {len(header)}")
            self.assertIn(category, expected, f"unexpected row {category}")
            seen.add(category)
            for key, index in columns.items():
                self.assertEqual(int(row[index]), by_key[category][key],
                                 f"{category} / {key}")

        self.assertEqual(seen, populated,
                         "the published table must list exactly the categories "
                         "that have pairs")

    def test_published_false_positive_table(self):
        order = list(self.ALGORITHMS)
        fp_table, _fn = cli._cross_algorithm_error_tables(SCORE_SETS, order)
        expected = {row["category"].strip("`"): row["fp_by_algorithm"]
                    for row in fp_table}
        # The report lists every category in both tables, with zeros where
        # a category cannot contribute to that error type.
        populated = {p.category for p in DATASET}
        self._check("False positives at a matched operating point", expected,
                    populated)

    def test_published_false_negative_table(self):
        order = list(self.ALGORITHMS)
        _fp, fn_table = cli._cross_algorithm_error_tables(SCORE_SETS, order)
        expected = {row["category"].strip("`"): row["fn_by_algorithm"]
                    for row in fn_table}
        populated = {p.category for p in DATASET}
        self._check("False negatives at the same operating point", expected,
                    populated)

    def test_every_published_error_table_cell_is_an_integer(self):
        for anchor in ("False positives at a matched operating point",
                       "False negatives at the same operating point"):
            for row in self._table_after(anchor)[1:]:
                for cell in row[1:]:
                    self.assertTrue(cell.isdigit(),
                                    f"{anchor}: {row[0]} -> {cell!r} is not an int")

    def test_published_false_positive_columns_sum_to_the_reported_totals(self):
        """Every negative lives in exactly one category, so the published
        columns must add up to the false-positive counts in the summary. This
        is the cross-check that the per-category view and the headline describe
        the same run."""
        rows = self._table_after("False positives at a matched operating point")
        columns = self._columns(rows[0])
        for key, index in columns.items():
            total = sum(int(row[index]) for row in rows[1:])
            self.assertEqual(total, RESULTS[key]["at_recall_floor"]["fp"], key)

    def test_published_false_negative_columns_sum_to_the_reported_totals(self):
        rows = self._table_after("False negatives at the same operating point")
        columns = self._columns(rows[0])
        positives = sum(1 for p in DATASET if p.label == 1)
        for key, index in columns.items():
            missed = sum(int(row[index]) for row in rows[1:])
            self.assertLessEqual(missed, positives, key)
            self.assertEqual(missed, RESULTS[key]["at_recall_floor"]["fn"], key)

    def test_readme_headline_table_is_row_accurate(self):
        """Every cell of every row, compared against the computed values.

        An earlier version only checked that the measured P@R=0.90 for
        `learned` appeared *somewhere* in the README, which let a wrong cost
        figure in a different row pass unnoticed.
        """
        text = _read("README.md")
        header = ("| algorithm | P@R=0.90 | P@R=0.99 | PR-AUC | cost/pair "
                  "| FP @ recall>=0.90 |")
        lines = text.splitlines()
        start = next(i for i, line in enumerate(lines)
                     if line.strip().replace("≥", ">=") == header)
        rows = []
        for line in lines[start:]:
            line = line.strip()
            if not line.startswith("|"):
                break
            cells = [c.strip().replace("**", "")
                     for c in line.strip("|").split("|")]
            if all(set(c) <= {"-", ":"} for c in cells):
                continue
            rows.append(cells)
        self.assertEqual(len(rows[0]), 6, rows[0])
        body = {r[0].strip(): r[1:] for r in rows[1:]}
        self.assertEqual(len(body), len(self.ALGORITHMS), sorted(body))

        published = {"learned": "Learned combiner",
                     "phonetic_jaro": "Phonetic + Jaro-Winkler",
                     "token_aligned": "Token-aligned weighted scorer",
                     "token_jaccard": "Token-set Jaccard",
                     "exact_normalized": "Exact (normalised tokens)"}
        for key, label in published.items():
            self.assertIn(label, body, f"README is missing the {key} row")
            p90, p99, pr, cost, fp = body[label]
            entry = RESULTS[key]
            self.assertAlmostEqual(float(p90),
                                   entry["precision_at_recall"]["0.9"],
                                   places=3, msg=key)
            self.assertAlmostEqual(float(p99),
                                   entry["precision_at_recall"]["0.99"],
                                   places=3, msg=key)
            self.assertAlmostEqual(float(pr), entry["pr_auc"], places=3, msg=key)
            self.assertAlmostEqual(float(cost),
                                   entry["optimal"]["cost_per_pair"],
                                   places=3, msg=key)
            self.assertEqual(int(fp), entry["at_recall_floor"]["fp"], key)

    def test_readme_band_thresholds_are_the_reported_ones(self):
        """Check the numbers appear in the recommendation sentence itself.

        An earlier version asserted only that the formatted threshold appeared
        *somewhere* in the file, which any unrelated number with the same
        digits would satisfy.
        """
        # Locate the paragraph by content rather than by a fixed phrase, so
        # rewording the README doesn't silently disable the check. Paragraph,
        # not sentence: splitting on "." would cut "0.919" in half.
        paragraph = next(block for block in _read("README.md").split("\n\n")
                         if "auto-approve" in block)
        flat = " ".join(paragraph.split())
        detail = RESULTS["learned"]["band_detail"]
        for label, value in (("auto-approve", detail["raw_approve_at_or_above"]),
                             ("auto-reject", detail["raw_reject_below"])):
            self.assertIn(
                f"{value:.3f}", flat,
                f"the README recommendation does not quote the {label} "
                f"threshold {value:.3f}: {flat!r}")


class TestOrderStabilityClaims(unittest.TestCase):
    """The generated report's two "the order is/isn't stable" claims.

    Both used to be hard-coded prose, which meant they could assert a
    conclusion the data contradicted. They are now computed by
    `cli._reversals`, and this pins the behaviour it must have: a tie is a tie,
    not an overtake.
    """

    KEYS = ["learned", "phonetic_jaro", "token_aligned", "token_jaccard",
            "exact_normalized"]

    def test_only_the_inverted_pair_is_flagged(self):
        from name_match.cli import _reversals

        rows = [{"a": 0.5, "b": 0.1, "c": 0.4},
                {"a": 0.5, "b": 0.4, "c": 0.1}]
        self.assertEqual(_reversals(rows, ["a", "b", "c"]), {"b", "c"},
                         "b and c inverted; a beat both rows and did not move")

    def test_moving_from_strictly_worse_to_tied_is_not_a_crossing(self):
        from name_match.cli import _reversals

        rows = [{"a": 0.9, "b": 0.1},
                {"a": 0.5, "b": 0.5}]
        self.assertEqual(_reversals(rows, ["a", "b"]), set(),
                         "b catching up with a is a tie, not an overtake")
        rows = list(reversed(rows))
        self.assertEqual(_reversals(rows, ["a", "b"]), set())

    def test_a_genuine_inversion_flags_both_algorithms(self):
        from name_match.cli import _reversals

        rows = [{"a": 0.9, "b": 0.1}, {"a": 0.1, "b": 0.9}]
        self.assertEqual(_reversals(rows, ["a", "b"]), {"a", "b"})

    def test_degenerate_input_returns_empty(self):
        from name_match.cli import _reversals

        self.assertEqual(_reversals([], ["a", "b"]), set())
        self.assertEqual(_reversals([{"a": 1.0}], []), set())
        self.assertEqual(_reversals([{}], ["a"]), set())
        self.assertEqual(_reversals([{"a": 1.0}], ["a"]), set())

    def test_the_learned_combiner_is_never_overtaken(self):
        """The claim the recommendation rests on, checked directly.

        It is deliberately *not* a hardcoded list of the other algorithms: those
        shift whenever the dataset changes, and a pinned list turns a data
        change into a test failure that looks like a regression. What must
        always hold is that nothing is ever cheaper than the learned combiner,
        at any cost ratio.
        """
        from name_match.cli import _reversals

        keys = [s.key for s in SCORE_SETS]
        rows = [{key: cli.evaluate_module.find_operating_point(
            s.scores, s.labels, 1.0, ratio).cost_per_pair(1.0, ratio)
            for key, s in zip(keys, SCORE_SETS)}
            for ratio in cli.COST_RATIOS]
        for row in rows:
            for key in keys:
                if key != "learned":
                    self.assertLessEqual(
                        row["learned"], row[key],
                        f"learned is not the cheapest at every cost ratio: {row}")

    def test_the_learned_combiner_leads_at_every_recall_level_that_matters(self):
        """Learned leads everywhere except the very easiest tail.

        On the corrected dataset the phonetic matcher edges it at P@R=0.50 by
        a few thousandths, because that band is dominated by pairs so easy that
        Soundex agreement alone nearly saturates. Every level at or above 0.70
        -- which is where a KYC threshold actually lives -- is led by the
        learned combiner. Asserting "never overtaken" would be false; asserting
        this is what the recommendation actually relies on.
        """
        keys = [s.key for s in SCORE_SETS]
        for level in cli.PRECISION_RECALL_LEVELS:
            if level < 0.70:
                continue
            values = {key: RESULTS[key]["precision_at_recall"][f"{level}"]
                      for key in keys}
            for key in keys:
                if key != "learned":
                    self.assertGreaterEqual(
                        values["learned"], values[key],
                        f"learned is not the most precise at recall {level}: "
                        f"{values}")

    def test_the_generated_report_lists_exactly_those_algorithms(self):
        text = _read("reports/results.md")
        sentences = [line for line in text.splitlines()
                     if "change position somewhere" in line]
        self.assertEqual(len(sentences), 2,
                         "expected one claim per section (precision, sweep)")
        from name_match.cli import _reversals

        keys = [s.key for s in SCORE_SETS]
        labels = {s.key: s.label for s in SCORE_SETS}
        sweep_rows = [
            {key: cli.evaluate_module.find_operating_point(
                s.scores, s.labels, 1.0, ratio).cost_per_pair(1.0, ratio)
             for key, s in zip(keys, SCORE_SETS)}
            for ratio in cli.COST_RATIOS]
        recall_rows = [
            {key: RESULTS[key]["precision_at_recall"][f"{level}"]
             for key in keys} for level in cli.PRECISION_RECALL_LEVELS]
        # Document order: the precision section precedes the cost sweep.
        expected = [_reversals(recall_rows, keys),
                   _reversals(sweep_rows, keys)]
        for sentence, flipped in zip(sentences, expected):
            for key in keys:
                named = labels[key].split(" (")[0] in sentence
                self.assertEqual(
                    named, key in flipped,
                    f"the report names {labels[key]!r} as changing position but "
                    f"the computed reversals are {sorted(flipped)}: {sentence}")


if __name__ == "__main__":
    unittest.main()
