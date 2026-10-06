"""Behavioural tests for the five matchers.

These assert the *shape* of each algorithm's behaviour -- the failure modes it
is supposed to have, not just the scores it happens to produce. A matcher that
silently became more permissive would still pass a score-based test but would
fail these.

Run with ``python3 -m unittest discover -s tests -t .`` from the repo root.
"""

from __future__ import annotations

import unittest

from name_match.algorithms import (ALL_MATCHERS, ExactNormalizedMatch,
                                    PhoneticJaroWinkler, TokenAlignedScorer,
                                    TokenSetJaccard, get_matcher, matcher_names)
from name_match.dataset import build_dataset
from name_match.normalize import normalize

DATASET = build_dataset()
PAIRS = [(p.name_a, p.name_b) for p in DATASET]


class TestContract(unittest.TestCase):
    def test_all_matchers_implement_the_protocol(self):
        for matcher in ALL_MATCHERS:
            self.assertTrue(hasattr(matcher, "key"))
            self.assertTrue(hasattr(matcher, "label"))
            self.assertTrue(callable(matcher.score))
            self.assertTrue(callable(matcher.explain))

    def test_scores_are_bounded(self):
        for matcher in ALL_MATCHERS:
            for name_a, name_b in PAIRS:
                score = matcher.score(name_a, name_b)
                self.assertGreaterEqual(score, 0.0, f"{matcher.key}: {name_a!r}")
                self.assertLessEqual(score, 1.0, f"{matcher.key}: {name_a!r}")

    def test_scores_are_symmetric(self):
        # A match decision must not depend on which document is presented first.
        for matcher in ALL_MATCHERS:
            for name_a, name_b in PAIRS[:60]:
                forward = matcher.score(name_a, name_b)
                backward = matcher.score(name_b, name_a)
                self.assertAlmostEqual(
                    forward, backward, places=9,
                    msg=f"{matcher.key} is asymmetric on {name_a!r}/{name_b!r}")

    def test_scores_are_idempotent(self):
        for matcher in ALL_MATCHERS:
            for name_a, name_b in PAIRS[:40]:
                first = matcher.score(name_a, name_b)
                second = matcher.score(name_a, name_b)
                self.assertEqual(first, second, matcher.key)

    def test_explain_agrees_with_score(self):
        for matcher in ALL_MATCHERS:
            for name_a, name_b in PAIRS[:40]:
                result = matcher.explain(name_a, name_b)
                self.assertAlmostEqual(result.score, matcher.score(name_a, name_b))
                self.assertIsInstance(result.components, dict)

    def test_registry_covers_every_matcher(self):
        self.assertEqual(len(matcher_names()), len(ALL_MATCHERS))
        for key in matcher_names():
            self.assertIs(get_matcher(key).key, key)

    def test_at_least_three_algorithms(self):
        # The brief requires at least three.
        self.assertGreaterEqual(len(ALL_MATCHERS), 3)


class TestEmptyInput(unittest.TestCase):
    """Empty or unextractable names must never be treated as a match."""

    def test_empty_names_do_not_match(self):
        for matcher in ALL_MATCHERS:
            for left, right in (("", ""), ("Suresh", ""), ("", "Suresh"),
                                ("   ", "\t"), ("...", "...")):
                score = matcher.score(left, right)
                self.assertLess(score, 0.5,
                                f"{matcher.key} scored {score} on {left!r}/{right!r}")


class TestNormalisation(unittest.TestCase):
    def test_honorific_removed(self):
        for base in ("Suresh Kumar Sharma", "Lakshmi Devi Rao"):
            self.assertEqual(normalize(base).canonical,
                             normalize(f"Smt. {base}").canonical)
            self.assertEqual(normalize(base).canonical,
                             normalize(f"Shri {base}").canonical)

    def test_relationship_qualifier_removed(self):
        self.assertEqual(
            normalize("Suresh Kumar Sharma S/o Ramesh Chandra").canonical,
            normalize("Suresh Kumar Sharma").canonical)

    def test_suffix_removed(self):
        self.assertEqual(normalize("Vijay Saxena Jr.").canonical,
                         normalize("Vijay Saxena").canonical)

    def test_bare_initial_preserved(self):
        # A single "V" or "I" is a given-name initial far more often than a
        # Roman numeral. Dropping it would destroy "V. Lakshmi Rao".
        self.assertIn("v", normalize("V. Lakshmi Rao").tokens)
        self.assertIn("i", normalize("I. Khan").tokens)

    def test_roman_numeral_still_removed(self):
        self.assertNotIn("ii", normalize("Ramesh Chandra II").tokens)

    def test_leading_kumar_is_always_kept(self):
        # "Kumar" is kept everywhere, including in leading position. It reads
        # like a title in "Kumar Suresh Sharma" but is a given name in
        # "Kumar San Das", and the two are structurally identical. Keeping the
        # token costs one extra token to align; deleting it destroys a name
        # part, so the ambiguity is resolved toward keeping.
        for name in ("Kumar Suresh Sharma", "Kumar San Das", "Kumar Suresh"):
            self.assertIn("kumar", normalize(name).canonical, name)

    def test_medial_kumar_treated_as_name_part(self):
        # "Anil Kumar Sharma": dropping Kumar here would delete a real token.
        self.assertIn("kumar", normalize("Anil Kumar Sharma").canonical)

    def test_diacritics_folded(self):
        self.assertEqual(normalize("Sūresh").canonical, normalize("Suresh").canonical)

    def test_initial_indices_align_after_particle_removal(self):
        name = normalize("Baby S. Suresh")
        self.assertEqual(name.tokens, ("s", "suresh"))
        self.assertEqual(sorted(name.initials), [0])

    def test_empty_input(self):
        self.assertEqual(normalize("").tokens, ())
        self.assertEqual(normalize("   ").tokens, ())
        self.assertTrue(normalize("!!!").is_empty())


class TestExactNormalized(unittest.TestCase):
    def setUp(self):
        self.matcher = ExactNormalizedMatch()

    def test_perfect_after_cleaning(self):
        self.assertEqual(self.matcher.score("Smt. Suresh  Kumar Sharma",
                                            "SURESH KUMAR SHARMA"),
                         1.0)

    def test_order_insensitive(self):
        self.assertEqual(self.matcher.score("Suresh Kumar Sharma",
                                            "Sharma Kumar Suresh"),
                         1.0)

    def test_initials_are_not_a_match(self):
        # The control's whole point: cleaning alone does not solve this.
        self.assertEqual(self.matcher.score("S. Kumar", "Suresh Kumar"), 0.0)

    def test_scores_are_binary(self):
        for name_a, name_b in PAIRS:
            self.assertIn(self.matcher.score(name_a, name_b), (0.0, 1.0))


class TestTokenJaccard(unittest.TestCase):
    def setUp(self):
        self.matcher = TokenSetJaccard()

    def test_structural_failure_on_initials(self):
        # {s, kumar} vs {suresh, kumar} shares 1 of 3 elements. This is the
        # documented weakness of the approach, asserted so it cannot be
        # "fixed" by accident without the report being updated too.
        self.assertAlmostEqual(self.matcher.score("S. Kumar", "Suresh Kumar"), 1 / 3, places=6)

    def test_order_insensitive(self):
        self.assertEqual(self.matcher.score("Suresh Kumar Sharma",
                                            "Sharma Suresh Kumar"),
                         1.0)

    def test_dropped_token_degrades_gracefully(self):
        self.assertAlmostEqual(self.matcher.score("Suresh Kumar Sharma",
                                                     "Suresh Sharma"), 2 / 3, places=6)

    def test_disjoint_scores_zero(self):
        self.assertEqual(self.matcher.score("Kiran Pawar", "Rekha Menon"), 0.0)


class TestPhonetic(unittest.TestCase):
    def setUp(self):
        self.matcher = PhoneticJaroWinkler()

    def test_resists_transliteration(self):
        self.assertGreater(self.matcher.score("Mohammed Iqbal Khan",
                                          "Muhammad Iqbal Khan"), 0.7)
        self.assertGreater(self.matcher.score("Mohammed Iqbal Khan",
                                             "Muhammad Iqbal Khan"),
                           self.matcher.score("Mohammed Iqbal Khan",
                                             "Mohammed"))

    def test_soundex_collision_scores_highly(self):
        # The documented trap, asserted from measured behaviour: this matcher
        # gives a *higher* score to two different people whose surnames differ
        # by one letter than it gives to a genuine match where the given name
        # has been reduced to an initial. No single threshold can serve both.
        collision = self.matcher.score("Amit Kumar Agrawal", "Amit Kumar Agarwal")
        true_match = self.matcher.score("S.", "Suresh Kumar Sharma")
        self.assertGreater(collision, 0.95)
        self.assertLess(true_match, 0.5)
        self.assertGreater(collision, true_match)

    def test_unrelated_names_score_low(self):
        self.assertLess(self.matcher.score("Kiran Pawar", "Rekha Menon"), 0.5)
        self.assertLess(self.matcher.score("Kiran Pawar", "Imran Khan Sheikh"), 0.5)

    def test_order_insensitive(self):
        self.assertAlmostEqual(
            self.matcher.score("Suresh Kumar Sharma",
                               "Sharma Suresh Kumar"),
            self.matcher.score("Sharma Suresh Kumar",
                               "Suresh Kumar Sharma"),
            places=9)


class TestTokenAligned(unittest.TestCase):
    def setUp(self):
        self.matcher = TokenAlignedScorer()

    def test_handles_initials(self):
        self.assertGreater(self.matcher.score("S. Kumar", "Suresh Kumar"), 0.6)
        self.assertGreater(self.matcher.score("S. Kumar", "Suresh Kumar"),
                           TokenSetJaccard().score("S. Kumar", "Suresh Kumar"))

    def test_handles_surname_first(self):
        self.assertGreater(self.matcher.score("Kumar Suresh", "Suresh Kumar"), 0.7)

    def test_handles_relationship_qualifier(self):
        self.assertGreater(
            self.matcher.score("Suresh Kumar Sharma S/o Ramesh Chandra",
                                  "Suresh Kumar Sharma"), 0.6)

    def test_separates_unrelated_people(self):
        self.assertLess(self.matcher.score("Kiran Pawar", "Rekha Menon"), 0.4)
        self.assertLess(self.matcher.score("Rekha Menon", "Ganesh Nadar"), 0.4)

    def test_does_not_manufacture_a_match_from_one_shared_token(self):
        # "Deepa Kumar Pillai" and "Deepa Kumar": two of three tokens match
        # exactly. A scorer that ignores coverage would call this a match.
        self.assertLess(self.matcher.score("Deepa Kumar Pillai", "Deepa Kumar"), 0.85)

    def test_parent_child_prefix_is_not_treated_as_identical(self):
        # "Krishna" is a prefix of "Krishnan" by exactly one character, so this
        # father/son pair is close to irreducible for any name-only matcher. The
        # assertion is the one that is actually true and useful: the pair sits
        # below the near-certain band, and clearly below an exact match.
        father_son = self.matcher.score("Krishna Kumar", "Krishnan Kumar")
        self.assertLess(father_son, 0.95,
                        "a different-person pair must stay out of the top band")
        self.assertLess(father_son,
                        self.matcher.score("Krishna Kumar", "Krishna Kumar"))

    def test_extension_credit_decays_monotonically_with_extension_length(self):
        # The credit a prefix-extended token earns must fall smoothly as the
        # extension grows. Asserting this on the aggregate score would also
        # be asserting claims about the *other* tokens, and token alignment
        # is a thresholded decision, so the aggregate is legitimately allowed
        # to step. The credit function itself has no threshold and must be
        # smooth.
        from name_match.algorithms import _token_similarity

        def credit(short: str, long: str) -> float:
            return _token_similarity(short, short, long, long, False, False)

        for base, longer in (("krishna", "krishnan"),
                             ("venkat", "venkatesh"),
                             ("vijay", "vijayalaxmi")):
            scores = [credit(base[:cut], longer)
                      for cut in range(len(base), 0, -1)]
            for shorter, following in zip(scores, scores[1:]):
                self.assertGreaterEqual(shorter + 1e-12, following,
                                        f"{base}/{longer}: credit rose as the "
                                        f"extension shrank: {scores}")

    def test_extra_tokens_reduce_the_score_monotonically(self):
        # Filler tokens that match nothing must lower the score as they
        # accumulate. A *matching* extra token must raise it instead, so the
        # filler here is deliberately unrelated to both names.
        extras = ["Rao", "Rao X", "Rao XY", "Rao XYZ", "Rao X Y Z"]
        scores = [self.matcher.score("Krishna Kumar", extra) for extra in extras]
        for earlier, later in zip(scores, scores[1:]):
            self.assertGreaterEqual(earlier + 1e-12, later,
                                    f"adding tokens did not lower the score: {scores}")

    def test_a_matching_extra_token_raises_the_score(self):
        # The counterpart to the test above: coverage is not a pure penalty.
        without = self.matcher.score("Krishna Kumar", "Rao")
        with_extra = self.matcher.score("Krishna Kumar", "Krishna Rao")
        self.assertGreater(with_extra, without)

    def test_one_character_extension_is_not_treated_as_a_new_token(self):
        # "Krishna"/"Krishnan" differ by one character; the aligned-token rule
        # must treat that as near-identity rather than as two different names.
        near = self.matcher.score("Krishna Kumar", "Krishnan Kumar")
        unrelated = self.matcher.score("Krishna Kumar", "Venkat Rao")
        self.assertGreater(near, unrelated)
        self.assertGreater(near, 0.9)

    def test_exact_match_is_top(self):
        self.assertAlmostEqual(self.matcher.score("Suresh Kumar Sharma",
                                                      "Suresh Kumar Sharma"), 1.0, places=6)

    def test_explanation_components_present(self):
        result = self.matcher.explain("S. Kumar", "Suresh Kumar")
        for key in ("coverage", "agreement", "order_bonus", "char_signal"):
            self.assertIn(key, result.components)


class TestLearnedCombiner(unittest.TestCase):
    def test_untrained_model_is_neutral_not_optimistic(self):
        from name_match.algorithms import LearnedCombiner

        matcher = LearnedCombiner()
        # An unfitted model must not return 1.0, which would make every pair
        # look like a match before any training has happened.
        self.assertAlmostEqual(matcher.score("Suresh Kumar Sharma",
                                             "Rekha Menon"), 0.5)

    def test_beats_chance_on_the_dataset(self):
        from name_match.algorithms import LearnedCombiner
        from name_match.features import FEATURE_NAMES, extract_features
        from name_match.model import LogisticRegression

        matrix = [extract_features(p.name_a, p.name_b)[0] for p in DATASET]
        labels = [p.label for p in DATASET]
        model = LogisticRegression().fit(matrix, labels, FEATURE_NAMES)
        trained = LearnedCombiner(model=model)

        positives = [p for p in DATASET if p.label == 1]
        negatives = [p for p in DATASET if p.label == 0]
        mean_positive = sum(trained.score(p.name_a, p.name_b) for p in positives) / len(positives)
        mean_negative = sum(trained.score(p.name_a, p.name_b) for p in negatives) / len(negatives)
        self.assertGreater(mean_positive, mean_negative + 0.2,
                           "trained model does not separate the classes")

    def test_saved_model_round_trips(self):
        import os
        import tempfile

        from name_match.algorithms import LearnedCombiner
        from name_match.features import FEATURE_NAMES, extract_features
        from name_match.model import LogisticRegression

        matrix = [extract_features(p.name_a, p.name_b)[0] for p in DATASET]
        labels = [p.label for p in DATASET]
        model = LogisticRegression().fit(matrix, labels, FEATURE_NAMES)

        with tempfile.TemporaryDirectory() as directory:
            path = model.save(os.path.join(directory, "model.json"))
            loaded = LogisticRegression.load(path)

        self.assertEqual(loaded.weights, model.weights)
        self.assertEqual(loaded.feature_names, model.feature_names)
        restored = LearnedCombiner(model=loaded)
        for name_a, name_b in PAIRS[:20]:
            self.assertAlmostEqual(restored.score(name_a, name_b),
                                   LearnedCombiner(model=model).score(name_a, name_b))


if __name__ == "__main__":
    unittest.main()
