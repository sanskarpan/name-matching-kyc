"""Unit tests for the string and phonetic primitives.

Run with ``python3 -m unittest discover -s tests -v`` from the repo root.
"""

from __future__ import annotations

import unittest

from name_match import phonetics, string_algos
from name_match.model import sigmoid, standardize


class TestLevenshtein(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(string_algos.levenshtein("suresh", "suresh"), 0)

    def test_empty(self):
        self.assertEqual(string_algos.levenshtein("", ""), 0)
        self.assertEqual(string_algos.levenshtein("", "abc"), 3)
        self.assertEqual(string_algos.levenshtein("abc", ""), 3)

    def test_substitution(self):
        self.assertEqual(string_algos.levenshtein("kitten", "sitting"), 3)

    def test_known_pair(self):
        # Appending " kumar" is six insertions.
        self.assertEqual(string_algos.levenshtein("suresh", "suresh kumar"), 6)

    def test_symmetry(self):
        for a, b in (("suresh", "sures"), ("kumar", "kumar sharma"), ("a", "abcd")):
            self.assertEqual(string_algos.levenshtein(a, b), string_algos.levenshtein(b, a))

    def test_ratio_bounds(self):
        for a, b in (("suresh", "suresh"), ("", "abc"), ("abc", "xyz"), ("a", "a")):
            ratio = string_algos.levenshtein_ratio(a, b)
            self.assertGreaterEqual(ratio, 0.0)
            self.assertLessEqual(ratio, 1.0)

    def test_two_empty_strings_match(self):
        self.assertEqual(string_algos.levenshtein_ratio("", ""), 1.0)


class TestDamerau(unittest.TestCase):
    def test_transposition_costs_one(self):
        # The whole reason OSA is used here: a swapped pair is one edit, not two.
        self.assertEqual(string_algos.damerau_osa("suersh", "suresh"), 1)

    def test_transposition_beats_levenshtein(self):
        self.assertLess(string_algos.damerau_osa("suersh", "suresh"),
                        string_algos.levenshtein("suersh", "suresh"))

    def test_identical(self):
        self.assertEqual(string_algos.damerau_osa("kumar", "kumar"), 0)

    def test_no_worse_than_levenshtein(self):
        for a, b in (("mohammed", "mohamed"), ("sarma", "sarmaa"), ("ab", "ba")):
            self.assertLessEqual(string_algos.damerau_osa(a, b),
                                 string_algos.levenshtein(a, b) + 1)


class TestJaro(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(string_algos.jaro("suresh", "suresh"), 1.0)

    def test_disjoint(self):
        self.assertEqual(string_algos.jaro("abc", "xyz"), 0.0)

    def test_empty(self):
        self.assertEqual(string_algos.jaro("", "abc"), 0.0)

    def test_order_insensitive_enough(self):
        # A transposition should still score high: this is the property that
        # makes character-level comparison viable for reordered names.
        self.assertGreater(string_algos.jaro("suresh", "suresh"), 0.8)

    def test_bounded(self):
        for a, b in (("suresh", "sunil"), ("kumar", "k"), ("", "x")):
            value = string_algos.jaro(a, b)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


class TestJaroWinkler(unittest.TestCase):
    def test_shared_prefix_scores_above_jaro(self):
        base = string_algos.jaro("suresh", "sures")
        boosted = string_algos.jaro_winkler("suresh", "sures")
        self.assertGreaterEqual(boosted, base)

    def test_prefix_boost_capped_at_one(self):
        for a, b in (("suresh", "suresh"), ("kumar", "kumar sharma")):
            self.assertLessEqual(string_algos.jaro_winkler(a, b), 1.0)

    def test_boost_not_applied_below_floor(self):
        # Standard behaviour: the prefix bonus only applies above 0.7 Jaro.
        self.assertEqual(string_algos.jaro_winkler("abcdef", "zzzzzz"),
                         string_algos.jaro("abcdef", "zzzzzz"))


class TestNgrams(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(string_algos.ngram_jaccard("suresh", "suresh"), 1.0)
        self.assertEqual(string_algos.ngram_dice("suresh", "suresh"), 1.0)

    def test_disjoint(self):
        self.assertEqual(string_algos.ngram_jaccard("aaaa", "bbbb"), 0.0)

    def test_dice_more_forgiving_than_jaccard(self):
        # Dropping a token from a three-token name costs Jaccard much more than
        # Dice. This is the reason Dice is a feature and Jaccard alone is not.
        a, b = "suresh kumar sharma", "suresh sharma"
        self.assertGreater(string_algos.ngram_dice(a, b),
                           string_algos.ngram_jaccard(a, b))

    def test_containment(self):
        # "kumar" is a 5-character substring of a 17-character name.
        self.assertAlmostEqual(
            string_algos.longest_common_substring_ratio("kumar", "anil kumar sharma"),
            5 / 17, places=6)

    def test_bounded(self):
        for a, b in (("a", "b"), ("", "abc"), ("same", "same")):
            for fn in (string_algos.ngram_jaccard, string_algos.ngram_dice):
                self.assertGreaterEqual(fn(a, b), 0.0)
                self.assertLessEqual(fn(a, b), 1.0)


class TestPrefix(unittest.TestCase):
    def test_prefix_detection(self):
        self.assertTrue(string_algos.is_prefix_of("s", "suresh"))
        self.assertTrue(string_algos.is_prefix_of("suresh", "s"))
        self.assertFalse(string_algos.is_prefix_of("x", "suresh"))
        self.assertFalse(string_algos.is_prefix_of("", "suresh"))

    def test_prefix_ratio(self):
        # The denominator is the shorter string, so a strict prefix is 1.0.
        self.assertAlmostEqual(string_algos.prefix_ratio("sur", "suresh"), 1.0)
        self.assertAlmostEqual(string_algos.prefix_ratio("suresh", "suresh"), 1.0)
        # Partial: the shared prefix runs out before the shorter string ends.
        # "sur" in common, 3 of the shorter string's 5 characters.
        self.assertAlmostEqual(string_algos.prefix_ratio("sursh", "suresh"), 0.6)
        # No shared prefix at all.
        self.assertAlmostEqual(string_algos.prefix_ratio("xyz", "suresh"), 0.0)
        # A long token containing a short one is *not* penalised for length.
        self.assertAlmostEqual(string_algos.prefix_ratio("sur", "sureshkumar"), 1.0)


class TestSoundex(unittest.TestCase):
    def test_length_and_shape(self):
        for name in ("suresh", "kumar", "mohammed", "sharma"):
            code = phonetics.soundex(name)
            self.assertEqual(len(code), 4, f"{name} -> {code}")
            self.assertTrue(code[0].isalpha(), f"{name} -> {code}")
            self.assertTrue(code[1:].isdigit(), f"{name} -> {code}")

    def test_same_first_letter_same_code(self):
        self.assertEqual(phonetics.soundex("sharma"), phonetics.soundex("sarma"))

    def test_known_different_families(self):
        self.assertNotEqual(phonetics.soundex("kumar"), phonetics.soundex("suresh"))

    def test_transliteration_collision(self):
        # This collision is the dataset's central trap, asserted here so that
        # changing the Soundex implementation cannot silently remove it.
        self.assertEqual(phonetics.soundex("mohammed"),
                         phonetics.soundex("mohammad"))

    def test_empty(self):
        self.assertEqual(phonetics.soundex(""), "")


class TestMetaphone(unittest.TestCase):
    def test_non_empty_for_real_names(self):
        for name in ("suresh", "kumar", "mohammed", "lakshmi"):
            self.assertTrue(phonetics.metaphone(name), name)

    def test_transliteration_agreement(self):
        self.assertEqual(phonetics.metaphone("mohammed"),
                         phonetics.metaphone("mohammad"))

    def test_pair_is_sorted_tuple(self):
        pair = phonetics.metaphone_pair("suresh")
        self.assertIsInstance(pair, tuple)
        self.assertEqual(list(pair), sorted(pair))

    def test_empty(self):
        self.assertEqual(phonetics.metaphone(""), "")


class TestPhoneticAgreement(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(phonetics.phonetic_agreement("kumar", "kumar"), 1.0)

    def test_disjoint(self):
        self.assertEqual(phonetics.phonetic_agreement("kumar", "xyzzy"), 0.0)

    def test_empty_input(self):
        self.assertEqual(phonetics.phonetic_agreement("", "kumar"), 0.0)

    def test_soundex_collision_is_weaker_than_real_agreement(self):
        # The whole point of the discount: a phonetic collision must never be
        # able to manufacture a match on its own.
        collision = phonetics.phonetic_agreement("sharma", "saxena")
        real = phonetics.phonetic_agreement("mohammed", "mohammad")
        self.assertLess(collision, real)


class TestModelPrimitives(unittest.TestCase):
    def test_sigmoid_bounds(self):
        self.assertAlmostEqual(sigmoid(0.0), 0.5)
        self.assertGreater(sigmoid(10.0), 0.99)
        self.assertLess(sigmoid(-10.0), 0.01)
        # Must not raise OverflowError.
        self.assertGreaterEqual(sigmoid(-1000.0), 0.0)
        self.assertLessEqual(sigmoid(1000.0), 1.0)

    def test_standardize_recovers_unit_variance(self):
        matrix = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]
        scaled, means, stds = standardize(matrix)
        for j in range(2):
            self.assertAlmostEqual(means[j], sum(r[j] for r in matrix) / 4)
            self.assertGreater(stds[j], 0.0)
        for row in scaled:
            self.assertEqual(len(row), 2)

    def test_standardize_handles_constant_column(self):
        # A zero-variance feature must not divide by zero.
        scaled, _means, stds = standardize([[1.0, 5.0], [1.0, 7.0]])
        self.assertEqual(stds[0], 1.0)
        for row in scaled:
            self.assertEqual(row[0], 0.0)


if __name__ == "__main__":
    unittest.main()
