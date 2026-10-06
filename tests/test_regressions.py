"""Regression tests for every bug found in the five-round audit.

Each test here corresponds to a defect that was live in this codebase and that
produced *plausible-looking output* rather than an error. They are collected in
one file, separated from the behavioural tests, so the reason each assertion
exists stays legible: most of these bugs were invisible from the outside and
were only caught by writing the assertion against a hand-computable case first.

Every entry is named `test_bug_<n>_<short description>` so it can be traced back
to the audit finding. If one of these ever fails, the specific defect it guards
has returned.

Run with ``python3 -m unittest discover -s tests -t .`` from the repo root.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import tempfile
import unittest

from name_match import phonetics, string_algos
from name_match.algorithms import (ALL_MATCHERS, BASELINE_MATCHERS,
                                    LearnedCombiner, PhoneticJaroWinkler,
                                    TokenAlignedScorer, _token_similarity)
from name_match.dataset import build_dataset
from name_match.evaluate import (_threshold_grid, cost_at,
                                  interpolated_precision_at_recall, pr_auc,
                                  roc_auc, zero_fp_point)
from name_match.features import FEATURE_NAMES, extract_features
from name_match.model import LogisticRegression, sigmoid
from name_match.normalize import TRANSLIT_LEXICON, normalize


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


class TestBugMetrics(unittest.TestCase):
    def test_bug_01_pr_auc_reported_a_non_monotone_value(self):
        """`interpolated_precision_at_recall` linearly interpolated between
        achievable points, producing precision values no threshold attained and
        precision that *rose* with recall -- impossible, since demanding more
        recall can only admit more items and add false positives.

        It now returns the standard envelope ``P(R) = max{p(r) : r >= R}``.
        """
        # A ranking whose achievable points are recall 0.43 @ precision 0.85
        # and recall 1.00 @ precision 0.47. Linear interpolation invented 0.80
        # at recall 0.50; the true best at recall >= 0.50 is 0.47.
        scores = [1.0] * 5 + [0.0] * 10
        labels = [1, 1, 1, 0, 0] + [0] * 10
        reported = interpolated_precision_at_recall(scores, labels, 0.5)
        achievable = [
            c.precision
            for c in (cost_at(scores, labels, t) for t in _threshold_grid(scores))
            if c.recall + 1e-9 >= 0.5 and c.predicted > 0
        ]
        self.assertAlmostEqual(reported, max(achievable), places=9)

    def test_bug_02_precision_at_recall_is_monotone(self):
        rng = random.Random(11)
        for _ in range(200):
            n = rng.randint(4, 40)
            scores = [round(rng.random(), 3) for _ in range(n)]
            labels = [rng.randint(0, 1) for _ in range(n)]
            if sum(labels) == 0:
                continue
            values = [interpolated_precision_at_recall(scores, labels, r)
                      for r in (0.1, 0.3, 0.5, 0.7, 0.9)]
            for earlier, later in zip(values, values[1:]):
                self.assertGreaterEqual(earlier + 1e-12, later)

    def test_bug_03_zero_fp_fallback_used_the_wrong_denominator(self):
        """The no-zero-FP-threshold fallback built `tn=0`, so `cost_per_pair`
        divided by the positive count instead of the pair count and inflated
        the reported cost by the negative fraction. It also returned
        `max(scores)`, which is usually NOT zero-FP.
        """
        scores = [0.5, 0.5]
        labels = [1, 0]
        point = zero_fp_point(scores, labels)
        self.assertEqual(point.tp + point.fp + point.tn + point.fn, 2)
        self.assertAlmostEqual(point.cost_per_pair(cost_fp=1.0, cost_fn=25.0), 12.5)
        # The published threshold must actually be free of false positives.
        self.assertGreater(point.threshold, max(scores))
        self.assertEqual(cost_at(scores, labels, point.threshold).fp, 0)

    def test_bug_04_zero_fp_fallback_reported_perfect_precision(self):
        """`Confusion.precision` returned 1.0 when nothing was predicted, so the
        degenerate reject-everything point was published as perfect precision."""
        confusion = cost_at([0.5, 0.5], [1, 0], 1.5)
        self.assertNotEqual(confusion.precision, confusion.precision)  # NaN
        self.assertEqual(confusion.predicted, 0)

    def test_bug_05_bootstrap_returned_the_wrong_key_on_empty_input(self):
        """The empty-input path returned `p_b_better` while both consumers read
        `p_a_cheaper` -- a KeyError on the empty path."""
        from name_match.evaluate import bootstrap_cost_delta

        result = bootstrap_cost_delta([], [], [], [], [])
        self.assertIn("p_a_cheaper", result)

    def test_bug_06_bootstrap_did_not_validate_pairing(self):
        from name_match.evaluate import bootstrap_cost_delta

        with self.assertRaises(ValueError):
            bootstrap_cost_delta([0.9, 0.8], [1, 1, 0], [0.9], 0.5, 0.5)
        with self.assertRaises(ValueError):
            bootstrap_cost_delta([0.9], [1, 1], [0.9], 0.5, 0.5)

    def test_bug_07_pr_auc_hung_on_a_nan_score(self):
        """`nan == nan` is False, so the tie-grouping loop could not advance
        past a NaN and spun forever."""
        for fn in (pr_auc, roc_auc):
            with self.assertRaises(ValueError):
                fn([float("nan"), 0.5], [1, 0])
        with self.assertRaises(ValueError):
            cost_at([float("nan")], [1], 0.5)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class TestBugModel(unittest.TestCase):
    def test_bug_08_log_loss_clamped_the_logit_instead_of_the_probability(self):
        """`log_loss` skipped the sigmoid, so its clamp operated on a logit
        rather than a probability: every logit below 0 collapsed to p=1e-12
        (loss 27.6) and every logit above 1 to p=1-1e-12 (loss ~0). Every
        training loss reported anywhere in this project was noise.
        """
        model = LogisticRegression()
        model.weights = [2.0]
        model.bias = 0.0
        model.means = [0.0]
        model.stds = [1.0]

        expected = -(1 * math.log(sigmoid(2.0)) + 0 * math.log(1 - sigmoid(2.0)))
        self.assertAlmostEqual(model.log_loss([[1.0]], [1]), expected, places=12)
        # The bug reported ~1e-12 here.
        self.assertGreater(model.log_loss([[1.0]], [1]), 0.1)

    def test_bug_09_log_loss_matches_a_hand_computed_value(self):
        model = LogisticRegression()
        model.weights = [0.0]
        model.bias = 0.0
        model.means = [0.0]
        model.stds = [1.0]
        # p = 0.5 for every row; BCE for a balanced set is exactly log 2.
        self.assertAlmostEqual(
            model.log_loss([[0.0], [0.0]], [1, 0]), math.log(2), places=12)

    def test_bug_10_fit_accepted_a_ragged_matrix(self):
        with self.assertRaises(ValueError):
            LogisticRegression().fit([[1.0, 2.0], [1.0]], [1, 0])

    def test_bug_11_fit_accepted_non_binary_labels(self):
        with self.assertRaises(ValueError):
            LogisticRegression().fit([[1.0]], [7])

    def test_bug_12_fit_rejects_a_feature_name_count_mismatch(self):
        """`coefficient_report` zipped names against weights, so a name-count
        mismatch silently truncated the published coefficients table."""
        with self.assertRaises(ValueError):
            LogisticRegression().fit([[1.0, 2.0]], [1], ["only_one"])

    def test_bug_13_predict_proba_accepted_a_stale_feature_vector(self):
        """Without a length check, a 3-weight artefact fed a 24-long feature
        vector used only the first three features and returned a confident
        number computed from almost nothing."""
        model = LogisticRegression()
        model.weights = [1.0, 1.0, 1.0]
        model.means = [0.0, 0.0, 0.0]
        model.stds = [1.0, 1.0, 1.0]
        with self.assertRaises(ValueError):
            model.predict_proba([0.1] * 24)

    def test_bug_14_load_did_not_check_the_feature_order(self):
        """A model saved before a feature was renamed loaded successfully and
        then scored every pair with the wrong weights in the wrong order."""
        from name_match.features import FEATURE_NAMES

        with tempfile.TemporaryDirectory() as directory:
            path = LogisticRegression().fit(
                [[1.0] * len(FEATURE_NAMES)], [1], FEATURE_NAMES
            ).save(os.path.join(directory, "m.json"))
            LogisticRegression.load(path, expected_features=FEATURE_NAMES)
            with self.assertRaises(ValueError):
                LogisticRegression.load(path, expected_features=("wrong", "names"))

    def test_bug_15_save_emitted_invalid_json(self):
        """Python's json emits bare `NaN`, which is not valid JSON. An unfitted
        model's loss is NaN, so its artefact failed a strict parser."""
        with tempfile.TemporaryDirectory() as directory:
            path = LogisticRegression().save(os.path.join(directory, "u.json"))

            def reject(constant):
                raise ValueError(f"bare {constant} token is not valid JSON")

            with open(path, encoding="utf-8") as handle:
                json.load(handle, parse_constant=reject)


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


class TestBugFeatures(unittest.TestCase):
    def test_bug_16_initial_features_made_the_learned_matcher_asymmetric(self):
        """`has_initial_a` / `has_initial_b` put the same fact in two
        order-dependent slots. The dataset only ever abbreviates the name in
        slot `a`, so `has_initial_b` was a constant-zero column pinned at
        weight 0.0 while `has_initial_a` learned "+0.92"; swapping the
        arguments then swung the probability by up to 0.64 and flipped 21
        match decisions.
        """
        self.assertNotIn("has_initial_a", FEATURE_NAMES)
        self.assertNotIn("has_initial_b", FEATURE_NAMES)
        self.assertIn("has_initial_either", FEATURE_NAMES)

        index = FEATURE_NAMES.index("has_initial_either")
        for a, b in (("S. Kumar Sharma", "Suresh Kumar Sharma"),
                     ("Suresh Kumar Sharma", "S. Kumar Sharma"),
                     ("K.", "Krishnan Kumar"),
                     ("Krishnan Kumar", "K.")):
            left = extract_features(a, b)[0]
            right = extract_features(b, a)[0]
            self.assertEqual(left[index], right[index], f"{a!r}/{b!r}")
            self.assertEqual(left, right, f"feature vector is order-dependent on {a!r}/{b!r}")

    def test_bug_17_every_matcher_is_symmetric_on_the_dataset(self):
        for matcher in ALL_MATCHERS:
            if matcher.key == "learned":
                continue  # untrained here; covered by test_bug_16
            for pair in build_dataset():
                self.assertAlmostEqual(
                    matcher.score(pair.name_a, pair.name_b),
                    matcher.score(pair.name_b, pair.name_a), places=12,
                    msg=f"{matcher.key} asymmetric on {pair.name_a!r}/{pair.name_b!r}")

    def test_bug_18_no_feature_is_constant_on_the_dataset(self):
        """A constant column gets `std=1.0` from the zero-variance guard, which
        silently pins its weight at exactly 0.0 -- a feature that looks present
        in the artefact and does nothing."""
        dataset = build_dataset()
        for index, name in enumerate(FEATURE_NAMES):
            values = {extract_features(p.name_a, p.name_b)[0][index]
                      for p in dataset}
            self.assertGreater(len(values), 1,
                               f"feature {name!r} is constant on the dataset")

    def test_bug_19_token_dice_mixed_sets_with_lists(self):
        """`_token_dice` took a set intersection but divided by list lengths,
        so a repeated token deflated the coefficient and made it disagree with
        the token-set Jaccard on the line above it."""
        index = FEATURE_NAMES.index("token_dice")
        repeated = extract_features("Anil Anil Kumar", "Anil Kumar")[0][index]
        self.assertAlmostEqual(repeated, 1.0, places=9)

    def test_bug_20_empty_alignment_invented_a_perfect_order_bonus(self):
        """When no tokens aligned, `TokenAlignedScorer.explain` returned a
        two-key components dict. `features.py` read it with `.get(..., 1.0)`
        for `order_bonus`, injecting the strongest possible "these names agree
        perfectly on order" for a pair sharing no tokens, and discarded the
        real 0.37 character similarity.
        """
        result = TokenAlignedScorer().explain("Manjula Naidu", "Joseph Thomas")
        self.assertIn("order_bonus", result.components)
        self.assertEqual(result.components["order_bonus"], 0.0,
                         "a pair with no aligned tokens cannot agree on order")
        self.assertGreater(result.components["char_signal"], 0.0,
                           "the real character similarity must not be discarded")

        vector = extract_features("Manjula Naidu", "Joseph Thomas")[0]
        self.assertEqual(vector[FEATURE_NAMES.index("order_bonus")], 0.0)
        self.assertAlmostEqual(
            vector[FEATURE_NAMES.index("char_signal")],
            string_algos.jaro_winkler(
                " ".join(sorted(normalize("Manjula Naidu").canonical)),
                " ".join(sorted(normalize("Joseph Thomas").canonical))),
            places=6)

    def test_bug_21_feature_vector_matches_its_names(self):
        dataset = build_dataset()
        for pair in dataset[:80]:
            vector = extract_features(pair.name_a, pair.name_b)[0]
            self.assertEqual(len(vector), len(FEATURE_NAMES))

    def test_bug_22_features_are_finite(self):
        for pair in build_dataset():
            for index, value in enumerate(extract_features(pair.name_a, pair.name_b)[0]):
                self.assertTrue(math.isfinite(value),
                                f"{FEATURE_NAMES[index]} is not finite: {value}")


# --------------------------------------------------------------------------
# String and phonetic primitives
# --------------------------------------------------------------------------


class TestBugPrimitives(unittest.TestCase):
    def test_bug_23_jaro_counted_every_match_as_a_transposition(self):
        """`transpositions = matches // 2` treated every matched character as
        half a transposition, penalising pairs that share characters in the
        right order. "martha"/"marhta" scored 0.833 instead of the canonical
        0.944, and jaro_winkler inherited the error into two of five matchers.
        """
        self.assertAlmostEqual(string_algos.jaro("martha", "marhta"),
                               0.944444, places=6)

        def reference(s1, s2):
            """Wikipedia pseudo-code, transcribed independently."""
            if s1 == s2:
                return 1.0
            len1, len2 = len(s1), len(s2)
            if not len1 or not len2:
                return 0.0
            window = max(max(len1, len2) // 2 - 1, 0)
            m1 = [False] * len1
            m2 = [False] * len2
            matches = 0
            for i in range(len1):
                for j in range(max(0, i - window), min(i + window + 1, len2)):
                    if m2[j] or s1[i] != s2[j]:
                        continue
                    m1[i] = m2[j] = True
                    matches += 1
                    break
            if not matches:
                return 0.0
            a = [s1[i] for i in range(len1) if m1[i]]
            b = [s2[j] for j in range(len2) if m2[j]]
            transpositions = sum(1 for x, y in zip(a, b) if x != y) / 2.0
            return (matches / len1 + matches / len2
                    + (matches - transpositions) / matches) / 3.0

        rng = random.Random(3)
        for _ in range(4000):
            a = "".join(rng.choice("abcd") for _ in range(rng.randint(0, 7)))
            b = "".join(rng.choice("abcd") for _ in range(rng.randint(0, 7)))
            self.assertAlmostEqual(string_algos.jaro(a, b), reference(a, b), places=12)

    def test_bug_24_jaro_winkler_could_exceed_one(self):
        """Nothing clamped the prefix boost, so any `prefix_weight` above
        ~0.3 returned a similarity above 1.0 -- which then silently poisoned
        every threshold downstream."""
        for weight in (0.3, 0.5, 0.9, 2.0):
            for a, b in (("marhta", "marhtb"), ("abcdX", "abcdY"), ("ab", "ab")):
                self.assertLessEqual(
                    string_algos.jaro_winkler(a, b, weight, 4), 1.0,
                    f"prefix_weight={weight} on {a!r}/{b!r}")

    def test_bug_25_metaphone_initial_exceptions_were_dead_code(self):
        """The initial-letter table was keyed lower-case and matched against an
        upper-cased word, so the silent initial letters in KNIFE, GNOME,
        PNEUMATIC, WRATH, WHILE and XY were all retained."""
        for token, expected in (("knife", "NIFE"), ("gnome", "NOME"),
                                ("wrath", "RET"), ("xylophone", "SYLOFONE")):
            self.assertEqual(phonetics.metaphone(token), expected, token)

    def test_bug_26_metaphone_truncated_multi_character_codes(self):
        """`out.append(replacement[0])` collapsed "SK"->"S", "KS"->"K",
        "NG"->"N" and "KW"->"K", so the encoder lost half its discriminative
        power."""
        self.assertEqual(phonetics.metaphone("school"), "SKUL")
        self.assertEqual(phonetics.metaphone("tax"), "TEKS")
        self.assertEqual(phonetics.metaphone("singh"), "SING")

    def test_bug_27_metaphone_vowel_softening_was_unreachable(self):
        """`_VOWELS` was lower-case while the output buffer was upper-case, so
        `out[-1] in _VOWELS` was never true and the documented "silent W/Y after
        a vowel" behaviour did not exist. The branch also popped the vowel
        rather than the W/Y."""
        self.assertIn("E", phonetics.metaphone("wager"))
        self.assertEqual(phonetics.metaphone("wager"), "WEKER")

    def test_bug_28_digit_tokens_scored_a_phantom_phonetic_match(self):
        """Both encoders return "" for a digit-only token and
        `metaphone_pair` returns a one-element tuple containing it, so any two
        different digit strings shared an "empty code" and scored 0.8."""
        self.assertEqual(phonetics.phonetic_agreement("123", "456"), 0.0)
        self.assertEqual(phonetics.phonetic_agreement("12", "34"), 0.0)
        self.assertEqual(phonetics.phonetic_agreement("1a", "2b"), 0.0)

    def test_bug_29_soundex_broke_its_four_character_contract(self):
        """`.upper()` is not length preserving ('ss'->'SS', 'ffl'->'FFL'), so the
        result could exceed four characters."""
        for token in ("ss", "ffl", "ß", "ﬄ", "suresh"):
            code = phonetics.soundex(token)
            self.assertLessEqual(len(code), 4, f"{token!r} -> {code!r}")

    def test_bug_30_vasudevan_wasudevan_never_collided(self):
        """`_v_f_variants` only swapped V and F, so the docstring's headline
        `Vasudevan`/`Wasudevan` pair could never produce a match -- the
        documented capability did not exist."""
        self.assertTrue(set(phonetics.metaphone_pair("vasudevan"))
                        & set(phonetics.metaphone_pair("wasudevan")))
        self.assertGreater(phonetics.phonetic_agreement("vasudevan", "wasudevan"), 0.0)


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


class TestBugNormalisation(unittest.TestCase):
    def test_bug_31_lal_was_deleted_as_if_it_were_a_title(self):
        """`lal` was listed as an ambiguous particle and stripped when leading,
        so "Lal Bahadur Shastri" lost the applicant's given name -- the
        unrecoverable deletion the function exists to prevent."""
        self.assertEqual(normalize("Lal Bahadur Shastri").tokens,
                         ("lal", "bahadur", "shastri"))
        self.assertNotIn("lal", __import__("name_match.normalize",
                                           fromlist=["x"]).AMBIGUOUS_PARTICLES)

    def test_bug_32_missing_relationship_markers_kept_a_stranger(self):
        """Only S/o D/o W/o W/d B/o C/o M/o F/o were protected, so H/o, S/d, M/d,
        F/d and B/d were missed and the *related person's* name was kept as the
        applicant's -- the opposite failure from truncation."""
        for name, expected in (("Lakshmi H/o Sita", ("lakshmi",)),
                               ("Lakshmi S/d Ram", ("lakshmi",)),
                               ("Lakshmi M/d Ram", ("lakshmi",)),
                               ("Lakshmi F/d Ram", ("lakshmi",)),
                               ("Lakshmi B/d Ram", ("lakshmi",))):
            self.assertEqual(normalize(name).tokens, expected, name)

    def test_bug_33_bare_two_letter_qualifiers_truncated_real_names(self):
        """`RELATIONSHIP_QUALIFIERS` held the slash-stripped spellings ("so",
        "do", "mo", "co", "bo", "wd") as standalone tokens, so any real name
        with such a token lost everything after it."""
        self.assertEqual(normalize("Ram So Das").tokens, ("ram", "so", "das"))
        self.assertEqual(normalize("Ravi Do").tokens, ("ravi", "do"))
        self.assertEqual(normalize("Raj Co Deep").tokens, ("raj", "co", "deep"))

    def test_bug_34_noise_words_ate_single_letter_initials(self):
        """`a` and `an` were in the noise set, so "A. Kumar" lost its initial and
        the initial-detection path could never see a whole class of names."""
        self.assertEqual(normalize("A. Kumar").tokens, ("a", "kumar"))
        self.assertEqual(normalize("A. Kumar").initials, frozenset({0}))
        self.assertEqual(normalize("An. Kumar").tokens, ("an", "kumar"))

    def test_bug_35_lexicon_split_five_classes_it_was_meant_to_merge(self):
        """Nine tokens belonged to two transliteration classes at once and
        "first writer wins" silently split them, so genuine spelling pairs did
        not unify."""
        for a, b in (("Chatterji", "Chatterjee"),
                     ("Srinivasan", "Sreenivasan"),
                     ("Dey", "De"),
                     ("Mohammed", "Mohammad"),
                     ("Sharma", "Sarma")):
            self.assertEqual(normalize(a).canonical, normalize(b).canonical,
                             f"{a}/{b} did not unify")

    def test_bug_36_lexicon_token_belongs_to_exactly_one_class(self):
        from name_match.normalize import _TRANSLIT_CLASSES

        seen: dict[str, str] = {}
        for canonical, members in _TRANSLIT_CLASSES:
            for member in members:
                self.assertNotIn(
                    member, seen,
                    f"{member!r} is in both {seen.get(member)!r} and {canonical!r}")
                seen[member] = canonical
        for token in seen:
            self.assertNotIn(" ", token, f"{token!r} can never be a token")

    def test_bug_37_ambiguous_particles_are_never_stripped(self):
        """Position-based stripping was removed as unresolvable: "Kumar San
        Das" is a real three-part name whose first part happens to be Kumar,
        and stripping it deletes the applicant's given name. The function that
        once popped by index over the unmutated list is now a no-op, which
        also removes the index-corruption hazard."""
        self.assertEqual(normalize("Kumar Suresh Sharma").tokens,
                         ("kumar", "suresh", "sharma"))
        self.assertEqual(normalize("Kumar Suresh").tokens, ("kumar", "suresh"))
        self.assertEqual(normalize("Anil Kumar Sharma").tokens,
                         ("anil", "kumar", "sharma"))
        # No position may lose a name part.
        for tokens in (["kumar", "suresh"], ["suresh", "kumar"],
                       ["anil", "kumar", "sharma"],
                       ["kumar", "san", "das"], ["kumari", "devi"]):
            self.assertEqual(normalize(" ".join(tokens)).tokens, tuple(tokens),
                             f"stripped a name part from {tokens}")

    def test_bug_38_normalisation_never_produces_an_empty_token(self):
        import itertools as it

        alphabet = "abc"
        for length in (1, 2, 3, 4):
            for combo in it.product(alphabet, repeat=length):
                token = "".join(combo)
                self.assertTrue(normalize(token).canonical, token)


# --------------------------------------------------------------------------
# Matcher behaviour
# --------------------------------------------------------------------------


class TestBugMatchers(unittest.TestCase):
    def test_bug_39_initial_mismatch_short_circuited_the_fallbacks(self):
        """Two *different* initials returned a hard 0.0, short-circuiting the
        phonetic and character fallbacks the surrounding ordering documents as
        the intent. They are absence of evidence, not evidence of mismatch."""
        # Different initials: weak evidence, not proof of mismatch.
        self.assertGreater(_token_similarity("a", "a", "k", "k", True, True), 0.0)
        self.assertLess(_token_similarity("a", "a", "k", "k", True, True), 0.5)
        # Matching initials never reach this branch -- identical tokens are
        # handled above it -- so they score a full 1.0.
        self.assertAlmostEqual(_token_similarity("s", "s", "s", "s", True, True),
                               1.0, places=6)

    def test_bug_40_a_one_character_extension_scored_like_an_identity(self):
        """A father and a son share a surname and differ by one character, and
        no name-only matcher can be expected to separate that. What must hold is
        the weaker, true statement: the pair scores well below an *exact* match,
        and below the learned combiner's auto-approve cut-off."""
        scorer = TokenAlignedScorer()
        father_son = scorer.score("Krishna Kumar", "Krishnan Kumar")
        self.assertLess(father_son, scorer.score("Krishna Kumar", "Krishna Kumar"))
        self.assertLess(father_son, 0.95,
                        "a different-person pair must stay out of the top band")

    def test_bug_40b_prefix_credit_is_monotone_in_extension_length(self):
        """The replacement rule was a step function: a one-character extension
        was discounted heavily while a three-character extension got full
        credit, so a smaller difference was treated as more dangerous than a
        larger one. Credit must fall monotonically as the extension grows."""
        values = [_token_similarity("krishna", "krishna", "krishna" + "z" * extra,
                                    "krishna" + "z" * extra, False, False)
                  for extra in range(5)]
        for earlier, later in zip(values, values[1:]):
            self.assertGreaterEqual(earlier + 1e-12, later,
                                    f"prefix credit rose with extension length: {values}")

    def test_bug_41_score_stays_in_range_for_adversarial_input(self):
        scorer = TokenAlignedScorer()
        adversarial = [
            ("a", "aaaa"), ("", "x"), ("s", "s"), ("q", "q" * 40),
            ("a b c d e f g h", "a b c d e f g h"),
            ("z" * 30, "z" * 30 + "q"),
        ]
        for a, b in adversarial:
            for matcher in (scorer, TokenSetJaccardAlias(), PhoneticJaroWinkler()):
                score = matcher.score(a, b)
                self.assertGreaterEqual(score, 0.0, f"{matcher.key}: {a!r}/{b!r}")
                self.assertLessEqual(score, 1.0, f"{matcher.key}: {a!r}/{b!r}")

    def test_bug_42_order_bonus_weight_comment_was_backwards(self):
        """The constant was documented as "a small value" when it is a floor and
        the order term moves the score by at most 3%."""
        from name_match.algorithms import ORDER_BONUS_WEIGHT

        self.assertGreaterEqual(ORDER_BONUS_WEIGHT, 0.9)
        influence = (1.0 - ORDER_BONUS_WEIGHT)
        self.assertLessEqual(influence, 0.1)


def TokenSetJaccardAlias():
    from name_match.algorithms import TokenSetJaccard

    return TokenSetJaccard()


# --------------------------------------------------------------------------
# Report integrity
# --------------------------------------------------------------------------


class TestBugReport(unittest.TestCase):
    def test_bug_43_results_json_is_strict_json(self):
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "reports", "results.json")
        if not os.path.exists(path):
            self.skipTest("reports/results.json not generated yet")

        def reject(constant):
            raise ValueError(f"bare {constant} is not valid JSON")

        with open(path, encoding="utf-8") as handle:
            json.load(handle, parse_constant=reject)

    def test_bug_44_cross_algorithm_tables_are_keyed_not_positional(self):
        """`json.dump(sort_keys=True)` sorts the `algorithms` mapping, so a bare
        list in cost-ranked column order would be read against the wrong
        algorithm by anyone zipping the two together."""
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "reports", "results.json")
        if not os.path.exists(path):
            self.skipTest("reports/results.json not generated yet")
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        table = report["cross_algorithm_errors_at_recall_floor"]
        self.assertIn("algorithm_order", table)
        self.assertEqual(set(table["algorithm_order"]), set(report["algorithms"]))
        for category, counts in table["false_positives"].items():
            self.assertEqual(set(counts), set(report["algorithms"]), category)

    def test_bug_45_band_report_flags_are_internally_consistent(self):
        """`bands_collapsed` alone was published as "bands separable?", which
        was true for two algorithms whose auto-reject band was empty."""
        from name_match import cli

        dataset = build_dataset()
        score_sets, _extras = cli.build_score_sets(dataset)
        results = cli.analyse(score_sets)
        for key, entry in results.items():
            detail = entry["band_detail"]
            bands = entry["bands"]
            self.assertEqual(
                detail["reject_band_empty"], bands["reject_n"] == 0,
                f"{key}: reject_band_empty disagrees with the reported band size")
            self.assertEqual(
                detail["approve_band_empty"], bands["approve_n"] == 0,
                f"{key}: approve_band_empty disagrees with the reported band size")
            self.assertEqual(
                detail["three_band_pipeline_usable"],
                not (detail["approve_band_empty"] or detail["reject_band_empty"]),
                f"{key}: usability flag is not derived from the band emptiness")
            self.assertAlmostEqual(
                bands["approve_n"] + bands["review_n"] + bands["reject_n"],
                len(dataset), places=6, msg=f"{key}: bands do not partition")

    def test_bug_46_auto_approve_target_is_reported_honestly(self):
        """`high_precision_point` silently fell back to "most precise available"
        when nothing reached the target, and nothing said so."""
        from name_match.evaluate import high_precision_point

        # The top-scoring item is a negative, so no threshold is pure: the best
        # attainable precision is 0.5 and the 0.99 target is unreachable.
        point = high_precision_point([0.9, 0.5], [0, 1], 0.99)
        self.assertFalse(point.target_met)
        self.assertLess(point.precision, 0.99)

        # And a case that does meet it is reported as meeting it.
        self.assertTrue(high_precision_point([0.9, 0.1], [1, 0], 0.99).target_met)


if __name__ == "__main__":
    unittest.main()

class TestBugCoverageForUntestedFixes(unittest.TestCase):
    """Regression tests for fixes that were documented but not pinned.

    Each of these was a real defect found during the audit. The code was fixed
    and NOTES.md section 7 listed the fix, but no test exercised it, so any of
    them could have been silently reverted. They are grouped here rather than
    numbered because they were found late, after the numbered sequence had
    settled.
    """

    def test_latin_letters_nfkd_cannot_decompose_are_transliterated(self):
        """NFKD strips the accents from Latin letters but silently deletes the
        letters themselves when it cannot decompose them: `Søren` became `Sren`
        and `Władysław` became `Wadysaw`. Both are common in the transliterated
        spellings this dataset exists to handle."""
        from name_match.normalize import strip_diacritics

        self.assertEqual(strip_diacritics("Søren Władysław"), "Soren Wladyslaw")
        for original, expected in (("ø", "o"), ("Ø", "O"), ("ł", "l"),
                                   ("đ", "d"), ("þ", "th"), ("æ", "ae"),
                                   ("œ", "oe"), ("ß", "ss"), ("Ž", "Z")):
            with self.subTest(letter=original):
                self.assertEqual(strip_diacritics(original).lower(),
                                 expected.lower())

    def test_km_is_treated_as_an_honorific(self):
        """`Km.` is the short form of `Kumari`. Treated as a name it survives
        normalisation and costs an extra token to align on every pair."""
        self.assertEqual(normalize("Km. Suresh Kumar Sharma").tokens,
                         ("suresh", "kumar", "sharma"))
        self.assertEqual(normalize("km Suresh Kumar Sharma").tokens,
                         ("suresh", "kumar", "sharma"))
        # The spelling difference is deliberate and worth pinning: "Km." is an
        # abbreviation of the title Kumari, while "Kumari" written out is a
        # given name that occurs in this roster. Treating the long form as a
        # title would delete a real name part, which is the error that made
        # position-based particle removal unsafe in the first place.
        self.assertEqual(normalize("Kumari Suresh").tokens,
                         ("kumari", "suresh"))

    def test_full_string_soundex_feature_ignores_empty_codes(self):
        """`soundex` returns "" for a digit-only string, so a naive equality
        test scored 1.0 for any two different digit strings."""
        from name_match.features import _soundex_collision

        self.assertEqual(_soundex_collision("123", "456"), 0.0)
        self.assertEqual(_soundex_collision("", ""), 0.0)
        self.assertEqual(_soundex_collision("12a3", "45b6"), 0.0)
        self.assertEqual(_soundex_collision("Sharma", "Sarma"), 1.0)

    def test_post_fold_lexicon_lookup_is_necessary_for_the_documented_pair(self):
        """`venkatt` is absent from the lexicon but folds to `venkat`, which is
        present. Without the second lookup the folded form is returned raw and
        never reaches its class, and the dataset's own held-out pair silently
        stops being resolved."""
        from name_match.normalize import TRANSLIT_LEXICON, _generic_fold, normalize

        self.assertNotIn("venkatt", TRANSLIT_LEXICON)
        self.assertEqual(_generic_fold("venkatt"), "venkat")
        self.assertIn(_generic_fold("venkatt"), TRANSLIT_LEXICON)
        self.assertEqual(normalize("Venkat Suresh Rao").canonical,
                         normalize("Venkatt Suresh Rao").canonical)

    def test_model_rejects_a_non_binary_label_without_masking_it(self):
        """`int(label)` before the membership test silently turned 0.5 into 0
        and trained on it. A mixed-type list must also report the offending
        values rather than raising TypeError out of `sorted`."""
        from name_match.model import LogisticRegression

        matrix = [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        for labels in ([0, 1, 0.5, 1], [0, 1, 2, 1]):
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    LogisticRegression().fit(matrix, labels)
        with self.assertRaises(ValueError) as caught:
            LogisticRegression().fit(matrix, [0.5, "x", 0, 1])
        self.assertIn("0.5", str(caught.exception))


class TestSilentWrongAnswers(unittest.TestCase):
    """Defects that returned a plausible number instead of raising.

    Every other bug class in this file announced itself. These did not: each
    produced an ordinary-looking score for a case where the right answer is
    "we cannot tell", which is the failure mode that survives review.
    """

    def test_unreadable_script_is_undecidable_not_a_mismatch(self):
        """Two *identical* Devanagari names scored 0.000 on all five matchers.

        The normaliser folds to `[a-z0-9]`, which silently deletes every
        character NFKD cannot decompose, so the name normalises to nothing and
        the matchers read "no shared content" rather than "no comparison
        possible". In identity verification that is the worst possible
        direction: it rejects a genuine customer with total confidence.
        """
        from name_match.normalize import normalize

        for name in ("राहुल शर्मा", "李雷", "محمد الرشيد"):
            with self.subTest(name=name):
                result = normalize(name)
                self.assertTrue(result.undecidable)
                self.assertTrue(result.unreadable,
                                "undecidable without recording what was lost")

    def test_undecidable_names_score_neutral_rather_than_zero(self):
        """0.5 sits above the auto-reject band and below auto-approve, so an
        unreadable name is routed to a human. 0.0 would auto-reject it."""
        from name_match.algorithms import ALL_MATCHERS

        for name in ("राहुल शर्मा", "李雷"):
            with self.subTest(name=name):
                for matcher in ALL_MATCHERS:
                    self.assertAlmostEqual(
                        matcher.score(name, name), 0.5, places=6,
                        msg=f"{matcher.key} scored an identical unreadable "
                            f"name as something other than neutral")

    def test_a_partially_unreadable_name_is_also_undecidable(self):
        """`Ramesh कुमार Sharma` keeps two tokens and loses the middle one.
        Scoring it against `Ramesh Kumar Sharma` as a confident match or a
        confident miss would both be wrong; the middle name is simply gone."""
        from name_match.normalize import normalize

        result = normalize("Ramesh कुमार Sharma")
        self.assertTrue(result.undecidable)
        self.assertEqual(result.tokens, ("ramesh", "sharma"))
        self.assertEqual(result.unreadable, ("कुमार",))

    def test_genuinely_empty_input_is_still_a_confident_non_match(self):
        """The distinction matters: an empty name carries no information, and
        the existing guarantee that empty input never scores as agreement must
        survive the change."""
        from name_match.algorithms import ALL_MATCHERS

        for name in ("", "   ", "!!!", "..."):
            with self.subTest(name=name):
                for matcher in ALL_MATCHERS:
                    self.assertEqual(
                        matcher.score(name, "Suresh Kumar"), 0.0,
                        msg=f"{matcher.key} treated empty input as evidence")

    def test_punctuation_only_is_not_reported_as_an_unreadable_script(self):
        """`_has_letters` is what separates an unreadable script from routine
        punctuation residue, and getting it wrong would flag every em-dash."""
        from name_match.normalize import normalize

        for name in ("!!!", "...", "-- --", "()"):
            with self.subTest(name=name):
                result = normalize(name)
                self.assertFalse(result.undecidable)
                self.assertEqual(result.unreadable, ())

    def test_digit_only_names_are_not_capped_by_a_missing_phonetic_term(self):
        """Soundex returns "" for a string with no letters, so the phonetic
        term contributed nothing and the blend capped the score at 0.5 however
        identical the two strings were -- four matchers said 1.0 and this one
        said a coin flip."""
        from name_match.algorithms import BASELINE_MATCHERS

        # The learned arm is excluded: its behaviour on a digit-only pair
        # depends on whether a model is loaded, and there is no digit-only row
        # in the dataset to have fitted one on. The four deterministic matchers
        # are the ones that made a number up.
        for matcher in BASELINE_MATCHERS:
            self.assertAlmostEqual(matcher.score("12345", "12345"), 1.0, places=6,
                                   msg=matcher.key)

    def test_phonetic_disagreement_is_still_penalised(self):
        """The other half of the same fix: two well-formed codes that differ is
        evidence, not a missing measurement. Renormalising onto the character
        term for that case pushed a genuine collision from 0.47 to 0.68."""
        from name_match.algorithms import PhoneticJaroWinkler

        matcher = PhoneticJaroWinkler()
        self.assertLess(matcher.score("Suresh Kumar", "Rekha Menon"), 0.45)
        self.assertLess(matcher.score("Ram", "Shyam"), 0.1)

    def test_an_unfitted_learned_model_warns_instead_of_returning_0_5(self):
        """The documented fallback is 'degrade to review', which holds only
        because 0.5 lands between the bands. A silent constant that happened to
        sit in the right place is not a safety property."""
        import warnings

        from name_match.features import FEATURE_NAMES
        from name_match.model import LogisticRegression

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            LogisticRegression().predict_proba([0.0] * len(FEATURE_NAMES))
        self.assertTrue(
            any(issubclass(w.category, RuntimeWarning) for w in caught),
            "an unfitted model returned a neutral score in silence")

    def test_a_fitted_model_does_not_warn(self):
        import warnings

        from name_match.dataset import build_dataset
        from name_match.features import FEATURE_NAMES, extract_features
        from name_match.model import LogisticRegression

        pairs = build_dataset()[:80]
        matrix = [extract_features(p.name_a, p.name_b)[0] for p in pairs]
        labels = [p.label for p in pairs]
        model = LogisticRegression().fit(matrix, labels, FEATURE_NAMES)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.predict_proba(matrix[0])
        self.assertFalse([w for w in caught
                          if issubclass(w.category, RuntimeWarning)])
