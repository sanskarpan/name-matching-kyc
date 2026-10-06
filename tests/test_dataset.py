"""Integrity tests for the generated dataset.

These are the tests that protect the *dataset*, not the code. A dataset that
quietly drifts into being easy, self-contradictory or label-leaked would make
every downstream conclusion worthless while all the algorithm tests still pass.

The most important block is :class:`TestRationaleIntegrity`: a rationale is
human-readable evidence about one row, and the three ways it used to lie --
quoting a name pair the row does not contain, repeating its own clause, and
describing a middle name the person does not have -- are all checkable, so they
are all checked.

Run with ``python3 -m unittest discover -s tests -t .`` from the repo root.
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest
from collections import Counter

from name_match import dataset as dataset_module
from name_match.dataset import (CATEGORIES, PEOPLE, build_dataset, read_csv,
                                 write_csv)
from name_match.normalize import normalize
from name_match.phonetics import phonetic_agreement

DATASET = build_dataset()
POSITIVES = [p for p in DATASET if p.label == 1]
NEGATIVES = [p for p in DATASET if p.label == 0]
HARD_NEGATIVES = [p for p in NEGATIVES if p.category.startswith("hard_negative")]

#: Floor per hard-negative category. A shape with fewer rows than this is not
#: being measured, whatever the total count says.
HARD_NEGATIVE_FLOORS = {
    "hard_negative_sibling": 3,
    "hard_negative_sibling_reordered": 3,
    "hard_negative_parent_child": 3,
    "hard_negative_identical_name": 3,
    "hard_negative_near_duplicate": 10,
    "hard_negative_dropped_token": 3,
    "hard_negative_phonetic": 10,
    "hard_negative_common_surname": 10,
    "hard_negative_partial_overlap": 10,
}


def _tokens(name: str) -> set[str]:
    """Normalised token set -- what a matcher would actually compare."""
    return set(normalize(name).tokens)


def _key(name_a: str, name_b: str) -> tuple[str, str]:
    return (name_a, name_b) if name_a <= name_b else (name_b, name_a)


def _tail(pair) -> str:
    return pair.rationale.split(": ", 1)[1]


def _quoted(text: str) -> list[str]:
    return re.findall(r"'([^']*)'", text)


class TestLabelIntegrity(unittest.TestCase):
    """The single most important property: labels come from identity."""

    def test_label_equals_identity_equality(self):
        for pair in DATASET:
            with self.subTest(pair=pair.pair_id):
                self.assertEqual(
                    pair.label == 1, pair.person_a == pair.person_b,
                    f"{pair.pair_id}: label {pair.label} disagrees with "
                    f"person identity {pair.person_a}/{pair.person_b}")

    def test_labels_are_binary(self):
        self.assertTrue(all(p.label in (0, 1) for p in DATASET))

    def test_no_negative_claims_same_person(self):
        for pair in NEGATIVES:
            self.assertNotEqual(pair.person_a, pair.person_b, pair.pair_id)

    def test_every_person_id_is_known(self):
        for pair in DATASET:
            self.assertIn(pair.person_a, PEOPLE, pair.pair_id)
            self.assertIn(pair.person_b, PEOPLE, pair.pair_id)

    def test_pair_ids_unique(self):
        ids = [p.pair_id for p in DATASET]
        self.assertEqual(len(ids), len(set(ids)))


class TestPairKeyIntegrity(unittest.TestCase):
    """One comparison, one row, one label."""

    def test_pairs_are_deduplicated_on_the_unordered_name_pair(self):
        # The generator used to key on the *ordered* tuple, so 'A' vs 'B' and
        # 'B' vs 'A' both survived -- and "Nisha Yadav" vs "Nisha Kumari Yadav"
        # appeared twice, once as a match and once as a non-match.
        seen: dict[tuple[str, str], str] = {}
        for pair in DATASET:
            key = _key(pair.name_a, pair.name_b)
            self.assertNotIn(key, seen,
                             f"{pair.pair_id} repeats {key}, already emitted as "
                             f"{seen.get(key)}")
            seen[key] = pair.pair_id

    def test_no_string_pair_carries_two_labels(self):
        labels: dict[tuple[str, str], int] = {}
        for pair in DATASET:
            key = _key(pair.name_a, pair.name_b)
            if key in labels:
                self.fail(f"{key} is labelled both {labels[key]} and {pair.label} "
                          f"({pair.pair_id})")
            labels[key] = pair.label

    def test_the_nisha_yadav_collision_is_gone(self):
        # The specific regression: one roster entry with a middle name and
        # another without it rendered the same comparison once as a match
        # (dropped middle name) and once as a non-match (two people).
        for pair in DATASET:
            if _key(pair.name_a, pair.name_b) == _key("Nisha Yadav",
                                                      "Nisha Kumari Yadav"):
                self.assertEqual(pair.label, 0,
                                 f"{pair.pair_id}: these are two different people")


class TestRationaleIntegrity(unittest.TestCase):
    """A rationale must describe the row it sits on."""

    def test_every_pair_has_a_rationale(self):
        for pair in DATASET:
            self.assertTrue(pair.rationale.strip(), pair.pair_id)

    def test_rationale_opens_with_the_rows_own_strings(self):
        # Composed by dataset._rationale, so a builder cannot quote a pair the
        # row does not have.
        for pair in DATASET:
            self.assertTrue(
                pair.rationale.startswith(f"{pair.name_a!r} vs {pair.name_b!r}: "),
                f"{pair.pair_id}: rationale does not lead with its own names: "
                f"{pair.rationale!r}")

    def test_rationale_quotes_only_the_rows_own_names(self):
        for pair in DATASET:
            quoted = sorted(set(_quoted(pair.rationale)))
            self.assertEqual(quoted, sorted({pair.name_a, pair.name_b}),
                             f"{pair.pair_id} quotes {quoted}")

    def test_rationale_never_repeats_a_clause(self):
        for pair in DATASET:
            clauses = _tail(pair).split("; ")
            self.assertEqual(len(clauses), len(set(clauses)),
                             f"{pair.pair_id} repeats a clause: {clauses}")

    def test_rationale_is_not_the_same_sentence_twice_over(self):
        texts = Counter(p.rationale for p in DATASET)
        worst = max(texts.values())
        self.assertLessEqual(worst, 4,
                             f"a rationale appears {worst} times: "
                             f"{[t for t, n in texts.items() if n == worst][:1]}")

    def test_typo_rationales_describe_the_edit_the_row_actually_has(self):
        # Catches a row claiming a duplicated character where the real edit was
        # a deleted space, or a dropped vowel where the edit was elsewhere.
        for pair in DATASET:
            if pair.category != "typo":
                continue
            tail = _tail(pair)
            match = re.search(r"\(([^()]+) -> ([^()]+)\)", tail)
            self.assertIsNotNone(match, pair.rationale)
            before, after = match.groups()
            # The annotation runs canonical -> as-written, and the typo is
            # always on the first side of the row.
            self.assertIn(before.lower(), _tokens(pair.name_b), pair.pair_id)
            self.assertIn(after.lower(), _tokens(pair.name_a), pair.pair_id)
            described = (dataset_module._edit_phrase(before, after),
                         dataset_module._dup_or_insert(before, after))
            self.assertTrue(any(phrase in tail for phrase in described),
                            f"{pair.pair_id} claims {described} but the strings say "
                            f"{pair.name_a!r} / {pair.name_b!r}")

    def test_middle_name_rows_lose_a_token_the_rationale_names(self):
        for pair in DATASET:
            if pair.category != "middle_name":
                continue
            tokens_a = {t.lower() for t in normalize(pair.name_a).tokens}
            tokens_b = {t.lower() for t in normalize(pair.name_b).tokens}
            tail = _tail(pair)
            named = re.findall(r"the middle name ([A-Za-z][A-Za-z ]*) appears", tail)
            qualifier = re.search(r"qualifies the name with [SDW]/o ([A-Za-z][A-Za-z ]*)",
                                  tail)
            reordered = "written before the given name" in tail
            self.assertTrue(named or qualifier or reordered,
                            f"{pair.pair_id} names no missing token: {tail!r}")
            if reordered and not named and not qualifier:
                # Nothing was dropped, so the difference is order alone.
                self.assertEqual(tokens_a, tokens_b, pair.pair_id)
                continue
            if named:
                self.assertNotEqual(tokens_a, tokens_b, pair.pair_id)
            for phrase in named:
                for token in phrase.split():
                    self.assertTrue((token.lower() in tokens_a)
                                    != (token.lower() in tokens_b),
                                    f"{pair.pair_id}: {token} is on both sides")
            if qualifier:
                # The related person's name belongs to the qualifier, never to
                # the applicant: it must be absent from both documents.
                related = list(normalize(qualifier.group(1)).tokens)
                self.assertNotIn(related[0], tokens_a, pair.pair_id)
                self.assertNotIn(related[0], tokens_b, pair.pair_id)

    def test_honorific_rows_never_invent_a_father(self):
        # "Shri Kiran Pawar S/o Kiran" was generated by falling back to the
        # person's own given name when the roster recorded no father.
        for pair in DATASET:
            if pair.category != "honorific":
                continue
            person = PEOPLE[pair.person_a]
            qualifier = re.search(r"([SDW])/o ([A-Za-z][A-Za-z ]*)", _tail(pair))
            if not qualifier:
                continue
            self.assertTrue(person.father or person.spouse,
                            f"{pair.pair_id} has no relative to qualify with")
            known = {person.father.lower(), person.spouse.lower()}
            self.assertIn(qualifier.group(2).lower(), known, pair.pair_id)

    def test_transliteration_rows_never_claim_a_dropped_middle_name_it_lacks(self):
        for pair in DATASET:
            if pair.category != "transliteration":
                continue
            person = PEOPLE[pair.person_a]
            for phrase in re.findall(r"the middle name ([A-Za-z][A-Za-z ]*) appears",
                                    _tail(pair)):
                self.assertIn(phrase.split()[-1].lower(),
                              {m.lower() for m in person.middle},
                              f"{pair.pair_id}: {person.person_id} has no middle "
                              f"name {phrase}")


class TestRosterIntegrity(unittest.TestCase):
    """The roster is part of the label, so its notes have to be true."""

    #: Words that make "these two roster entries are different people" explicit.
    _DIFFERENCE_MARKERS = (
        "different person", "unrelated", "another", "second", "third",
        "not in", "brother", "sister", "sibling", "father", "son", "mother",
        "parent", "co-parent", "husband", "wife", "spouse", "household",
    )

    def test_no_note_claims_two_identities_are_one_person(self):
        for person in dataset_module._ROSTER:
            referenced = set(re.findall(r"P\d{3}", person.note))
            referenced.discard(person.person_id)
            self.assertNotIn("same person", person.note.lower(),
                             f"{person.person_id}: {person.note}")
            self.assertFalse(
                re.search(r"same (identity|household entry)", person.note.lower()),
                f"{person.person_id}: {person.note}")

    def test_notes_that_name_another_entry_say_they_differ(self):
        for person in dataset_module._ROSTER:
            referenced = set(re.findall(r"P\d{3}", person.note))
            referenced.discard(person.person_id)
            if not referenced:
                continue
            note = person.note.lower()
            self.assertTrue(any(marker in note for marker in self._DIFFERENCE_MARKERS),
                            f"{person.person_id} names {sorted(referenced)} but never "
                            f"says how it differs: {person.note}")

    def test_every_identity_fact_is_used_by_some_row(self):
        # _IDENTITY_FACTS is hand-written and can rot; a fact that no row uses
        # is a claim about the data that has quietly stopped being true.
        used = {pair.rationale for pair in DATASET}
        for left, right, clause in dataset_module._IDENTITY_FACTS:
            self.assertTrue(
                any(clause in rationale for rationale in used),
                f"identity fact for {left}/{right} is attached to no row")

    def test_held_out_surfaces_are_canonical_renderings_of_their_person(self):
        # A held-out variant must belong to the person it is filed under. It
        # used to be allowed to be another roster entry's spelling, which made
        # one comparison simultaneously a match and a non-match.
        for table in (dataset_module._HELD_OUT_TRANSLIT_UNRESOLVED,
                      dataset_module._HELD_OUT_TRANSLIT_FOLDED):
            for pid, surface_a, surface_b, _why in table:
                self.assertEqual(
                    list(normalize(surface_a).tokens),
                    list(normalize(PEOPLE[pid].full).tokens),
                    f"{pid}: {surface_a!r} is not that identity's canonical form")
                self.assertNotEqual(surface_a, surface_b, pid)


class TestTransliterationLookupsHit(unittest.TestCase):
    """The variant tables are keyed lowercase and looked up lowercase."""

    def test_variant_tables_are_lowercase_keys(self):
        for table_name in ("_GIVEN_VARIANTS", "_FAMILY_VARIANTS"):
            table = getattr(dataset_module, table_name)
            for key in table:
                self.assertEqual(key, key.lower(), f"{table_name}: {key!r}")
                self.assertEqual(key, key.strip(), f"{table_name}: {key!r}")

    def test_lookup_hits_for_every_registered_variant(self):
        for person in dataset_module._ROSTER:
            if dataset_module._GIVEN_VARIANTS.get(person.given.lower()):
                result = dataset_module._b_translit_given(person)
                self.assertIsNotNone(
                    result, f"{person.person_id}: {person.given!r} has a registered "
                            f"variant but the lookup missed")
                self.assertEqual(result[2], "transliteration", person.person_id)
            if dataset_module._FAMILY_VARIANTS.get(person.family.lower()):
                result = dataset_module._b_translit_family(person)
                self.assertIsNotNone(
                    result, f"{person.person_id}: {person.family!r} has a registered "
                            f"variant but the lookup missed")
                self.assertEqual(result[2], "transliteration", person.person_id)

    def test_no_variant_is_the_identity_it_keys(self):
        for table_name in ("_GIVEN_VARIANTS", "_FAMILY_VARIANTS"):
            table = getattr(dataset_module, table_name)
            for key, variants in table.items():
                for variant in variants:
                    self.assertNotEqual(variant.lower(), key.lower(),
                                        f"{table_name}[{key!r}] offers no change")

    def test_transliteration_rows_use_registered_spellings(self):
        # The old builders missed the table and emitted _typo(given,
        # "substitute") instead, so every transliteration row was a one-character
        # typo wearing the category's name.
        for pair in DATASET:
            if pair.category != "transliteration":
                continue
            person = PEOPLE[pair.person_a]
            canonical = {t.lower() for t in normalize(person.full).tokens}
            for token in {t.lower() for t in normalize(pair.name_a).tokens} - canonical:
                self.assertIn(
                    token, dataset_module._ALL_VARIANTS,
                    f"{pair.pair_id}: {token!r} is not a registered spelling variant")

    def test_transliteration_category_has_depth(self):
        translit = [p for p in DATASET if p.category == "transliteration"]
        self.assertGreaterEqual(len(translit), 12)


class TestHardNegativesAreHard(unittest.TestCase):
    """A category name is a claim about the two strings."""

    def _has_token_shared_or_phonetic_trap(self, pair) -> bool:
        tokens_a = {t.lower() for t in normalize(pair.name_a).tokens}
        tokens_b = {t.lower() for t in normalize(pair.name_b).tokens}
        if tokens_a & tokens_b:
            return True
        return any(left != right and phonetic_agreement(left, right) >= 0.8
                   for left in tokens_a for right in tokens_b)

    def test_every_hard_negative_is_confusable(self):
        for pair in HARD_NEGATIVES:
            self.assertTrue(
                self._has_token_shared_or_phonetic_trap(pair),
                f"{pair.pair_id}: {pair.name_a!r} / {pair.name_b!r} shares no token and "
                f"has no phonetic agreement, so it is trivial for every algorithm")

    def test_most_hard_negatives_share_a_token(self):
        shared = [p for p in HARD_NEGATIVES if self._has_token_shared_or_phonetic_trap(p)]
        self.assertGreaterEqual(
            len([p for p in HARD_NEGATIVES
                 if {t.lower() for t in normalize(p.name_a).tokens}
                 & {t.lower() for t in normalize(p.name_b).tokens}]),
            len(HARD_NEGATIVES) * 0.7,
            "most hard negatives should share at least one name token; the rest "
            "must earn the label with a phonetic trap")

    def test_sibling_rows_share_a_household(self):
        # _n_sibling used to be applied to arbitrary cross-person pairs, which
        # produced "siblings" rationales for two people with no shared household.
        for pair in DATASET:
            if not pair.category.startswith("hard_negative_sibling"):
                continue
            a, b = PEOPLE[pair.person_a], PEOPLE[pair.person_b]
            self.assertTrue(dataset_module._same_household(a, b),
                            f"{pair.pair_id}: {a.person_id} and {b.person_id} do not "
                            f"share a household")

    def test_sibling_reordered_rows_are_siblings(self):
        reordered = [p for p in DATASET
                     if p.category == "hard_negative_sibling_reordered"]
        self.assertGreaterEqual(len(reordered), 3,
                                "the reordered sibling shape is declared and must be "
                                "emitted")
        for pair in reordered:
            self.assertTrue(dataset_module._same_household(
                PEOPLE[pair.person_a], PEOPLE[pair.person_b]), pair.pair_id)
            # One side is genuinely surname-first.
            self.assertNotEqual(
                pair.name_b.split()[0], PEOPLE[pair.person_b].given, pair.pair_id)

    def test_dropped_token_rows_are_token_subsets(self):
        # 'Meena Shah' vs 'Rohit' is not a dropped token, it is a short name.
        for pair in DATASET:
            if pair.category != "hard_negative_dropped_token":
                continue
            left = {t.lower() for t in normalize(pair.name_a).tokens}
            right = {t.lower() for t in normalize(pair.name_b).tokens}
            self.assertTrue(left < right or right < left,
                            f"{pair.pair_id}: neither side is a strict subset of the "
                            f"other: {sorted(left)} / {sorted(right)}")

    def test_phonetic_rows_have_a_phonetic_trap(self):
        for pair in DATASET:
            if pair.category != "hard_negative_phonetic":
                continue
            left = {t.lower() for t in normalize(pair.name_a).tokens}
            right = {t.lower() for t in normalize(pair.name_b).tokens}
            trap = any(a != b and phonetic_agreement(a, b) >= 0.8
                       for a in left for b in right)
            self.assertTrue(trap, f"{pair.pair_id} has zero phonetic agreement")

    def test_identical_name_rows_are_byte_identical(self):
        for pair in DATASET:
            if pair.category != "hard_negative_identical_name":
                continue
            self.assertEqual(pair.name_a, pair.name_b, pair.pair_id)
            self.assertNotEqual(pair.person_a, pair.person_b, pair.pair_id)

    def test_common_surname_rows_really_share_a_surname(self):
        for pair in DATASET:
            if pair.category != "hard_negative_common_surname":
                continue
            shared = dataset_module._shared_family_token(PEOPLE[pair.person_a],
                                                         PEOPLE[pair.person_b])
            self.assertTrue(shared,
                            f"{pair.pair_id}: no shared family name for a "
                            f"common-surname row")

    def test_near_duplicate_rows_differ_by_one_token_or_one_character(self):
        for pair in DATASET:
            if pair.category != "hard_negative_near_duplicate":
                continue
            left = list(normalize(pair.name_a).tokens)
            right = list(normalize(pair.name_b).tokens)
            self.assertTrue(len(left) == len(right) and len(left) >= 2, pair.pair_id)
            diffs = [(a, b) for a, b in zip(left, right) if a != b]
            self.assertEqual(len(diffs), 1, pair.pair_id)

    def test_unrelated_rows_are_trivially_separable(self):
        for pair in DATASET:
            if pair.category != "unrelated":
                continue
            left = {t.lower() for t in normalize(pair.name_a).tokens}
            right = {t.lower() for t in normalize(pair.name_b).tokens}
            self.assertFalse(left & right, pair.pair_id)
            self.assertFalse(any(a != b and phonetic_agreement(a, b) >= 0.8
                                 for a in left for b in right), pair.pair_id)


class TestSizeAndBalance(unittest.TestCase):
    def test_meets_assignment_minimum(self):
        # The brief asks for 100+ labelled pairs.
        self.assertGreaterEqual(len(DATASET), 100)

    def test_both_classes_well_represented(self):
        self.assertGreaterEqual(len(POSITIVES), 80)
        self.assertGreaterEqual(len(NEGATIVES), 80)

    def test_negatives_outnumber_or_match_positives(self):
        # In production, name comparisons are overwhelmingly non-matches, and a
        # dataset where positives dominate makes precision look far better than
        # it is.
        self.assertGreaterEqual(len(NEGATIVES), len(POSITIVES) * 0.8)

    def test_hard_negatives_are_a_real_share(self):
        # A dataset dominated by trivially separable negatives flatters every
        # algorithm and hides the failure modes that matter.
        self.assertGreaterEqual(len(HARD_NEGATIVES), 60,
                                "too few hard negatives to be informative")

    def test_hard_negatives_dominate_the_negatives(self):
        self.assertGreater(len(HARD_NEGATIVES) / len(NEGATIVES), 0.8)

    def test_easy_unrelated_negatives_are_a_minority(self):
        unrelated = [p for p in DATASET if p.category == "unrelated"]
        self.assertLess(len(unrelated), len(DATASET) * 0.2)
        self.assertGreaterEqual(len(unrelated), 8,
                                "the trivially separable control group should still "
                                "be present")


class TestCategoryCoverage(unittest.TestCase):
    def test_categories_are_declared(self):
        for pair in DATASET:
            self.assertIn(pair.category, CATEGORIES, pair.pair_id)

    def test_every_declared_category_is_emitted(self):
        # A category in the header that no row carries is a category the reader
        # will assume was measured.
        present = {p.category for p in DATASET}
        for declared in CATEGORIES:
            self.assertIn(declared, present,
                          f"{declared} is documented but never emitted")

    def test_every_seed_category_is_present(self):
        present = {p.category for p in DATASET}
        for required in (
            "initials",              # from the brief
            "surname_first",         # from the brief
            "transliteration",       # from the brief
            "middle_name",           # from the brief
            "honorific",             # from the brief
            "suffix",                # from the brief
            "hard_negative_sibling",          # from the brief
            "hard_negative_common_surname",   # from the brief
            "hard_negative_near_duplicate",   # from the brief
            "transliteration_heldout",        # our own addition
        ):
            self.assertIn(required, present, f"category {required} is missing")

    def test_positive_categories_have_depth(self):
        counts = Counter(p.category for p in POSITIVES)
        for category in counts:
            # compound_surname is thinner by construction: only identities whose
            # surname is genuinely joint can produce one, and a roster padded
            # with joint surnames purely to hit a count would be padding.
            floor = 8 if category == "compound_surname" else 10
            self.assertGreaterEqual(counts[category], floor, category)

    def test_hard_negative_categories_have_depth(self):
        counts = Counter(p.category for p in HARD_NEGATIVES)
        for category, floor in HARD_NEGATIVE_FLOORS.items():
            self.assertGreaterEqual(counts.get(category, 0), floor, category)

    def test_difficulty_values_valid(self):
        for pair in DATASET:
            self.assertIn(pair.difficulty, ("easy", "medium", "hard"), pair.pair_id)

    def test_difficulty_agrees_with_category(self):
        for pair in DATASET:
            if pair.category == "unrelated":
                self.assertEqual(pair.difficulty, "easy", pair.pair_id)
            if pair.category == "hard_negative_identical_name":
                self.assertEqual(pair.difficulty, "hard", pair.pair_id)


class TestDatasetIsNonTrivial(unittest.TestCase):
    """The dataset must actually be hard, or the evaluation means nothing."""

    def test_positives_are_not_all_identical_strings(self):
        identical = [p for p in POSITIVES if p.name_a == p.name_b]
        self.assertEqual(identical, [],
                         "positives must differ on the surface to be informative")

    def test_most_positives_survive_normalisation(self):
        # Some positives *should* be solvable by cleaning alone (case,
        # honorific). But if that were all of them, the dataset would not
        # exercise any matching logic.
        trivial = [p for p in POSITIVES
                   if normalize(p.name_a).canonical == normalize(p.name_b).canonical]
        self.assertLess(len(trivial), len(POSITIVES) * 0.5,
                        "too many positives are resolved by normalisation alone")

    def test_identical_string_negatives_exist(self):
        # Two different people with the *same* name. This is irreducible for any
        # name-only matcher and must be represented, otherwise the reported
        # false-positive floor looks artificially low.
        identical = [p for p in NEGATIVES if p.name_a == p.name_b]
        self.assertGreaterEqual(len(identical), 1,
                                "no irreducible same-name-different-person pair")

    def test_transliteration_heldout_pairs_are_actually_held_out(self):
        """The ``transliteration_heldout`` category claims normalisation cannot
        bridge the pair. If an edit to the lexicon or the generic fold starts
        resolving it, the category silently stops measuring generalisation and
        this test fails.
        """
        from name_match.dataset import _HELD_OUT_TRANSLIT_UNRESOLVED

        for pid, surface_a, surface_b, _why in _HELD_OUT_TRANSLIT_UNRESOLVED:
            left = normalize(surface_a).canonical
            right = normalize(surface_b).canonical
            self.assertNotEqual(
                left, right,
                f"{pid}: normalisation now resolves {surface_a!r} / {surface_b!r}, "
                f"so this pair no longer tests generalisation")

        present = [p for p in DATASET if p.category == "transliteration_heldout"]
        self.assertGreaterEqual(len(present), 10)

    def test_held_out_translit_table_matches_what_normalisation_actually_does(self):
        """The UNRESOLVED/FOLDED split is a claim about the normaliser.

        It was previously asserted only in a comment. Three of the entries in
        the FOLDED table were in fact resolved by a lexicon entry rather than
        by the generic fold, which made "held out from the lexicon" false for
        them, and one was resolved by neither. Checking the mechanism, not just
        the outcome, is what catches the lexicon case.
        """
        from name_match.dataset import (_HELD_OUT_TRANSLIT_FOLDED,
                                         _HELD_OUT_TRANSLIT_UNRESOLVED)
        from name_match.normalize import TRANSLIT_LEXICON, _generic_fold

        def novel_tokens(surface_a: str, surface_b: str) -> list[str]:
            """Tokens of the variant spelling that are not simply copied through
            from the canonical rendering. A surname like `Singh` is a lexicon
            member in *both* spellings and says nothing about whether the pair
            is held out; the changed given-name spelling is what matters."""
            carried = set(normalize(surface_a).tokens)
            return [t for t in normalize(surface_b).tokens if t not in carried]

        for table, name in ((_HELD_OUT_TRANSLIT_UNRESOLVED, "UNRESOLVED"),
                            (_HELD_OUT_TRANSLIT_FOLDED, "FOLDED")):
            for pid, surface_a, surface_b, _why in table:
                with self.subTest(table=name, pair=(surface_a, surface_b)):
                    novel = novel_tokens(surface_a, surface_b)
                    self.assertTrue(
                        novel,
                        f"{pid}: {surface_b!r} introduces no new spelling, so "
                        f"there is nothing held out about it")
                    # At least one *changed* spelling must be one the lexicon
                    # has never seen. Not all of them need to be: a variant
                    # that also carries a known surname spelling
                    # ("Priyanaka Chaterjee") is still a genuine
                    # generalisation test through the given name, and
                    # insisting otherwise would only force the dataset to
                    # invent surnames nobody writes.
                    unseen = [t for t in novel if t not in TRANSLIT_LEXICON]
                    self.assertTrue(
                        unseen,
                        f"{pid}: every changed spelling in {surface_b!r} "
                        f"({novel}) is a curated lexicon member, so the pair is "
                        f"not held out from the lexicon at all")
                    resolved = (normalize(surface_a).canonical
                                == normalize(surface_b).canonical)
                    self.assertEqual(
                        resolved, name == "FOLDED",
                        f"{pid}: {surface_a!r}/{surface_b!r} is "
                        f"{'resolved' if resolved else 'unresolved'} but sits in "
                        f"{name}")

        def lexicon_only(surface: str) -> tuple[str, ...]:
            """Canonical tokens with the generic fold disabled.

            This is the check that actually tests the documented claim. Asserting
            that a novel token *changes* under the fold was not enough: it passed
            for "Venkatt", which folds to "venkat" but is only resolved because
            the folded form then reaches a lexicon class. Disabling the fold
            shows what the lexicon alone can do, which is nothing.
            """
            tokens = normalize(surface).tokens
            return tuple(TRANSLIT_LEXICON.get(t, t) for t in tokens)

        for pid, surface_a, surface_b, _why in _HELD_OUT_TRANSLIT_FOLDED:
            with self.subTest(pair=(surface_a, surface_b)):
                self.assertTrue(
                    any(_generic_fold(t) != t
                        for t in novel_tokens(surface_a, surface_b)),
                    f"{pid}: no novel token of {surface_b!r} changes under the "
                    f"generic fold")
                self.assertNotEqual(
                    lexicon_only(surface_a), lexicon_only(surface_b),
                    f"{pid}: the lexicon alone resolves {surface_a!r}/"
                    f"{surface_b!r}, so the generic fold is not what closes the "
                    f"gap and this pair does not belong in FOLDED")

    def test_hard_negative_families_are_present(self):
        # Siblings, parent/child and same-name-different-person are the three
        # structures the brief calls out explicitly.
        for category in ("hard_negative_sibling",
                         "hard_negative_parent_child",
                         "hard_negative_near_duplicate"):
            self.assertGreaterEqual(
                sum(1 for p in DATASET if p.category == category), 3, category)

    def test_no_positive_reuses_a_comparison_a_negative_claims(self):
        keys = {_key(p.name_a, p.name_b) for p in NEGATIVES}
        for pair in POSITIVES:
            self.assertNotIn(_key(pair.name_a, pair.name_b), keys, pair.pair_id)


class TestDeterminism(unittest.TestCase):
    def test_generation_is_reproducible(self):
        first = [(p.pair_id, p.name_a, p.name_b, p.label) for p in build_dataset()]
        second = [(p.pair_id, p.name_a, p.name_b, p.label) for p in build_dataset()]
        self.assertEqual(first, second)

    def test_csv_round_trip_is_lossless(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "pairs.csv")
            write_csv(DATASET, path)
            reloaded = read_csv(path)
        self.assertEqual(len(reloaded), len(DATASET))
        for original, loaded in zip(DATASET, reloaded):
            self.assertEqual(original.as_row(), loaded.as_row())

    def test_csv_header_documents_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "pairs.csv")
            write_csv(DATASET, path)
            with open(path, encoding="utf-8") as handle:
                header = handle.read(4000)
        self.assertIn("# label: 1 = same person", header)
        self.assertIn("never from string similarity", header)
        for category in CATEGORIES:
            self.assertIn(category, header, f"{category} not documented in header")


if __name__ == "__main__":
    unittest.main()

class TestPositiveCategoryShapes(unittest.TestCase):
    """Every positive row must genuinely have the shape its category claims.

    The negatives have had shape tests all along; the positives did not, and
    the asymmetry was not cosmetic. A stride that collapsed onto a single
    builder meant `surname_first` tested reordering only in combination with an
    initialised given name, so the per-category error tables could not separate
    "fails on ordering" from "fails on initials" -- and no test noticed.
    """

    def _rows(self, category):
        return [p for p in DATASET if p.category == category]

    def _initialised(self, name):
        return any(len(t) == 1 and t.isalpha() for t in normalize(name).tokens)

    def test_initials_rows_really_contain_an_initial(self):
        rows = self._rows("initials")
        self.assertGreaterEqual(len(rows), 10)
        for pair in rows:
            with self.subTest(pair=(pair.name_a, pair.name_b)):
                self.assertTrue(
                    self._initialised(pair.name_a),
                    f"{pair.name_a!r} carries no initial, so it is not an "
                    f"initials rendering")

    def test_most_initials_rows_abbreviate_the_given_name(self):
        """The brief's seeded shape is specifically *given* name initials."""
        rows = self._rows("initials")
        given = [p for p in rows
                 if len(normalize(p.name_a).tokens[0]) == 1]
        self.assertGreaterEqual(
            len(given), len(rows) - 3,
            f"only {len(given)} of {len(rows)} initials rows abbreviate the "
            f"given name; the seeded category is under-tested")

    def test_surname_first_rows_really_reverse_the_token_order(self):
        rows = self._rows("surname_first")
        self.assertGreaterEqual(len(rows), 10)
        for pair in rows:
            with self.subTest(pair=(pair.name_a, pair.name_b)):
                left = normalize(pair.name_a).tokens
                right = normalize(pair.name_b).tokens
                self.assertTrue(
                    left[0] in right,
                    f"{pair.name_a!r} does not begin with a token of "
                    f"{pair.name_b!r}: {left} vs {right}")
                self.assertNotEqual(
                    left[0], right[0],
                    f"{pair.name_a!r} and {pair.name_b!r} both start with the "
                    f"same token, so no reordering happened")
                # The trailing token is the given name: either the full name
                # moved to the end, or it was reduced to an initial.
                given = left[-1]
                self.assertTrue(
                    given == right[0] or (len(given) == 1 and given.isalpha()),
                    f"{pair.name_a!r} does not end with the given name: "
                    f"{left} vs {right}")

    def test_surname_first_is_not_entirely_confounded_with_initials(self):
        """Ordering has to be testable on its own, or the per-category error
        tables cannot attribute a failure to ordering rather than initials."""
        rows = self._rows("surname_first")
        pure = [p for p in rows if not self._initialised(p.name_a)]
        self.assertGreaterEqual(
            len(pure), len(rows) // 2,
            f"only {len(pure)} of {len(rows)} surname_first rows preserve the "
            f"full given name, so ordering is never tested in isolation")

    def test_middle_name_rows_really_differ_in_the_middle_name(self):
        """A middle-name row must differ *in the raw strings*.

        Normalised tokens are allowed to be equal, because that is precisely
        what a relationship qualifier does: "Kavitha Reddy W/o Ravi Reddy" and
        "Kavitha Reddy" are the same applicant with and without her husband's
        name, and the normaliser is supposed to drop the trailing qualifier so
        the two documents compare equal. Asserting on normalised tokens here
        would fail on the most correct rows in the category.
        """
        from name_match.normalize import (HONORIFICS, SLASH_QUALIFIERS,
                                           SPELLED_QUALIFIERS, SUFFIXES)

        consumable = (SLASH_QUALIFIERS | SPELLED_QUALIFIERS | SUFFIXES
                      | HONORIFICS)
        rows = self._rows("middle_name")
        self.assertGreaterEqual(len(rows), 10)
        for pair in rows:
            with self.subTest(pair=(pair.name_a, pair.name_b)):
                self.assertNotEqual(pair.name_a, pair.name_b,
                                    f"{pair.name_a!r} appears on both sides")
                left = normalize(pair.name_a).tokens
                right = normalize(pair.name_b).tokens
                if left == right:
                    # Only legitimate when the extra text is something the
                    # normaliser is *supposed* to consume.
                    extra = (set(pair.name_a.lower().split())
                             | set(pair.name_b.lower().split())) - set(left)
                    self.assertTrue(
                        extra & {q.lower() for q in consumable},
                        f"{pair.name_a!r} / {pair.name_b!r} normalise "
                        f"identically and the extra words {sorted(extra)} are "
                        f"not a qualifier this normaliser consumes")

    def test_some_middle_name_rows_use_a_relational_qualifier(self):
        """Half the category should be plain drops/reorders and half
        qualifiers, so neither shape is tested only by accident."""
        rows = self._rows("middle_name")
        qualified = [p for p in rows
                     if any(marker in f"{p.name_a} {p.name_b}"
                            for marker in ("S/o", "W/o", "D/o"))
                     or any(q in f"{p.name_a} {p.name_b}"
                            for q in ("daughter of", "son of", "wife of"))]
        self.assertGreaterEqual(len(qualified), 3,
                                "too few middle_name rows exercise the "
                                "relationship-qualifier shape")

    def test_all_three_relationship_qualifiers_are_present(self):
        """The brief names father's- and husband's-name conventions. `S/o`,
        `W/o` and `D/o` are three different renderings and all three must be
        exercised -- `D/o` was dead code until an identity with a father and no
        husband was added, while three places documented it as covered."""
        rendered = " ".join(f"{p.name_a} {p.name_b}" for p in DATASET)
        for marker in ("S/o", "W/o", "D/o"):
            with self.subTest(marker=marker):
                self.assertIn(
                    marker, rendered,
                    f"no row uses {marker}, so the category description "
                    f"advertises a shape the dataset does not contain")

    def test_suffix_rows_really_carry_a_suffix(self):
        """Checked on the raw string: the normaliser consumes the suffix, so
        after normalisation it is (correctly) no longer in the token list."""
        from name_match.normalize import SUFFIXES

        rows = self._rows("suffix")
        self.assertGreaterEqual(len(rows), 10)
        wanted = {value.lower() for value in SUFFIXES}
        for pair in rows:
            with self.subTest(pair=(pair.name_a, pair.name_b)):
                # "Jr." must lose its dot to match the lexicon entry "jr".
                raw = f"{pair.name_a} {pair.name_b}".lower()
                words = {w.strip(".,") for w in raw.split()}
                words |= {w.strip(".,") for w in raw.split()}
                self.assertTrue(
                    words & wanted,
                    f"neither side of {pair.name_a!r} / {pair.name_b!r} "
                    f"carries a generational suffix from {sorted(wanted)}")

    def test_every_roster_identity_contributes_at_least_one_pair(self):
        """The roster tail used to be unreachable: the thinning stride sampled
        from the same start in every category, so the last identity never
        appeared in any comparison."""
        used = {p.person_a for p in DATASET} | {p.person_b for p in DATASET}
        self.assertEqual(set(PEOPLE) - used, set(),
                         f"roster members never used: "
                         f"{sorted(set(PEOPLE) - used)}")

    def test_every_declared_qualifier_string_appears_in_the_dataset(self):
        """Each declared category description is echoed into the shipped CSV
        header, so a qualifier named there must be exercised."""
        import csv

        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "data", "name_pairs.csv")
        if not os.path.exists(path):
            self.skipTest("data/name_pairs.csv not generated yet")
        with open(path, encoding="utf-8", newline="") as handle:
            rendered = " ".join(
                row["name_a"] + " " + row["name_b"]
                for row in csv.DictReader(
                    line for line in handle if not line.startswith("#")))
        for marker in ("Smt.", "Shri", "Mr.", "Km.", "Jr.", "II"):
            with self.subTest(marker=marker):
                self.assertIn(marker, rendered,
                              f"{marker} is named in a category description but "
                              f"appears in no pair")
