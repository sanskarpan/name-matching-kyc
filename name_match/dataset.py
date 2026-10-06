"""Deterministic generator for the labelled name-pair dataset.

Design principle: **labels come from identity, not from string similarity.**

Every :class:`Person` carries a stable ``person_id``. A pair is labelled
``match`` if and only if both sides were rendered from the same ``person_id``.
Nothing in the labelling path ever looks at the resulting strings. This is the
single most important structural property of this dataset: it makes label
leakage structurally impossible, and it is what allows near-identical negatives
("Anil Kumar Sharma" vs "Anil Kumar Saxena", or two different people both
literally named "Sourav Ghosh") to be labelled correctly instead of being
discarded as "too close to hand-label honestly".

Two builders exist:
  * the ``_POSITIVE_BUILDERS`` family -- each renders two *different* surface
    forms of one person. A builder either produces its own category or returns
    ``None``; it never claims a category it did not produce and never silently
    degrades into a different one.
  * :func:`_negative_shape` -- renders two *different* people, and only the
    shapes the pair genuinely supports. A "sibling" row is only emitted for two
    roster members who declare the same father, a "dropped token" row only when
    one side's tokens are a strict subset of the other's, and so on.

Two further invariants are enforced mechanically rather than by review:

*Rationales describe the row.* Every rationale is composed by
:func:`_rationale`, which prefixes the two strings the row actually contains
and refuses to emit a clause twice. A rationale can therefore never quote a name
pair the row does not have, and cannot duplicate its own clause.

*No pair of strings carries two labels.* Rows are deduplicated on the
**unordered** pair of rendered strings, and a positive whose strings collide
with a negative is never emitted -- so the dataset cannot contain "Nisha Yadav
vs Nisha Kumari Yadav" twice with opposite labels.

Generation is fully deterministic: no RNG state, byte-identical output on any
platform.
"""

from __future__ import annotations

import csv
import difflib
import io
import os
from dataclasses import dataclass
from itertools import combinations
from math import gcd
from typing import Callable, Iterable, Sequence

from . import string_algos
from .normalize import normalize
from .phonetics import phonetic_agreement

__all__ = [
    "Person",
    "NamePair",
    "CATEGORIES",
    "PEOPLE",
    "build_dataset",
    "write_csv",
    "read_csv",
    "DEFAULT_DATA_PATH",
]

DEFAULT_DATA_PATH = os.path.join("data", "name_pairs.csv")
SCHEMA_VERSION = "1.2"

#: Target number of positive pairs per category. Keeps the category histogram
#: flat and the whole dataset a sensible size to eyeball by hand.
POSITIVES_PER_CATEGORY = 14

#: How many trivially separable ``unrelated`` negatives to top the dataset up
#: with. Deliberately a small control group: it exists so that a matcher which
#: calls everything a match has somewhere to be caught, not so that it can pad
#: a precision number with pairs no algorithm could ever get wrong.
UNRELATED_FILLER = 14

#: Category definitions, echoed into the CSV header and into reports.
CATEGORIES: dict[str, str] = {
    "initials": "A name rendered with an initial -- usually the given name "
                "(S. Kumar vs Suresh Kumar), sometimes the middle-name block "
                "(Ananya K. Bose vs Ananya Kumar Bose)",
    "surname_first": "Family name written before the given name (Kumar Suresh). "
                     "Some rows also reduce the given name to an initial, which "
                     "makes them this category *and* the initials category",
    "transliteration": "Two valid spellings of the same name (Mohammed/Mohammad)",
    "transliteration_heldout": "Transliteration whose variant is absent from the shipped lexicon",
    "middle_name": "Middle / father / husband name present on one side only, or reordered",
    "honorific": "Title present on one document only (Smt., Shri, Kumari)",
    "suffix": "Generational suffix on the male name (Jr., II). Some rows also "
              "add a title, so they are this category *and* the honorific category",
    "compound_surname": "Joined or hyphenated compound surname (K-Singh vs K Singh)",
    "typo": "One or two character-level transcription / OCR errors",
    "formatting": "Case, spacing and punctuation differ only",
    "hard_negative_sibling": "NEGATIVE: siblings who provably share a household and family name",
    "hard_negative_sibling_reordered": "NEGATIVE: the same siblings, one document surname-first",
    "hard_negative_parent_child": "NEGATIVE: a declared parent/child pair sharing a family name",
    "hard_negative_phonetic": "NEGATIVE: different people whose differing tokens are phonetically identical",
    "hard_negative_common_surname": "NEGATIVE: different people sharing one of the most common surnames",
    "hard_negative_partial_overlap": "NEGATIVE: different people whose names overlap on one given or middle name only",
    "hard_negative_near_duplicate": "NEGATIVE: different people whose names differ by one token or one character",
    "hard_negative_dropped_token": "NEGATIVE: different people, one document omits a token the other carries",
    "hard_negative_identical_name": "NEGATIVE: two different people with byte-identical names (irreducible)",
    "unrelated": "NEGATIVE: different people with no shared token and no phonetic agreement",
}


# --------------------------------------------------------------------------
# Person roster
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Person:
    """A canonical identity. Surface forms are derived from this, never the reverse."""

    person_id: str
    given: str
    middle: tuple[str, ...]
    family: str
    region: str
    gender: str = "m"          # "m" | "f" | "u"
    father: str = ""           # father's given name, for S/o / D/o qualifiers
    spouse: str = ""           # spouse's given name, for W/o qualifiers
    note: str = ""
    role: str = "person"       # "elder" = declared as somebody's parent

    @property
    def full(self) -> str:
        return " ".join(p for p in (self.given, *self.middle, self.family) if p)

    @property
    def initials_form(self) -> str:
        """Given name reduced to initials, one per part ("V." from "Venkat")."""
        return " ".join(f"{part[0]}." for part in self.given.split())

    @property
    def initials_form_tight(self) -> str:
        """Given name reduced to a single bare initial, no dot ("S")."""
        return self.given[0]

    @property
    def title(self) -> str:
        if self.gender == "f":
            return "Smt." if self.region in ("North", "East", "West") else "Km."
        if self.gender == "m":
            return "Shri" if self.region in ("South", "East", "West") else "Mr."
        return "Shri"

    @property
    def relational(self) -> str:
        """The relationship qualifier appropriate to this person's gender."""
        if self.gender == "f":
            return "D/o" if not self.spouse else "W/o"
        return "S/o"


#: Region-clustered roster.
#:
#: Two rules govern the notes, because a note is part of the label:
#:
#: 1. A note never claims that two distinct ``person_id`` values are one
#:    identity. Earlier revisions said "same person as P011" on P012 while the
#:    generator labelled their pair a non-match; that is label noise a human
#:    reviewer cannot see through, and it is the single worst defect this
#:    dataset can have.
#: 2. Where two identities must stay confusable in order to be a valid hard
#:    negative, the note says so outright ("unrelated households that spell the
#:    surname differently") instead of pretending they are one person.
#:
#: The clustering is not decorative: siblings, parent/child pairs, co-parents,
#: spouses and phonetic traps are placed inside clusters on purpose so the
#: negative builders can find them by design rather than by luck.
_ROSTER: tuple[Person, ...] = (
    # -- North: Sharma/Saxena cluster ---------------------------------------
    Person("P001", "Suresh", ("Kumar",), "Sharma", "North", "m",
           father="Ramesh Chandra", note="PAN + Aadhaar; Kumar is a middle name"),
    Person("P002", "Anil", ("Kumar",), "Sharma", "North", "m",
           father="Ramesh Chandra", note="elder brother of P001"),
    Person("P003", "Anil", ("Kumar",), "Saxena", "North", "m",
           father="Ashok Kumar", note="different household from P002, who happens to "
                                     "share the given and middle name; a surname trap"),
    Person("P004", "Ramesh", ("Chandra",), "Sharma", "North", "m",
           father="Mohan Lal", note="father of P001, P002, P005", role="elder"),
    Person("P005", "Sunil", ("Kumar",), "Sharma", "North", "m",
           father="Ramesh Chandra", note="younger brother of P001"),
    Person("P006", "Vijay", (), "Saxena", "North", "m",
           note="DIFFERENT person from P007; the two households pronounce their surnames alike"),
    Person("P007", "Vijay", (), "Saksena", "North", "m",
           note="DIFFERENT person from P006; an unrelated Saksena household that spells "
                "the surname the old way"),
    Person("P008", "Meena", (), "Shah", "West", "f", note="very common surname"),
    Person("P009", "Neena", (), "Shah", "West", "f",
           note="DIFFERENT person from P008; a second Shah household, one letter apart "
                "in the given name"),
    Person("P010", "Rohit", (), "Sarma", "North", "m",
           note="DIFFERENT person from every Sharma in the roster; this household drops "
                "the h"),
    # -- The Mohammed cluster: three unrelated men, three spellings -----------
    Person("P011", "Mohammed", ("Iqbal",), "Khan", "South", "m",
           note="PAN spelling; UNRELATED to P012 and P013, who spell the name differently"),
    Person("P012", "Mohammad", ("Iqbal",), "Khan", "North", "m",
           note="DIFFERENT person from P011/P013: another household in another region "
                "that spells it Mohammad; the spelling gap is not one identity"),
    Person("P013", "Muhammed", ("Iqbal",), "Khan", "East", "m",
           note="DIFFERENT person; the third spelling in this cluster, and a third "
                "household"),
    Person("P014", "Shahnaz", (), "Begum", "South", "f", note="unrelated to P011"),
    Person("P015", "Salma", (), "Khan", "South", "f",
           note="different person, shares the surname with P011-P013"),
    # -- South: given-name inflation and initials ---------------------------
    Person("P016", "Venkat", ("Suresh",), "Rao", "South", "m", father="Subba Rao",
           note="contracted given name on PAN"),
    Person("P017", "Venkatesh", ("Suresh",), "Rao", "North", "m",
           father="Hanumantha Rao",
           note="DIFFERENT person from P016: an unrelated Rao household that writes the "
                "given name in full"),
    Person("P018", "V.", ("Lakshmi",), "Rao", "South", "f", spouse="Venkat Suresh",
           note="given name recorded as a bare initial; a different household from P019"),
    Person("P019", "Lakshmi", ("Devi",), "Rao", "South", "f", spouse="Srinivasan Rao",
           note="DIFFERENT person from P018; same region and surname, different household"),
    Person("P020", "Srinivasan", (), "Krishnan", "South", "m", note="PAN spelling"),
    Person("P021", "Shrinivasan", (), "Krishnan", "South", "m",
           note="DIFFERENT person from P020; a second Krishnan household that writes an "
                "h after the S"),
    Person("P022", "Padmavathi", (), "Krishnan", "South", "f", note="different person, shares surname"),
    Person("P023", "Lakshmi", (), "Iyer", "South", "f", note="same given name as P019, unrelated"),
    Person("P024", "Sourav", (), "Ghosh", "East", "m", note="bank-statement spelling"),
    Person("P025", "Sourav", (), "Ghosh", "East", "m", note="DIFFERENT person, identical name"),
    Person("P026", "Sourav", (), "Das", "East", "m", note="unrelated"),
    Person("P027", "Ananya", ("Kumar",), "Bose", "East", "f", note="uses Kumar as a middle name"),
    Person("P028", "Ananya", ("Kumari",), "Bose", "East", "f", note="Kumari here is a middle name, not a title"),
    # -- West ---------------------------------------------------------------
    Person("P029", "Prakash", (), "Deshmukh", "West", "m", note="PAN"),
    Person("P030", "Prakash", (), "Deshmukh", "West", "m", note="DIFFERENT person, identical name"),
    Person("P031", "Prafull", (), "Deshmukh", "North", "m",
           note="DIFFERENT person from P029/P030; an unrelated Deshmukh household in "
                "another region that spells the given name differently"),
    Person("P032", "Vaishali", (), "Joshi", "West", "f", note="unrelated"),
    Person("P033", "Vaishali", (), "Joisar", "West", "f",
           note="DIFFERENT person from P032; an unrelated household with an old spelling "
                "of the surname"),
    Person("P034", "Gurpreet", ("Singh",), "Sandhu", "North", "m", father="Harjeet Singh"),
    Person("P035", "Gurpreet", ("Singh",), "Sandhu", "North", "m",
           note="DIFFERENT person, identical name; NOT in the Sandhu household of "
                "P034/P036/P037"),
    Person("P036", "Gurjant", ("Singh",), "Sandhu", "North", "m", father="Harjeet Singh",
           note="sibling of P034"),
    Person("P037", "Harpreet", ("Singh",), "Sandhu", "North", "m", father="Harjeet Singh",
           note="sibling of P034"),
    # -- South: compound / joint family names -------------------------------
    Person("P038", "Krishna", (), "Kumar", "South", "m", note="father of P039",
           role="elder"),
    Person("P039", "Krishnan", (), "Kumar", "South", "m", father="Krishna",
           note="son of P038, one-letter given-name difference"),
    Person("P040", "Lakshmi", (), "Kumar", "South", "f", spouse="Krishna",
           note="the other parent of P039; the co-parent of P038"),
    Person("P041", "Deepa", ("Kumar",), "Pillai", "South", "f", note="PAN, with middle name"),
    Person("P042", "Deepa", (), "Kumar", "South", "f", note="different person, same given name"),
    # -- Spelling-drift singleton pairs -------------------------------------
    Person("P043", "Amit", ("Kumar",), "Agrawal", "North", "m",
           father="Ramesh Agrawal", note="spelling-drift surname"),
    Person("P044", "Amit", ("Kumar",), "Agarwal", "North", "m",
           father="Suresh Agarwal",
           note="DIFFERENT person from P043: an unrelated Agarwal household that spells "
                "the surname the other way"),
    Person("P045", "Nisha", ("Kumari",), "Yadav", "North", "f", note="Kumari as a middle name"),
    Person("P046", "Nisha", (), "Yadav", "North", "f",
           note="DIFFERENT person from P045: a second Yadav household, and no middle "
                "name recorded for her"),
    Person("P047", "Rahul", ("Kumar",), "Gupta", "North", "m", note="PAN"),
    Person("P048", "Rahul", (), "Gupta", "North", "f", note="different person, same given name"),
    Person("P049", "Sneha", (), "Bhattacharya", "East", "f", note="long typo-prone surname"),
    Person("P050", "Snehalata", (), "Bhattacharya", "East", "f", note="different person, shared surname"),
    Person("P051", "Joseph", (), "Thomas", "South", "m", note="Christian name, PAN style"),
    Person("P052", "Joseph", (), "Tomas", "South", "m",
           note="DIFFERENT person from P051; an unrelated Thomas family that drops the "
                "silent h"),
    Person("P053", "Farida", (), "Begum", "North", "f", note="unrelated to P014"),
    Person("P054", "Arjun", ("Singh",), "Chauhan", "North", "m", note="utility bill"),
    Person("P055", "Arjun", ("Singh",), "Chowhan", "North", "m",
           note="DIFFERENT person from P054; an unrelated household that spells the "
                "surname Chowhan"),
    Person("P056", "Bhavna", (), "Trivedi", "West", "f", note="PAN"),
    Person("P057", "Bhavana", (), "Trivedi", "East", "f",
           note="DIFFERENT person from P056; a separate Trivedi family in another region, "
                "given name spelled with a long vowel"),
    Person("P058", "Kiran", (), "Pawar", "West", "m", note="PAN"),
    Person("P059", "Kiran", (), "Pawar", "West", "f", note="DIFFERENT person, identical name"),
    Person("P060", "Rekha", (), "Menon", "South", "f", note="unrelated"),
    Person("P061", "Rekha", (), "Menon", "South", "f", note="DIFFERENT person, identical name"),
    Person("P062", "Ganesh", (), "Nadar", "South", "m", note="unrelated"),
    Person("P063", "Harsh", ("Kumar",), "Mehta", "West", "m", note="PAN"),
    Person("P064", "Harsh", ("Kumar",), "Mehta", "West", "m", note="DIFFERENT person, identical name"),
    # -- Additional households, to broaden the negative pool ----------------
    Person("P065", "Kavitha", (), "Reddy", "South", "f", spouse="Ravi Reddy"),
    Person("P066", "Ravi", ("Kumar",), "Reddy", "South", "m", note="husband of P065"),
    Person("P067", "Manjula", (), "Naidu", "South", "f", note="unrelated"),
    Person("P068", "Satish", (), "Naidu", "South", "m", note="unrelated, shared surname"),
    Person("P069", "Priyanka", (), "Chatterjee", "East", "f", note="PAN"),
    Person("P070", "Priyanka", (), "Chowdhury", "East", "f", note="different person, long-surname variant"),
    Person("P071", "Tarun", (), "Chauhan", "North", "m", note="unrelated, shares surname with P054"),
    Person("P072", "Shubha", (), "Mukherjee", "East", "f", note="passport"),
    Person("P073", "Shubha", (), "Mukherji", "East", "f",
           note="DIFFERENT person from P072; an unrelated Mukherji household, surname "
                "spelled differently"),
    Person("P074", "Nitin", (), "Kapoor", "North", "m", note="PAN"),
    Person("P075", "Nita", (), "Kapoor", "North", "f", note="DIFFERENT person, one-char given-name difference"),
    Person("P076", "Balaji", (), "Venkataraman", "South", "m", note="long compound surname"),
    Person("P077", "Hemant", ("Kumar",), "Gupta", "North", "m",
           note="DIFFERENT person from P047; another household that records Gupta"),
    Person("P078", "Sonal", (), "Patil", "West", "f", note="unrelated"),
    Person("P079", "Imran", ("Khan",), "Sheikh", "South", "m", note="compound name+surname"),
    # Her father makes the `D/o` qualifier reachable: `Person.relational`
    # returns "D/o" only for a woman with no husband, and without such a
    # person the branch was dead code while three places documented it as
    # covered. She is a North Indian Muslim name, and `D/o` rather than
    # `W/o` is what a document of hers would actually carry.
    Person("P080", "Zoya", (), "Fatma", "North", "f",
           father="Abdul Rehman", note="unrelated"),
    # -- Joint / hyphenated surnames, so the compound-surname category is real --
    Person("P081", "Simran", (), "Kumar Singh", "North", "f",
           note="joint surname as one token, PAN style"),
    Person("P082", "Simran", (), "Kumar Singh", "North", "f",
           note="DIFFERENT person, identical joint surname"),
    Person("P083", "Harbhajan", (), "Singh Gill", "North", "m",
           note="second joint-surname example, spelled out on the document"),
    Person("P084", "Harbhajan", (), "Singh", "North", "m",
           note="DIFFERENT person from P083; a different household that records only the "
                "first element of the joint surname"),
    Person("P085", "Nandini", (), "Rao Naidu", "South", "f",
           note="third joint-surname household, Telugu style"),
    Person("P086", "Srinivas", (), "Rao Naidu", "South", "m",
           note="DIFFERENT person from P085; a second Rao Naidu household"),
    Person("P087", "Kavya", (), "Menon Iyer", "South", "f",
           note="fourth joint-surname household, Kerala style"),
    Person("P088", "Mohan", (), "Menon Iyer", "South", "m",
           note="DIFFERENT person from P087; a second Menon Iyer household"),
    Person("P089", "Ritika", (), "Singh Bhalla", "North", "f",
           note="fifth joint-surname household"),
    Person("P090", "Sudha", (), "Rao Naidu", "South", "f",
           note="DIFFERENT person from P085/P086; a third Rao Naidu household"),
    Person("P091", "Gopal", (), "Singh Gill", "North", "m",
           note="DIFFERENT person from P083; a second Singh Gill household"),
)

PEOPLE: dict[str, Person] = {p.person_id: p for p in _ROSTER}


# --------------------------------------------------------------------------
# Pair record
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NamePair:
    """One labelled comparison."""

    pair_id: str
    name_a: str
    name_b: str
    label: int
    person_a: str
    person_b: str
    category: str
    difficulty: str
    rationale: str

    @property
    def is_match(self) -> bool:
        return self.label == 1

    def as_row(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "name_a": self.name_a,
            "name_b": self.name_b,
            "label": self.label,
            "person_a": self.person_a,
            "person_b": self.person_b,
            "category": self.category,
            "difficulty": self.difficulty,
            "rationale": self.rationale,
        }


# --------------------------------------------------------------------------
# Rationale composition
# --------------------------------------------------------------------------
#
# A rationale is evidence about *this row*. Two failure modes are worth more
# than the sentence itself:
#
#   * the builder corrupted the pair after the sentence was written, so the
#     rationale quoted a name pair the row does not contain;
#   * the rationale repeated its own clause, which reads as if two
#     independent facts had been checked and neither was.
#
# :func:`_rationale` makes both impossible to express. The name prefix is
# generated from the row's own strings, and a duplicated clause is an assertion
# failure at generation time rather than a row in the CSV.


def _rationale(name_a: str, name_b: str, *clauses: str) -> str:
    """Compose a rationale that provably describes ``name_a`` / ``name_b``."""
    for name in (name_a, name_b):
        if "'" in name:
            raise ValueError(f"name containing a quote breaks the rationale format: {name!r}")
    kept = [clause for clause in clauses if clause]
    if not kept:
        raise ValueError("a rationale needs at least one clause")
    for clause in kept:
        if "'" in clause:
            raise ValueError(f"clause would break the name-quote format: {clause!r}")
    if len(set(kept)) != len(kept):
        raise ValueError(f"duplicated rationale clause: {kept}")
    return f"{name_a!r} vs {name_b!r}: " + "; ".join(kept)


def _char_edits(before: str, after: str) -> list[tuple[str, str, str]]:
    """``(tag, removed, inserted)`` for every character run that differs."""
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    edits = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            edits.append((tag, before[i1:i2], after[j1:j2]))
    return edits


def _edit_phrase(before: str, after: str) -> str:
    """A true, specific description of how ``after`` differs from ``before``.

    Computed from the strings rather than asserted by the caller, so a builder
    cannot claim a duplicated character where the real edit was a deleted space.
    """
    phrases = []
    for tag, removed, inserted in _char_edits(before, after):
        if (tag == "replace" and len(removed) == len(inserted) == 2
                and removed[0] == inserted[1] and removed[1] == inserted[0]):
            phrases.append(f"two characters transposed ({removed} -> {inserted})")
        elif tag == "replace" and len(removed) == len(inserted) == 1:
            phrases.append(f"one character substituted ({removed} -> {inserted})")
        elif tag == "delete":
            noun = "character" if len(removed) == 1 else "characters"
            phrases.append(f"{len(removed)} {noun} deleted ({removed})")
        elif tag == "insert":
            noun = "character" if len(inserted) == 1 else "characters"
            phrases.append(f"{len(inserted)} {noun} inserted ({inserted})")
        else:
            phrases.append(f"{len(removed)} characters replaced by {len(inserted)} "
                           f"({removed} -> {inserted})")
    return "; ".join(phrases) if phrases else "text unchanged"


def _dup_or_insert(before: str, after: str) -> str:
    """Distinguish a duplicated character from a genuinely new one."""
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "insert" or len(after[j1:j2]) != 1:
            continue
        char = after[j1]
        neighbours = {before[i1 - 1] if i1 else None, before[i1] if i1 < len(before) else None}
        if char in neighbours:
            return f"one character duplicated ({char})"
        return f"one character inserted ({char})"
    return _edit_phrase(before, after)


# --------------------------------------------------------------------------
# Surface-rendering helpers
# --------------------------------------------------------------------------


def _case(text: str, style: str) -> str:
    if style == "upper":
        return text.upper()
    if style == "title":
        return " ".join(word.capitalize() for word in text.split())
    if style == "lower":
        return text.lower()
    return text


def _typo(text: str, kind: str) -> str:
    """Inject one realistic transcription error into a single name token.

    Callers must pass a token, never a multi-word string: an edit chosen by
    index inside "Kumar Singh" lands on the joining space, which produces a
    joined surname masquerading as a typo.
    """
    if " " in text:
        raise ValueError(f"_typo expects a single token, got {text!r}")
    if len(text) < 3:
        raise ValueError(f"token too short to edit in place: {text!r}")
    mid = len(text) // 2
    if kind == "drop":
        return text[:mid] + text[mid + 1:]
    if kind == "transpose":
        return text[:mid] + text[mid + 1] + text[mid] + text[mid + 2:]
    if kind == "double":
        return text[:mid] + text[mid] + text[mid:]
    if kind == "substitute":
        replacement = "e" if text[mid].lower() == "i" else "i"
        return text[:mid] + replacement + text[mid + 1:]
    raise ValueError(f"unknown typo kind {kind!r}")


def _can_typo(text: str) -> bool:
    return " " not in text and len(text) >= 3


def _join(tokens: Sequence[str]) -> str:
    return " ".join(t for t in tokens if t).strip()


# --------------------------------------------------------------------------
# Spelling-variant tables
# --------------------------------------------------------------------------
#
# Both tables are keyed by the **lowercase** roster spelling and are always
# looked up with ``.lower()``. They used to be keyed lowercase and looked up
# with the title-case ``p.given`` / ``p.family``, so every lookup missed and
# the transliteration builders silently degraded into a one-character
# substitution -- which is a typo, not a transliteration.

#: Given-name variants keyed by the lowercase roster spelling.
_GIVEN_VARIANTS: dict[str, tuple[str, ...]] = {
    "mohammed": ("Mohammad", "Muhammed"),
    "mohammad": ("Mohammed", "Muhammad"),
    "muhammed": ("Mohammed", "Muhammad"),
    "venkat": ("Venkatesh",),
    "venkatesh": ("Venkat",),
    "srinivasan": ("Shrinivasan", "Sreenivasan"),
    "shrinivasan": ("Srinivasan", "Sreenivasan"),
    "sourav": ("Sourab", "Saurabh"),
    "prakash": ("Praful",),
    "prafull": ("Praful", "Prakash"),
    "amit": ("Ameet",),
    "nisha": ("Nishaa",),
    "sneha": ("Snehaa",),
    "snehalata": ("Snehalatha",),
    "joseph": ("Josef",),
    "vaishali": ("Vaisali",),
    "krishnan": ("Krishana",),
    "krishna": ("Krishnappa",),
    "lakshmi": ("Lakshmee", "Laxmi"),
    "padmavathi": ("Padmavati",),
    "ganesh": ("Ganeesh",),
    "harsh": ("Harsha",),
    "bhavna": ("Bhavana",),
    "bhavana": ("Bhavna",),
    "suresh": ("Surish",),
    "ramesh": ("Raamesh",),
    "ananya": ("Ananyaa",),
    "kavitha": ("Kavita",),
    "priyanka": ("Priyanaka",),
    "shubha": ("Shubhaa",),
    "manjula": ("Manjulaa",),
    "satish": ("Sateesh",),
    "rekha": ("Rekhaa",),
    "sonal": ("Sonali",),
    "imran": ("Imraan",),
    "nitin": ("Niteen",),
    "meena": ("Meenaa",),
    "tarun": ("Taron",),
    "nishaa": ("Nisha",),
}

#: Surname variants keyed by the lowercase roster spelling.
_FAMILY_VARIANTS: dict[str, tuple[str, ...]] = {
    "sharma": ("Sarma",),
    "sarma": ("Sharma",),
    "saxena": ("Saksena",),
    "saksena": ("Saxena",),
    "krishnan": ("Krishana",),
    "agrawal": ("Agarwal",),
    "agarwal": ("Agrawal",),
    "bhattacharya": ("Bhattacherya", "Bhattacharjee"),
    "bhattacherya": ("Bhattacharya",),
    "chauhan": ("Chowhan",),
    "chowhan": ("Chauhan",),
    "trivedi": ("Trivredi", "Trivedy"),
    "joshi": ("Joisar",),
    "joisar": ("Joshi",),
    "deshmukh": ("Deshmokh",),
    "ghosh": ("Ghose",),
    "chatterjee": ("Chowdhury", "Chattopadhyay"),
    "chowdhury": ("Chatterjee",),
    "mukherjee": ("Mukherji", "Mookerjee"),
    "mukherji": ("Mukherjee",),
    "nadar": ("Nadkar",),
    "mehta": ("Mehtaa",),
    "kapoor": ("Kapur",),
    "reddy": ("Reddi",),
    "naidu": ("Naidoo",),
    "yadav": ("Yadhav",),
    "thomas": ("Tomas",),
    "tomas": ("Thomas",),
    "pawar": ("Pavar",),
    "pillai": ("Pillay",),
    "iyer": ("Iyyer",),
    "iyyer": ("Iyer",),
    "kumar": ("Kummar",),
    "venkataraman": ("Venkataram",),
    "sheikh": ("Shaikh",),
}


def _given_variant(person: Person) -> str:
    """First given-name variant that is a genuinely different spelling."""
    for variant in _GIVEN_VARIANTS.get(person.given.lower(), ()):
        if variant.lower() != person.given.lower():
            return variant
    return ""


def _family_variant(person: Person) -> str:
    """First surname variant that is a genuinely different spelling."""
    for variant in _FAMILY_VARIANTS.get(person.family.lower(), ()):
        if variant.lower() != person.family.lower():
            return variant
    return ""


#: Every spelling this generator is willing to call a transliteration. Used by
#: the test suite to prove that no `transliteration` row is really a typo.
_ALL_VARIANTS: frozenset[str] = frozenset(
    v.lower() for v in
    [s for group in _GIVEN_VARIANTS.values() for s in group]
    + [s for group in _FAMILY_VARIANTS.values() for s in group])


# --------------------------------------------------------------------------
# Positive builders
# --------------------------------------------------------------------------

#: A builder returns ``(name_a, name_b, category_achieved, difficulty,
#: rationale)``, or ``None`` when this particular identity cannot support the
#: category at all. It never returns a row tagged with a category it did not
#: produce, and it never reuses another category's row as a stand-in.
_PositiveBuilder = Callable[[Person], "tuple[str, str, str, str, str] | None"]


def _missing_middle_clause(person: Person, rendered: str) -> str:
    """True statement about which middle name the rendered side omits."""
    missing = [m for m in person.middle if m.lower() not in rendered.lower()]
    if not missing:
        return ""
    return f"the middle name {' and '.join(missing)} appears on one document only"


def _b_initials_dotted(p: Person):
    # Only the *given* name is abbreviated: dropping the family name as well
    # would be a truncation, not an initials rendering, and the category claims
    # "given name abbreviated to an initial".
    side = _join([p.initials_form, " ".join(p.middle), p.family])
    return (side, p.full, "initials", "easy",
            _rationale(side, p.full,
                       f"the given name is reduced to its dotted initial ({p.given} -> "
                       f"{p.initials_form})",
                       _missing_middle_clause(p, side)))


def _b_initials_bare(p: Person):
    side = _join([p.initials_form_tight, " ".join(p.middle), p.family])
    return (side, p.full, "initials", "easy",
            _rationale(side, p.full,
                       f"the given name is reduced to a bare initial with no dot "
                       f"({p.given} -> {p.initials_form_tight})",
                       _missing_middle_clause(p, side)))


def _b_initials_middle(p: Person):
    if not p.middle:
        return None
    side = f"{p.given} {' '.join(m[0] + '.' for m in p.middle)} {p.family}"
    return (side, p.full, "initials", "medium",
            _rationale(side, p.full,
                       f"the given name is in full but the middle name is reduced to "
                       f"initials ({' '.join(p.middle)} -> "
                       f"{' '.join(m[0] + '.' for m in p.middle)})"))


def _b_surname_first(p: Person):
    side = f"{p.family} {p.given}"
    return (side, p.full, "surname_first", "easy",
            _rationale(side, p.full,
                       f"the family name is written before the given name, as in state "
                       f"voter-ID and driving-licence layouts ({p.full} -> {side})"))


def _b_surname_first_full(p: Person):
    side = f"{p.family} {_join(p.middle)} {p.given}"
    return (side, p.full, "surname_first", "medium",
            _rationale(side, p.full,
                       "the family name and the whole middle-name block are both written "
                       "before the given name",
                       _missing_middle_clause(p, side)))


def _b_surname_first_initial(p: Person):
    side = f"{p.family} {p.initials_form_tight}"
    return (side, p.full, "surname_first", "hard",
            _rationale(side, p.full,
                       "the family name is written first and the given name is reduced "
                       "to a bare initial at the same time",
                       _missing_middle_clause(p, side)))


def _b_translit_given(p: Person):
    variant = _given_variant(p)
    if not variant:
        return None
    side = _join([variant, " ".join(p.middle), p.family])
    return (side, p.full, "transliteration", "medium",
            _rationale(side, p.full,
                       f"the given name is spelled the other valid way "
                       f"({p.given} -> {variant})",
                       _missing_middle_clause(p, side)))


def _b_translit_family(p: Person):
    variant = _family_variant(p)
    if not variant:
        return None
    side = _join([p.given, " ".join(p.middle), variant])
    return (side, p.full, "transliteration", "medium",
            _rationale(side, p.full,
                       f"the surname is spelled the other valid way "
                       f"({p.family} -> {variant})",
                       _missing_middle_clause(p, side)))


def _b_translit_both(p: Person):
    given = _given_variant(p)
    family = _family_variant(p)
    if not given and not family:
        return None
    clauses = []
    if given:
        clauses.append(f"the given name is spelled the other valid way ({p.given} -> {given})")
    if family:
        clauses.append(f"the surname is spelled the other valid way ({p.family} -> {family})")
    side = _join([given or p.given, family or p.family])
    clauses.append(_missing_middle_clause(p, side))
    return (side, p.full, "transliteration", "hard",
            _rationale(side, p.full, *clauses))


def _b_drop_middle(p: Person):
    if not p.middle:
        return None
    side = f"{p.given} {p.family}"
    return (side, p.full, "middle_name", "easy",
            _rationale(side, p.full,
                       f"the middle name {' and '.join(p.middle)} appears on one "
                       f"document only"))


def _b_middle_reordered(p: Person):
    if not p.middle:
        return None
    side = f"{p.family} {' '.join(p.middle)} {p.given}"
    return (side, p.full, "middle_name", "hard",
            _rationale(side, p.full,
                       "the family name and the middle name are both written before the "
                       "given name"))


def _b_middle_partial(p: Person):
    if len(p.middle) < 2:
        return None
    side = f"{p.given} {p.middle[0]} {p.family}"
    return (side, p.full, "middle_name", "medium",
            _rationale(side, p.full,
                       f"only the first of two middle names is written "
                       f"({p.middle[0]} appears, {p.middle[1]} does not)"))


def _b_honorific(p: Person):
    side = f"{p.title} {p.full}"
    return (side, p.full, "honorific", "easy",
            _rationale(side, p.full,
                       f"the title {p.title.strip()} is printed on one document only"))


def _b_honorific_and_middle(p: Person):
    if p.father:
        side = f"{p.title} {p.given} {p.family} {p.relational} {p.father}"
        clauses = (f"the title {p.title.strip()} is printed on one document only",
                   f"the other document qualifies the name with "
                   f"{p.relational} {p.father}",
                   _missing_middle_clause(p, side))
    elif p.middle:
        side = f"{p.title} {p.given} {p.family}"
        clauses = (f"the title {p.title.strip()} is printed on one document only",
                   f"the middle name {' and '.join(p.middle)} appears on one document only")
    elif p.spouse:
        side = f"{p.title} {p.given} {p.family} W/o {p.spouse}"
        clauses = (f"the title {p.title.strip()} is printed on one document only",
                   f"the other document qualifies the name with W/o {p.spouse}")
    else:
        return None
    return (side, p.full, "honorific", "hard", _rationale(side, p.full, *clauses))


def _b_suffix(p: Person):
    if p.gender == "f" or p.role == "elder":
        # P004 and P038 are defined as somebody's father, so a generational
        # suffix on them would claim a namesake ancestor the roster never
        # mentions.
        return None
    side = f"{p.full} Jr."
    return (side, p.full, "suffix", "easy",
            _rationale(side, p.full,
                       "the generational suffix Jr. is written on one document only"))


def _b_suffix_and_honorific(p: Person):
    if p.gender == "f":
        return None
    if p.role == "elder":
        # P004 and P038 are defined as somebody's father, so a "II" would claim a
        # namesake ancestor the roster never mentions.
        return None
    side, other = f"{p.full} II", f"{p.title} {p.full}"
    return (side, other, "suffix", "medium",
            _rationale(side, other,
                       "the generational suffix II is written on one document only",
                       f"the other document carries the title {p.title.strip()} instead"))


def _absent_clause(other: str, related: str) -> str:
    """Claim that ``related`` is a new name element only if that is true."""
    tokens = list(normalize(related).tokens)
    if not tokens:
        return ""
    if tokens[0] in _tokens(other):
        return ""
    return ("the name of the related person is not among the name tokens of the "
            "other document")


def _b_relational_father(p: Person):
    if not p.father:
        return None
    side = f"{p.given} {p.family} {p.relational} {p.father}"
    return (side, p.full, "middle_name", "hard",
            _rationale(side, p.full,
                       f"the first document qualifies the name with {p.relational} "
                       f"{p.father}",
                       _absent_clause(p.full, p.father),
                       _missing_middle_clause(p, side)))


def _b_relational_spouse(p: Person):
    if not p.spouse:
        return None
    side = f"{p.full} W/o {p.spouse}"
    return (side, p.full, "middle_name", "hard",
            _rationale(side, p.full,
                       f"the first document qualifies the name with W/o {p.spouse}",
                       _absent_clause(p.full, p.spouse),
                       _missing_middle_clause(p, side)))


def _b_compound_joined(p: Person):
    if " " not in p.family:
        return None
    side = f"{p.given} {p.family.replace(' ', '')}"
    return (side, p.full, "compound_surname", "medium",
            _rationale(side, p.full,
                       f"the two-word surname is written joined ({p.family} -> "
                       f"{p.family.replace(' ', '')})",
                       _missing_middle_clause(p, side)))


def _b_compound_hyphen(p: Person):
    if " " not in p.family:
        return None
    side = f"{p.given} {'-'.join(p.family.split())}"
    return (side, p.full, "compound_surname", "medium",
            _rationale(side, p.full,
                       f"the two-word surname is hyphenated ({p.family} -> "
                       f"{'-'.join(p.family.split())})",
                       _missing_middle_clause(p, side)))


def _b_typo_given(p: Person):
    if not _can_typo(p.given):
        return None
    variant = _typo(p.given, "transpose")
    return (variant, p.full, "typo", "medium",
            _rationale(variant, p.full,
                       f"the given name is mistyped in the first document: "
                       f"{_edit_phrase(p.given, variant)} ({p.given} -> {variant})"))


def _b_typo_family(p: Person):
    # Edit the last element of the surname, never the join between the two
    # elements of a compound surname: deleting that space is a compound-surname
    # difference, not a transcription error.
    parts = p.family.split()
    if not _can_typo(parts[-1]):
        return None
    variant = _typo(parts[-1], "drop")
    side = _join([p.given, " ".join(p.middle), " ".join(parts[:-1] + [variant])])
    return (side, p.full, "typo", "medium",
            _rationale(side, p.full,
                       f"the surname is mistyped in the first document: "
                       f"{_edit_phrase(parts[-1], variant)} ({parts[-1]} -> {variant})"))


def _b_typo_double(p: Person):
    parts = p.family.split()
    if not _can_typo(parts[-1]):
        return None
    variant = _typo(parts[-1], "double")
    side = _join([p.given, " ".join(p.middle), " ".join(parts[:-1] + [variant])])
    return (side, p.full, "typo", "hard",
            _rationale(side, p.full,
                       f"the surname is repeated in the first document, as an OCR repeat "
                       f"does: {_dup_or_insert(parts[-1], variant)} "
                       f"({parts[-1]} -> {variant})"))


def _b_formatting(p: Person):
    side = _case(p.full, "upper")
    return (side, p.full, "formatting", "easy",
            _rationale(side, p.full,
                       "the same name is written in capitals; the tokens are identical "
                       "and only the case differs"))


def _b_combined_hard(p: Person):
    """Three simultaneous perturbations -- the realistic worst case."""
    side = _join([p.title, p.initials_form_tight, p.family])
    clauses = [f"the title {p.title.strip()} is printed on one document only",
               f"the given name is reduced to a bare initial ({p.given} -> "
               f"{p.initials_form_tight})",
               _missing_middle_clause(p, side)]
    return (side, p.full, "initials", "hard", _rationale(side, p.full, *clauses))


def _b_combined_translit(p: Person):
    given = _given_variant(p)
    family = _family_variant(p)
    if not given and not family:
        return None
    clauses = []
    if given:
        clauses.append(f"the given name is spelled the other valid way ({p.given} -> {given})")
    if family:
        clauses.append(f"the surname is spelled the other valid way ({p.family} -> {family})")
    side = _join([given or p.given, " ".join(p.middle), family or p.family])
    clauses.append(_missing_middle_clause(p, side))
    return (side, p.full, "transliteration", "hard", _rationale(side, p.full, *clauses))


#: Every positive builder, grouped by the category it is *designed* to produce.
#: The dataset planner walks this structure so category coverage stays even; it
#: then verifies that the builder really produced that category for this
#: particular person and falls through to the next candidate if not.
def _stride_for(count: int) -> int:
    """Smallest stride greater than one that is coprime with ``count``.

    Walking builders by a stride that shares a factor with the builder count
    visits only a fraction of them, which is how a category ends up testing one
    shape instead of several while still looking evenly covered.
    """
    for candidate in range(2, count + 1):
        if gcd(candidate, count) == 1:
            return candidate
    return 1


_POSITIVE_BUILDERS: dict[str, tuple[_PositiveBuilder, ...]] = {
    "initials": (_b_initials_dotted, _b_initials_bare, _b_initials_middle, _b_combined_hard),
    "surname_first": (_b_surname_first, _b_surname_first_full, _b_surname_first_initial),
    "transliteration": (_b_translit_given, _b_translit_family, _b_translit_both,
                        _b_combined_translit),
    "middle_name": (_b_drop_middle, _b_middle_reordered, _b_middle_partial,
                    _b_relational_father, _b_relational_spouse),
    "honorific": (_b_honorific, _b_honorific_and_middle),
    "suffix": (_b_suffix, _b_suffix_and_honorific),
    "compound_surname": (_b_compound_joined, _b_compound_hyphen),
    "typo": (_b_typo_given, _b_typo_family, _b_typo_double),
    "formatting": (_b_formatting,),
}

#: Category rotation order. ``unrelated`` is absent because it is negative-only.
_POSITIVE_CATEGORY_ORDER: tuple[str, ...] = (
    "initials", "surname_first", "transliteration", "middle_name", "honorific",
    "suffix", "compound_surname", "typo", "formatting",
)


# --------------------------------------------------------------------------
# Held-out transliteration positives
# --------------------------------------------------------------------------
#
# Held-out transliteration, split by whether the *normaliser* resolves it.
#
# Two distinct claims are being tested and they must not be conflated:
#
# :data:`_HELD_OUT_TRANSLIT_UNRESOLVED`
#     The variant introduces at least one spelling the curated lexicon has
#     never seen, *and* normalisation does not collapse the pair. Only the
#     string-similarity machinery can bridge these, so they measure
#     generalisation.
# :data:`_HELD_OUT_TRANSLIT_FOLDED`
#     Same, except normalisation *does* collapse the pair -- but only because the
#     generic character fold runs first (a geminate consonant, a doubled vowel).
#     The variant as written is not a lexicon entry, so the pair cannot be
#     resolved by a lookup. For some of these the folded form then lands on a
#     lexicon class, so it is the fold-plus-second-lookup that resolves them,
#     never the lexicon alone.
#
# "At least one changed spelling" is deliberate. A variant may also carry a
# surname the lexicon knows -- "Priyanaka Chaterjee" keeps a listed spelling of
# Chaterjee -- and that pair is still a genuine generalisation test through the
# given name. Demanding that every changed token be unknown would only push the
# dataset toward invented surnames nobody writes.
#
# Collapsing these into one "held out" bucket would quietly claim credit for
# the normaliser on the second group, which is exactly the kind of flattering
# mislabelling the dataset exists to avoid.
#
# Both tables are checked against the real normaliser by
# ``test_held_out_translit_table_matches_what_normalisation_actually_does``:
# the split is a claim about the code, and an earlier version of this file
# asserted it in a comment while two of its entries were resolved by lexicon
# entries rather than by the fold -- which made "held out" false for them.
#
# Every ``surface_a`` here is the identity's canonical rendering, and every
# ``surface_b`` is a variant of that same identity -- never a variant belonging
# to a *different* roster member, which is how an earlier revision ended up
# asking for Bhavna Trivedi / Bhavana Trivedi to be simultaneously a match and a
# non-match.

_HELD_OUT_TRANSLIT_UNRESOLVED: tuple[tuple[str, str, str, str], ...] = (
    ("P051", "Joseph Thomas", "Josef Thomas",
     "Joseph against Josef: a silent short vowel is dropped; absent from the lexicon "
     "and not reachable by the generic fold"),
    ("P033", "Vaishali Joisar", "Vaisali Josar",
     "Vaishali against Vaisali plus a surname change: two simultaneous vowel edits"),
    ("P024", "Sourav Ghosh", "Sourab Ghosh",
     "Sourav against Sourab: a single vowel substitution in a Bengali given name"),
    ("P049", "Sneha Bhattacharya", "Sneha Bhattacheria",
     "a vowel change deep inside a long Bengali surname"),
    ("P054", "Arjun Singh Chauhan", "Arjaan Singh Chawhan",
     "a doubled vowel plus a w/v change across two tokens"),
    ("P020", "Srinivasan Krishnan", "Srinivasan Krishnaen",
     "a vowel insertion inside a long surname"),
    ("P069", "Priyanka Chatterjee", "Priyanaka Chaterjee",
     "two single-character errors across two tokens"),
    ("P076", "Balaji Venkataraman", "Balaji Venkatraman",
     "a vowel omission inside a long compound surname"),
    ("P034", "Gurpreet Singh Sandhu", "Gurpreet Sngh Sandhu",
     "a vowel deletion inside the middle name"),
    ("P056", "Bhavna Trivedi", "Bavna Trivedi",
     "a vowel omission in the given name; the other Bhavana Trivedi in the roster is a "
     "different person, so this variant is deliberately not that spelling"),
    ("P012", "Mohammad Iqbal Khan", "Mohamad Iqbal Khan",
     "unstressed vowel reduction; absent from the lexicon and not reachable by the "
     "generic fold"),
    ("P022", "Padmavathi Krishnan", "Padmavathi Krisnan",
     "a vowel omission inside a long surname; absent from the lexicon and not reachable "
     "by the generic fold"),
    ("P034", "Gurpreet Singh Sandhu", "Gurpreet Sing Sandhu",
     "a final cluster reduction; the trailing letter is dropped rather than replaced, so "
     "no lexicon entry and no generic fold closes the gap"),
)

_HELD_OUT_TRANSLIT_FOLDED: tuple[tuple[str, str, str, str], ...] = (
    ("P016", "Venkat Suresh Rao", "Venkatt Suresh Rao",
     "a geminate consonant difference; the variant is not a lexicon entry, and is "
     "resolved only because the repeated-letter fold produces the canonical spelling, "
     "which the lexicon then maps on a second lookup"),
    ("P045", "Nisha Kumari Yadav", "Nisha Kumaari Yadav",
     "a geminate vowel inside a middle name, resolved by the repeated-letter fold"),
    ("P063", "Harsh Kumar Mehta", "Harsh Kumaar Mehta",
     "a geminate vowel inside a middle name, resolved by the repeated-letter fold"),
    ("P062", "Ganesh Nadar", "Ganeesh Nadar",
     "a long-vowel change, resolved by the repeated-letter fold"),
    ("P059", "Kiran Pawar", "Kiran Paawar",
     "a geminate vowel in a common Marathi surname, resolved by the repeated-letter fold"),
)


# --------------------------------------------------------------------------
# Negative shapes
# --------------------------------------------------------------------------
#
# Every negative is described by a *shape* -- a property of the two strings that
# can be checked -- and the shape decides both the category and the wording of
# the rationale. Nothing is labelled "sibling" because two people happen to
# share a family name, and nothing is labelled "dropped token" when one side is
# simply a two-token name that happens to be short.

_IDENTICAL_CLAUSE = ("two different people whose names are character-for-character "
                     "identical, so no name-only matcher can separate this pair")


def _tokens(name: str) -> list[str]:
    """Normalised tokens, i.e. what a matcher would actually compare."""
    return list(normalize(name).tokens)


# The dataset's near-duplicate detection needs edit distance, and it needs the
# *same* implementation the matchers use. This used to be a private second copy
# of Levenshtein, which meant the headline "implemented Levenshtein from
# scratch" contributed nothing to the pipeline that produced the dataset, and a
# divergence between the two copies would have silently changed which pairs
# counted as near-duplicates. It now calls the shared primitive.
_edit_distance = string_algos.levenshtein


def _same_household(a: Person, b: Person) -> bool:
    """Two roster members who provably share a household and a family name.

    Derived from the roster's own ``father`` field rather than hand-listed, so a
    "siblings" rationale cannot end up attached to two people with no shared
    household -- which is exactly what the old ``_n_sibling`` did when it was
    applied to arbitrary cross-person pairs.
    """
    return bool(a.father) and a.father == b.father and a.family == b.family


def _is_parent(parent: Person, child: Person) -> bool:
    """True when the roster says ``parent`` is ``child``'s father."""
    if not child.father or parent.family != child.family:
        return False
    parent_block = _join([parent.given, " ".join(parent.middle)])
    return child.father in (parent.given, parent_block)


def _phonetic_trap(tokens_a: Sequence[str], tokens_b: Sequence[str]):
    """The strongest non-identical token pair that sounds the same."""
    best = None
    for token_a in tokens_a:
        for token_b in tokens_b:
            if token_a == token_b:
                continue
            score = phonetic_agreement(token_a, token_b)
            if score >= 0.8 and (best is None or score > best[0]):
                best = (score, token_a, token_b)
    return best


def _differing_position(tokens_a: Sequence[str], tokens_b: Sequence[str]):
    """The single (before, after) token pair at the one position that differs."""
    if len(tokens_a) != len(tokens_b) or len(tokens_a) < 2:
        return None
    diffs = [(a, b) for a, b in zip(tokens_a, tokens_b) if a != b]
    return diffs[0] if len(diffs) == 1 else None


def _shared_family_token(a: Person, b: Person) -> list[str]:
    """Tokens that are a family name on one side and appear in the other name.

    Not the same as "a shared token": "Suresh Kumar Sharma" and
    "Anil Kumar Saxena" share Kumar, but Kumar is a middle name on both sides
    and the surnames are different, so calling that a shared-surname pair
    would misdescribe the row.
    """
    tokens_a, tokens_b = set(_tokens(a.full)), set(_tokens(b.full))
    family_a, family_b = set(_tokens(a.family)), set(_tokens(b.family))
    return sorted((family_a & tokens_b) | (family_b & tokens_a))


def _negative_shape(a: Person, b: Person) -> list[tuple[str, str, str, list[str]]]:
    """Every hard-negative shape this pair genuinely supports.

    Ordered from most to least specific, so a pair that is both a household
    relation and a one-character trap is filed under the structure a reader
    would name first. Returns ``[]`` when the pair is not hard at all, which is
    what stops the sweep from padding the dataset with filler.
    """
    name_a, name_b = a.full, b.full
    tokens_a, tokens_b = _tokens(name_a), _tokens(name_b)
    shared = set(tokens_a) & set(tokens_b)

    # Irreducible: two identities, one string.
    if name_a == name_b:
        return [("hard_negative_identical_name", name_a, name_b, [_IDENTICAL_CLAUSE])]

    household = _same_household(a, b)
    if household:
        fact = (f"both are recorded as children of {a.father} in the {a.family} household, "
                f"so they are siblings and therefore two different people")
        return [
            ("hard_negative_sibling", name_a, name_b, [fact]),
            ("hard_negative_sibling_reordered", f"{a.given} {a.family}",
             f"{b.family} {b.given}",
             [fact, "one document writes the family name first, so the two strings even "
                    "disagree about which half of the name is which"]),
        ]

    if _is_parent(a, b):
        return [("hard_negative_parent_child", name_a, name_b,
                 [f"the roster records {name_a} as the father of {name_b}, so the shared "
                  f"family name and the near-identical given name are a household fact, "
                  f"not one identity"])]
    if _is_parent(b, a):
        return [("hard_negative_parent_child", name_b, name_a,
                 [f"the roster records {name_b} as the father of {name_a}, so the shared "
                  f"family name and the near-identical given name are a household fact, "
                  f"not one identity"])]

    # Dropped token: one document's tokens are a strict subset of the other's.
    if len(tokens_a) > len(tokens_b) and not [t for t in tokens_b if t not in tokens_a]:
        missing = sorted(set(tokens_a) - set(tokens_b))
        return [("hard_negative_dropped_token", name_a, name_b,
                 [f"every token of the second name also appears in the first; the only "
                  f"difference is that the second document omits {', '.join(missing)}"])]
    if len(tokens_b) > len(tokens_a) and not [t for t in tokens_a if t not in tokens_b]:
        missing = sorted(set(tokens_b) - set(tokens_a))
        return [("hard_negative_dropped_token", name_a, name_b,
                 [f"every token of the first name also appears in the second; the only "
                  f"difference is that the first document omits {', '.join(missing)}"])]

    # One token apart.
    differing = _differing_position(tokens_a, tokens_b)
    if differing is not None:
        token_a, token_b = differing
        if _edit_distance(token_a, token_b) <= 1:
            clause = (f"the two names differ in exactly one token and that token is a "
                      f"single character edit apart ({token_a} against {token_b})")
        elif phonetic_agreement(token_a, token_b) >= 0.8:
            clause = (f"the two names differ in exactly one token, and {token_a} and "
                      f"{token_b} are the same word written two ways")
        else:
            clause = (f"the two names differ in exactly one token: {token_a} against "
                      f"{token_b}")
        return [("hard_negative_near_duplicate", name_a, name_b, [clause])]

    if shared:
        families = _shared_family_token(a, b)
        if families:
            return [("hard_negative_common_surname", name_a, name_b,
                     [f"both records carry {', '.join(families)} as an element of a "
                      f"family name, from different households"])]
        return [("hard_negative_partial_overlap", name_a, name_b,
                 [f"the only name element the two records share is "
                  f"{', '.join(sorted(shared))}, which is a given or middle name on one "
                  f"side of the pair; the given name and the family name both differ"])]

    trap = _phonetic_trap(tokens_a, tokens_b)
    if trap is not None:
        score, token_a, token_b = trap
        encoders = ("both Soundex and Metaphone" if score >= 1.0
                    else "one of the two phonetic encoders")
        return [("hard_negative_phonetic", name_a, name_b,
                 [f"no token is shared, but {token_a} and {token_b} collide on "
                  f"{encoders}, so the pair is indistinguishable to a phonetic matcher"])]

    return []


#: Hand-written identity facts, appended to whichever shape a pair supports.
#: These are the statements only the roster can make: who is related to whom, and
#: why two confusable strings are two people rather than one. They must be
#: true of the pair and must not repeat the shape's own clause.
_IDENTITY_FACTS: tuple[tuple[str, str, str], ...] = (
    ("P002", "P003",
     "the two share a given and middle name but not a father, so they are two households "
     "that happened to name a son the same way"),
    ("P006", "P007",
     "two unrelated households, one writing the surname Saxena and the other Saksena"),
    ("P008", "P009",
     "two different women recorded under the same common surname, one letter apart in the "
     "given name"),
    ("P010", "P001",
     "the Sarma household is not the Sharma household, although a matcher cannot hear the "
     "difference"),
    ("P011", "P012",
     "one of three unrelated men in this roster who spell the same name differently, so "
     "the spelling gap is not evidence of one identity"),
    ("P011", "P013",
     "one of three unrelated men in this roster who spell the same name differently, so "
     "the spelling gap is not evidence of one identity"),
    ("P012", "P013",
     "one of three unrelated men in this roster who spell the same name differently, so "
     "the spelling gap is not evidence of one identity"),
    ("P014", "P053",
     "two unrelated women who both record Begum as a family name"),
    ("P016", "P017",
     "different households and different regions; one writes the given name contracted "
     "and the other in full"),
    ("P018", "P019",
     "two different Rao households; the given name is an initial on one document only"),
    ("P020", "P022",
     "two different households in the same region, both recorded under Krishnan"),
    ("P024", "P025",
     "two different men whose source documents carry exactly the same name"),
    ("P027", "P028",
     "two different women; Kumar and Kumari sit in the same middle-name slot on each"),
    ("P029", "P030",
     "two different men whose source documents carry exactly the same name"),
    ("P029", "P031",
     "two different Deshmukh households, one of them in another region, spelling the "
     "given name differently"),
    ("P032", "P033",
     "two unrelated women sharing a given name, one household using an old spelling of the "
     "surname"),
    ("P034", "P035",
     "the second man is not in the Sandhu household, so the identical name does not make "
     "him a sibling of P036 or P037"),
    ("P039", "P040",
     "the roster records these two as the parents of the same child, so the shared family "
     "name is a household fact and not one identity"),
    ("P041", "P042",
     "two different women; the same given and middle name is written by two households, "
     "one of which also records a surname"),
    ("P043", "P044",
     "unrelated households that spell the surname Agrawal and Agarwal"),
    ("P045", "P046",
     "two different women recorded under the same family name; only one of them has a "
     "middle name"),
    ("P047", "P048",
     "two different people, one of them a woman, sharing a given name and a surname"),
    ("P049", "P050",
     "two different women in one Bhattacharya household cluster"),
    ("P051", "P052",
     "two unrelated Thomas families, one of which drops the silent h"),
    ("P054", "P055",
     "unrelated households, one spelling the surname Chauhan and the other Chowhan"),
    ("P056", "P057",
     "two different Trivedi families in different regions, one writing the given name with "
     "a long vowel"),
    ("P058", "P059",
     "two different people, one of them a woman, whose documents carry exactly the same "
     "name"),
    ("P060", "P061",
     "two different women whose documents carry exactly the same name"),
    ("P063", "P064",
     "two different men whose documents carry exactly the same name"),
    ("P065", "P066",
     "the roster records them as husband and wife, so the shared family name is not "
     "evidence of one identity"),
    ("P069", "P070",
     "two different women sharing a given name, one household writing a long Bengali "
     "surname and the other an anglicised one"),
    ("P071", "P054",
     "two unrelated households that both record Chauhan as the family name"),
    ("P072", "P073",
     "two unrelated households, one spelling the surname Mukherjee and the other Mukherji"),
    ("P074", "P075",
     "two unrelated households sharing a surname, one letter apart in the given name"),
    ("P077", "P047",
     "two unrelated households that both record Gupta as the family name"),
    ("P079", "P011",
     "the middle name on one side and the surname on the other are the same word, in two "
     "unrelated households"),
    ("P081", "P082",
     "two different women whose documents carry exactly the same joint surname"),
    ("P083", "P084",
     "one document carries both elements of the joint surname and the other carries only "
     "the first; they are still two different people"),
)


def _fact_for(person_a: str, person_b: str) -> str:
    key = (person_a, person_b) if person_a <= person_b else (person_b, person_a)
    for left, right, clause in _IDENTITY_FACTS:
        pair = (left, right) if left <= right else (right, left)
        if pair == key:
            return clause
    return ""


#: How deceptive a negative is intended to be. This is a *design* label about
#: what the row was built to test, not a measurement, and the two disagree on
#: purpose: 210 rows carry ``hard_negative_*`` in the category column and
#: ``medium`` here, because 137 shared-surname pairs are individually easy and
#: only dangerous in aggregate. The category says what the shape is; this says
#: how hard it was meant to be, and nothing in the pipeline consumes it.
_NEGATIVE_DIFFICULTY = {
    "hard_negative_identical_name": "hard",
    "hard_negative_sibling": "hard",
    "hard_negative_sibling_reordered": "hard",
    "hard_negative_parent_child": "hard",
    "hard_negative_dropped_token": "hard",
    "hard_negative_near_duplicate": "hard",
    "hard_negative_phonetic": "hard",
    "hard_negative_common_surname": "medium",
    "hard_negative_partial_overlap": "medium",
    "unrelated": "easy",
}


def _build_negatives() -> list[dict[str, object]]:
    """Every hard negative the roster genuinely supports, plus a small control.

    The roster is walked exhaustively rather than sampled with a stride: a pair
    is emitted only if it has a hard shape, so exhaustiveness costs nothing in
    filler and covers the confusable space completely. The only "easy" rows are
    the deliberate ``unrelated`` control group at the end.
    """
    roster = list(_ROSTER)
    rows: list[dict[str, object]] = []
    emitted: set[tuple[str, str]] = set()
    filler: list[dict[str, object]] = []

    for a, b in combinations(roster, 2):
        for category, name_a, name_b, clauses in _negative_shape(a, b):
            key = _pair_key(name_a, name_b)
            if key in emitted:
                continue
            emitted.add(key)
            fact = _fact_for(a.person_id, b.person_id)
            rows.append({
                "name_a": name_a,
                "name_b": name_b,
                "label": 0,
                "person_a": a.person_id,
                "person_b": b.person_id,
                "category": category,
                "difficulty": _NEGATIVE_DIFFICULTY[category],
                "rationale": _rationale(name_a, name_b, *clauses, fact),
            })

    # The control group: pairs with no shared token and no phonetic agreement at
    # all. A stride over the roster keeps them spread across regions instead of
    # clustered, and the count is capped so it can never become filler.
    size = len(roster)
    for index in range(size):
        person = roster[index]
        other = roster[(index * 37 + 11) % size]
        if other.person_id == person.person_id:
            continue
        name_a, name_b = person.full, other.full
        tokens_a, tokens_b = _tokens(name_a), _tokens(name_b)
        if set(tokens_a) & set(tokens_b):
            continue
        if _phonetic_trap(tokens_a, tokens_b) is not None:
            continue
        key = _pair_key(name_a, name_b)
        if key in emitted:
            continue
        emitted.add(key)
        filler.append({
            "name_a": name_a,
            "name_b": name_b,
            "label": 0,
            "person_a": person.person_id,
            "person_b": other.person_id,
            "category": "unrelated",
            "difficulty": "easy",
            "rationale": _rationale(
                name_a, name_b,
                "no name token is shared and no pair of tokens collides phonetically, so "
                "the pair is trivially separable"),
        })
        if len(filler) >= UNRELATED_FILLER:
            break

    return rows + filler


# --------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------


def _pair_key(name_a: str, name_b: str) -> tuple[str, str]:
    """The *unordered* key two rows must not share.

    Deduplicating on the ordered tuple let the same comparison appear twice
    because the two names were written in opposite orders -- and because one
    copy was generated as a match and the other as a non-match, "Nisha Yadav"
    vs "Nisha Kumari Yadav" appeared with both labels. The orientation is kept
    for display; only the key is sorted.
    """
    return (name_a, name_b) if name_a <= name_b else (name_b, name_a)


def _positive_plans(reserved: set[tuple[str, str]]) -> list[tuple[Person, str, _PositiveBuilder]]:
    """Assign a (person, category, builder) triple to every roster member.

    Each person gets one positive pair per category they can support, so the
    category histogram is flat by construction rather than by luck. Inside a
    category, builder choice rotates with the person's index so no single
    builder dominates any given category.

    ``reserved`` holds the string keys already claimed by negatives. A positive
    whose rendered strings collide with one of them is skipped rather than
    emitted: two people really can render the same comparison that one person
    renders with themselves, and in that case only the negative is defensible,
    because the identity difference it encodes is the part a matcher would have
    to get right.
    """
    roster = list(_ROSTER)
    by_category: dict[str, list[tuple[Person, _PositiveBuilder]]] = {
        category: [] for category in _POSITIVE_CATEGORY_ORDER
    }
    planned: set[tuple[str, str]] = set()

    for person_index, person in enumerate(roster):
        for category_index, category in enumerate(_POSITIVE_CATEGORY_ORDER):
            candidates = _POSITIVE_BUILDERS[category]
            # The stride must be coprime with the number of builders, or it
            # collapses. `(person_index * 3) % 3` is 0 for every person, so a
            # three-builder category silently used exactly one builder for its
            # entire column -- which is how `surname_first` ended up testing
            # reordering only in combination with an initialised given name.
            stride = _stride_for(len(candidates))
            offset = (person_index * stride + category_index * 2) % len(candidates)
            for step in range(len(candidates)):
                builder = candidates[(offset + step) % len(candidates)]
                result = builder(person)
                if result is None or result[2] != category:
                    continue
                key = _pair_key(result[0], result[1])
                if key in reserved or key in planned:
                    continue
                by_category[category].append((person, builder))
                planned.add(key)
                break
            # If no builder in this category can serve this person (for example
            # "suffix" on a female identity), the pair is simply not emitted.

    # Thin each category down to POSITIVES_PER_CATEGORY with an even stride, so
    # every category is well represented but no single one dominates the set.
    plans: list[tuple[Person, str, _PositiveBuilder]] = []
    for category_index, (category, entries) in enumerate(by_category.items()):
        if not entries:
            continue
        target = min(POSITIVES_PER_CATEGORY, len(entries))
        stride = len(entries) / target
        # The sampling window has to rotate between categories. Sampling
        # `entries[0::stride]` from the same start for every category drops the
        # tail of the roster from the whole dataset, which is how one identity
        # was present in the roster and absent from every pair -- and with it
        # the only reachable `D/o` row.
        offset = (category_index * 7) % len(entries)
        for slot in range(target):
            person, builder = entries[(offset + int(slot * stride)) % len(entries)]
            plans.append((person, category, builder))
    return plans


def _held_out_rows(reserved: set[tuple[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pid, surface_a, surface_b, why in _HELD_OUT_TRANSLIT_UNRESOLVED:
        if _pair_key(surface_a, surface_b) in reserved:
            raise AssertionError(
                f"{pid}: held-out pair {surface_a!r} / {surface_b!r} collides with a "
                f"negative; two labels for one comparison is not a dataset")
        rows.append({"name_a": surface_a, "name_b": surface_b, "label": 1,
                     "person_a": pid, "person_b": pid,
                     "category": "transliteration_heldout", "difficulty": "hard",
                     "rationale": _rationale(surface_a, surface_b, why)})
    for pid, surface_a, surface_b, why in _HELD_OUT_TRANSLIT_FOLDED:
        if _pair_key(surface_a, surface_b) in reserved:
            raise AssertionError(
                f"{pid}: held-out pair {surface_a!r} / {surface_b!r} collides with a "
                f"negative; two labels for one comparison is not a dataset")
        rows.append({"name_a": surface_a, "name_b": surface_b, "label": 1,
                     "person_a": pid, "person_b": pid,
                     "category": "transliteration_heldout", "difficulty": "medium",
                     "rationale": _rationale(surface_a, surface_b, why)})
    return rows


def build_dataset() -> list[NamePair]:
    """Build the full labelled dataset. Deterministic and side-effect free."""
    # Negatives are assembled first so their string keys can be reserved: a
    # positive is never allowed to reuse a comparison a negative already claims.
    negative_rows = _build_negatives()
    reserved = {_pair_key(str(row["name_a"]), str(row["name_b"])) for row in negative_rows}

    # The hand-curated held-out rows reserve their keys too, so the generic
    # planner does not regenerate the same comparison as a plain transliteration:
    # one observation, one row.
    held_out_rows = _held_out_rows(reserved)
    reserved |= {_pair_key(str(row["name_a"]), str(row["name_b"]))
                 for row in held_out_rows}

    positive_rows: list[dict[str, object]] = []

    # ---- positives: roster sweep (label 1, person_a == person_b) ---------
    for person, category, builder in _positive_plans(reserved):
        result = builder(person)
        if result is None:
            continue
        name_a, name_b, achieved, difficulty, rationale = result
        # Structural guarantee: a "match" pair must reference one identity, and
        # the recorded category must be the one actually achieved.
        assert achieved == category, f"{builder.__name__} produced {achieved!r}"
        positive_rows.append({"name_a": name_a, "name_b": name_b, "label": 1,
                              "person_a": person.person_id, "person_b": person.person_id,
                              "category": category, "difficulty": difficulty,
                              "rationale": rationale})

    rows = positive_rows + held_out_rows + negative_rows

    # ---- integrity gate --------------------------------------------------
    # A dataset that hands the same two strings two different labels is
    # self-contradictory, and no metric computed from it means anything.
    by_key: dict[tuple[str, str], int] = {}
    for row in rows:
        key = _pair_key(str(row["name_a"]), str(row["name_b"]))
        label = int(row["label"])  # type: ignore[arg-type]
        if key in by_key:
            raise AssertionError(
                f"{row['name_a']!r} / {row['name_b']!r} appears twice "
                f"(labels {by_key[key]} and {label})")
        by_key[key] = label

    pairs = [
        NamePair(
            pair_id=f"N{index:04d}",
            name_a=str(row["name_a"]),
            name_b=str(row["name_b"]),
            label=int(row["label"]),          # type: ignore[arg-type]
            person_a=str(row["person_a"]),
            person_b=str(row["person_b"]),
            category=str(row["category"]),
            difficulty=str(row["difficulty"]),
            rationale=str(row["rationale"]),
        )
        for index, row in enumerate(rows, start=1)
    ]

    for pair in pairs:
        assert (pair.label == 1) == (pair.person_a == pair.person_b), pair.pair_id
        assert pair.category in CATEGORIES, pair.pair_id
        assert pair.difficulty in ("easy", "medium", "hard"), pair.pair_id
        assert pair.rationale.startswith(f"{pair.name_a!r} vs {pair.name_b!r}: "), pair.pair_id
    return pairs


# --------------------------------------------------------------------------
# CSV I/O
# ------------------------------------------------------------------------# --------------------------------------------------------------------------

CSV_FIELDS = (
    "pair_id", "name_a", "name_b", "label", "person_a", "person_b",
    "category", "difficulty", "rationale",
)

_HEADER_LINES = (
    "# Name-matching exercise -- labelled name pairs",
    f"# schema_version: {SCHEMA_VERSION}",
    "# label: 1 = same person, 0 = different people",
    "# Labels are assigned from person identity, never from string similarity.",
    "# category/difficulty: what the surface difference actually IS, not what was intended.",
    "# rationale: describes the two strings on its own row, and no pair of strings"
    " carries two labels.",
    "# categories: " + "; ".join(f"{key} = {value}" for key, value in CATEGORIES.items()),
)


def write_csv(pairs: Iterable[NamePair], path: str = DEFAULT_DATA_PATH) -> str:
    """Write the dataset to ``path`` with a commented header. Returns the path."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for line in _HEADER_LINES:
            handle.write(f"{line}\n")
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair.as_row())
    return path


def read_csv(path: str = DEFAULT_DATA_PATH) -> list[NamePair]:
    """Read a dataset written by :func:`write_csv`."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        body = "".join(line for line in handle if not line.startswith("#"))
    return [
        NamePair(
            pair_id=row["pair_id"],
            name_a=row["name_a"],
            name_b=row["name_b"],
            label=int(row["label"]),
            person_a=row["person_a"],
            person_b=row["person_b"],
            category=row["category"],
            difficulty=row["difficulty"],
            rationale=row.get("rationale", ""),
        )
        for row in csv.DictReader(io.StringIO(body))
    ]
