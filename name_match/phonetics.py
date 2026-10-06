"""Phonetic encodings: Soundex and a lightweight Double Metaphone.

Both are implemented from scratch (the algorithms are public-domain classic
algorithms) so the package stays dependency-free.

Why phonetics matter here: the single largest source of *false positives* in
name matching is that orthographically different names are pronounced the
same. Soundex collapses "Sharma" and "Sarma" to the same code,
which is precisely the trap we want to measure and report.
"""

from __future__ import annotations

__all__ = ["soundex", "metaphone", "metaphone_pair", "phonetic_agreement"]

_SOUNDEX_MAP = {
    **dict.fromkeys("bfpv", "1"),
    **dict.fromkeys("cgjkqsxz", "2"),
    **dict.fromkeys("dt", "3"),
    "l": "4",
    **dict.fromkeys("mn", "5"),
    "r": "6",
}


def soundex(token: str) -> str:
    """Classic 4-character Soundex code, e.g. ``"suresh" -> "S620"``.

    H/W are transparent (ignored, and do not reset the previous digit). This is
    the original Russell/Odell algorithm, kept unmodified on purpose so the
    known collision behaviour is measurable.
    """
    if not token:
        return ""

    # ASCII only. `.upper()` is not length-preserving: 'ss'.upper() is 'SS' and
    # 'ffl'.upper() is 'FFL', either of which would break the 4-character
    # contract the rest of the module (and every caller) relies on.
    letters = [ch for ch in token.lower() if ch.isalpha() and ch.isascii()]
    if not letters:
        return ""

    first_letter = letters[0]
    # Drop a leading H/W if present (they carry no code).
    rest = letters[1:]
    while rest and rest[0] in "hw":
        rest = rest[1:]

    codes: list[str] = []
    previous_code = _SOUNDEX_MAP.get(first_letter, "")
    for ch in rest:
        code = _SOUNDEX_MAP.get(ch, "")
        if code:
            if code != previous_code:
                codes.append(code)
            previous_code = code
        elif ch not in "hw":
            # A vowel (or other unmapped letter) breaks the run.
            previous_code = ""

    digits = ("".join(codes) + "000")[:3]
    return (first_letter.upper() + digits)


# --------------------------------------------------------------------------
# Metaphone
# --------------------------------------------------------------------------

# Initial-letter exceptions, applied before the main rules.
# Keys are stored UPPER-case because they are matched against an upper-cased
# word. Stored lower-case they never matched anything, so the whole table was
# dead code: the silent initial letters in KNIFE, GNOME, PNEUMATIC, WRATH,
# WHILE and XY were all being retained, which is precisely where metaphone is
# supposed to earn its keep.
_METAPHONE_INITIAL = {
    "AE": "E",
    "GN": "N",
    "KN": "N",
    "PN": "N",
    "WR": "R",
    "X": "S",
    "WH": "W",
}

# (pattern, replacement) rules. Applied longest-pattern-first at each position.
# ``"#"`` is a hard boundary: the character is dropped and the two sides of the
# boundary may not cohere. ``"+"`` marks a secondary (alternative) pronunciation.
# Patterns are stored lower-case; the encoder upper-cases the input word, so
# the table is upper-cased once at import time for fast matching.
_METAPHONE_RULES: list[tuple[str, str]] = [
    ("tch", "X"),
    ("dge", "J"),
    ("sch", "SK"),
    ("ph", "F"),
    ("gh", "K"),
    ("kh", "K"),
    ("th", "T"),
    ("ch", "X"),
    ("sh", "X"),
    ("wh", "W"),
    ("ck", "K"),
    ("ng", "NG"),
    ("qu", "KW"),
    ("dg", "J"),
    ("ae", "E"),
    ("oe", "O"),
    ("ee", "E"),
    ("ea", "E"),
    ("ie", "I"),
    ("ai", "E"),
    ("oa", "O"),
    ("oo", "U"),
    ("au", "O"),
    ("ou", "U"),
    ("ei", "E"),
    ("ey", "E"),
    ("igh", "I"),
    ("a", "E"),
    ("b", "B"),
    ("c", "K"),
    ("d", "T"),
    ("e", "E"),
    ("f", "F"),
    ("g", "K"),
    ("h", ""),
    ("i", "I"),
    ("j", "J"),
    ("k", "K"),
    ("l", "L"),
    ("m", "M"),
    ("n", "N"),
    ("o", "O"),
    ("p", "P"),
    ("q", "K"),
    ("r", "R"),
    ("s", "S"),
    ("t", "T"),
    ("u", "U"),
    ("v", "F"),
    ("w", "W"),
    ("x", "KS"),
    ("y", "Y"),
    ("z", "S"),
]

#: Letters that go silent after a vowel. Compared against the upper-cased
#: output buffer, so this set must be upper-case too -- as a lower-case set it
#: never matched and the documented "silent W/Y after a vowel" behaviour simply
#: did not exist.
_VOWEL_SOFTENING = frozenset("WKY")

#: Vowels, upper-case to match the output buffer.
_VOWELS = frozenset("AEIOU")

# Upper-cased view of the rule table, sorted longest-pattern-first so that
# "tch" is tried before "t" and "sch" before "c".
_METAPHONE_RULES_CI: list[tuple[str, str]] = sorted(
    ((pattern.upper(), replacement) for pattern, replacement in _METAPHONE_RULES),
    key=lambda rule: -len(rule[0]),
)


def metaphone(token: str) -> str:
    """A compact metaphone encoding.

    Simplified relative to Lawrence Philips' full Double Metaphone: it applies
    the prefix, suffix and medial consonant rules, and softens ``W``/``Y`` in
    vowel-adjacent positions. It is intentionally a single code; the pair form
    below adds the one split that actually matters for Indian names
    (``VOWEL`` -> ``F``/``W``).
    """
    if not token:
        return ""

    word = token.upper()
    if not word.isalpha():
        word = "".join(ch for ch in word if ch.isalpha())
        if not word:
            return ""

    # Drop a silent leading letter.
    for prefix, replacement in _METAPHONE_INITIAL.items():
        if word.startswith(prefix) and len(word) > 1:
            word = replacement + word[len(prefix) :]
            break

    out: list[str] = []
    index = 0
    length = len(word)

    while index < length:
        ch = word[index]
        consumed = 0

        for pattern, replacement in _METAPHONE_RULES_CI:
            if word.startswith(pattern, index):
                consumed = len(pattern)
                # Append the WHOLE replacement. Truncating to `replacement[0]`
                # silently collapsed "SK"->"S", "KS"->"K", "NG"->"N" and
                # "KW"->"K", so SCH-, -X-, -QU- and -NG- all lost half their
                # code and the encoder was less discriminative than it claimed.
                #
                # Suppress a duplicate only when the full replacement already
                # ends the output, since "SS" and "S" are different codes.
                if replacement and "".join(out[-len(replacement):]) != replacement:
                    out.append(replacement)
                index += consumed
                break
        else:
            index += 1
            continue

        # Skip a second copy of a doubled letter (LL -> L, SS -> S).
        if consumed == 1 and index < length and word[index] == ch:
            index += 1

        # Vowel-following letter softening: W/Y after a vowel is silent.
        # The code being tested for silencing must be the W/Y that was *just*
        # emitted. Popping unconditionally removed the preceding vowel instead,
        # so even with the case fixed this branch corrupted its neighbours.
        if (consumed == 1 and ch in _VOWEL_SOFTENING and out
                and out[-1] == ch and len(out) >= 2 and out[-2] in _VOWELS):
            out.pop()

    return "".join(out)


def metaphone_pair(token: str) -> tuple[str, ...]:
    """Primary and alternate metaphone codes, sorted for set comparison.

    The only alternate produced here is the V/F split, which is the one that
    fires on real surname pairs in this dataset (Vasudevan / Wasudevan).
    """
    primary = metaphone(token)
    alternates = {primary}

    upper = token.upper()
    # Medial "V" behaves as F; at the start of a name it behaves as W.
    for variant in _v_f_variants(upper):
        code = metaphone(variant)
        if code:
            alternates.add(code)

    return tuple(sorted(alternates))


def _v_f_variants(word: str) -> list[str]:
    """Enumerate plausible alternative readings of a medial V/F/W.

    Two substitutions are generated, not one:

    * ``V`` -> ``F``, the reading the main rule already applies.
    * a *leading* ``V`` -> ``W``, which is the real pronunciation of
      "Vasudevan" and the only thing that makes the
      ``Vasudevan``/``Wasudevan`` pair in the docstring above actually collide.

    The previous version only swapped V and F, so the pair it documented could
    never produce a match and the documented capability did not exist.
    """
    variants: list[str] = []

    for index, char in enumerate(word):
        if char not in "VF":
            continue
        swapped = list(word)
        swapped[index] = "F" if char == "V" else "V"
        candidate = "".join(swapped)
        if candidate != word:
            variants.append(candidate)

    # A word-initial V is pronounced W, which the leading-letter rules do not
    # cover because V is not among them.
    if word.startswith("V"):
        variants.append("W" + word[1:])

    return variants


def phonetic_agreement(token_a: str, token_b: str) -> float:
    """Best agreement between two tokens across Soundex and Metaphone.

    Returns 0.0 (no phonetic agreement), or a value in (0, 1] where 1.0 means
    an exact Soundex-code collision -- deliberately a *low* value, because a
    Soundex collision is weak evidence, not proof. See NOTES.md.
    """
    if not token_a or not token_b:
        return 0.0

    # Phonetics are a claim about pronunciation, which requires letters. A
    # digit-only token has no code from either encoder, and because
    # `metaphone_pair` returns a one-element tuple containing the empty string,
    # two *different* digit strings shared an "empty code" and scored 0.8 --
    # pure noise injected into two of the five matchers.
    if not (token_a.isalpha() and token_b.isalpha()):
        return 0.0

    sx_a, sx_b = soundex(token_a), soundex(token_b)
    sx_match = bool(sx_a) and sx_a == sx_b

    mp_a = set(metaphone_pair(token_a))
    mp_b = set(metaphone_pair(token_b))
    mp_match = bool(mp_a) and bool(mp_b) and bool(mp_a & mp_b)

    if sx_match and mp_match:
        return 1.0
    if sx_match or mp_match:
        return 0.8
    return 0.0
