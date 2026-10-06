"""Core string distance / similarity primitives.

Implemented from scratch so the whole exercise runs on a bare Python runtime
and so the failure modes of each metric are explicit rather than hidden inside
a third-party implementation.

All functions operate on plain ``str``. Callers are expected to pass already
case-folded / punctuation-stripped text (see :mod:`name_match.normalize`).
"""

from __future__ import annotations

from functools import lru_cache

__all__ = [
    "levenshtein",
    "damerau_osa",
    "levenshtein_ratio",
    "jaro",
    "jaro_winkler",
    "ngram_set",
    "ngram_jaccard",
    "ngram_dice",
    "longest_common_substring_ratio",
    "prefix_ratio",
    "is_prefix_of",
]


# --------------------------------------------------------------------------
# Edit distance
# --------------------------------------------------------------------------


def levenshtein(a: str, b: str) -> int:
    """Classic unit-cost Levenshtein distance.

    O(len(a) * len(b)) time, O(min(len(a), len(b))) space.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Iterate over the shorter string in the inner loop for cache locality.
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            substitute_cost = previous[j - 1] + (ca != cb)
            current[j] = min(insert_cost, delete_cost, substitute_cost)
        previous = current
    return previous[-1]


def damerau_osa(a: str, b: str) -> int:
    """Optimal string alignment distance (restricted Damerau-Levenshtein).

    Unlike plain Levenshtein this allows a single adjacent transposition, so
    "Vijay" and the typo "Vjiay" are distance 1 here and distance 2 under plain
    Levenshtein. That transposition is the single most common real-world
    spelling slip in typed names, and it is why a plain edit distance
    systematically over-charges for it.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    len_a, len_b = len(a), len(b)
    d = [[0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(len_a + 1):
        d[i][0] = i
    for j in range(len_b + 1):
        d[0][j] = j

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,  # deletion
                d[i][j - 1] + 1,  # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )
            if (
                i > 1
                and j > 1
                and a[i - 1] == b[j - 2]
                and a[i - 2] == b[j - 1]
            ):
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)  # transposition
    return d[len_a][len_b]


def levenshtein_ratio(a: str, b: str) -> float:
    """Edit distance mapped to [0, 1]; 1.0 means identical.

    ``1 - d / max(len(a), len(b))`` degrades gracefully for empty strings
    (two empty strings are a perfect match, one empty string is a total miss).
    """
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return 1.0 - levenshtein(a, b) / longest


# --------------------------------------------------------------------------
# Jaro / Jaro-Winkler
# --------------------------------------------------------------------------


def _jaro_matches(a: str, b: str, match_window: int) -> tuple[int, list[tuple[int, int]]]:
    """Count matching characters and return the aligned index pairs.

    The aligned pairs are returned because :func:`jaro` needs them to count
    transpositions correctly, and discarding them forced the wrong count.
    """
    len_a, len_b = len(a), len(b)
    if len_a == 0 or len_b == 0:
        return 0, []

    flags_b = [False] * len_b
    matches = 0
    aligned: list[tuple[int, int]] = []

    for i, ca in enumerate(a):
        start = max(0, i - match_window)
        end = min(i + match_window + 1, len_b)
        for j in range(start, end):
            if not flags_b[j] and b[j] == ca:
                flags_b[j] = True
                matches += 1
                aligned.append((i, j))
                break
    return matches, aligned


def jaro(a: str, b: str) -> float:
    """Jaro similarity in [0, 1].

    Gives partial credit for shared prefixes and for characters that appear in
    the same relative neighbourhood, which is why it behaves much better than
    raw edit distance on *reordered* names.
    """
    if a == b:
        return 1.0
    len_a, len_b = len(a), len(b)
    if len_a == 0 or len_b == 0:
        return 0.0

    match_window = max(len_a, len_b) // 2 - 1
    if match_window < 0:
        match_window = 0

    matches, aligned = _jaro_matches(a, b, match_window)
    if matches == 0:
        return 0.0

    # Transpositions = half the number of matched characters that are out of
    # order. The correct way to count those is to walk the two aligned
    # subsequences in the same order and count the positions where they
    # disagree; using `matches // 2` instead treats *every* matched character as
    # half a transposition, which penalises any pair that shares characters in
    # the right order. "martha"/"marhta" should score 0.944 and scored 0.833.
    #
    # Cost: jaro_winkler inherits this, so the error propagated into the
    # phonetic and token-aligned scorers.
    left_sequence = [a[i] for i, _j in aligned]
    right_sequence = [b[j] for i, j in sorted(aligned, key=lambda pair: pair[1])]
    mismatches = sum(
        1 for left_char, right_char in zip(left_sequence, right_sequence)
        if left_char != right_char
    )
    transpositions = mismatches / 2.0

    return (
        matches / len_a + matches / len_b + (matches - transpositions) / matches
    ) / 3.0


@lru_cache(maxsize=8192)
def jaro_winkler(a: str, b: str, prefix_weight: float = 0.1, max_prefix: int = 4) -> float:
    """Jaro-Winkler similarity: Jaro plus a bonus for a shared prefix.

    The prefix bonus is a liability for this task (see NOTES.md) -- it rewards
    "surname-first" orderings on one side only -- but it is included as a
    baseline because it is the default choice most teams reach for.
    """
    jaro_score = jaro(a, b)
    if jaro_score < 0.7:
        # Standard practice: only apply the boost above a similarity floor.
        return jaro_score

    prefix = 0
    for ca, cb in zip(a[:max_prefix], b[:max_prefix]):
        if ca != cb:
            break
        prefix += 1
    if not prefix:
        return jaro_score
    # Clamped: with a large enough `prefix_weight` the boost alone can exceed the
    # remaining headroom, and a similarity above 1.0 would then silently poison
    # every downstream threshold. The default weight of 0.1 cannot do this,
    # which is exactly why it went unnoticed until the parameter was varied.
    return min(1.0, jaro_score + prefix * prefix_weight * (1.0 - jaro_score))


# --------------------------------------------------------------------------
# n-gram overlap
# --------------------------------------------------------------------------


def ngram_set(s: str, n: int = 2) -> frozenset[str]:
    """Character n-gram set, space-padded so word boundaries are visible."""
    if not s:
        return frozenset()
    padded = f" {s} "
    if len(padded) < n:
        return frozenset({padded})
    return frozenset(padded[i : i + n] for i in range(len(padded) - n + 1))


def ngram_jaccard(a: str, b: str, n: int = 2) -> float:
    """Jaccard index over character n-grams. 1.0 iff the n-gram sets are equal."""
    set_a, set_b = ngram_set(a, n), ngram_set(b, n)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def ngram_dice(a: str, b: str, n: int = 2) -> float:
    """Sorensen-Dice coefficient over character n-grams.

    Included because Dice is far more forgiving of a single missing token than
    Jaccard: dropping "Kumar" from "Suresh Kumar Sharma" costs Jaccard far more.
    """
    set_a, set_b = ngram_set(a, n), ngram_set(b, n)
    if not set_a and not set_b:
        return 1.0
    total = len(set_a) + len(set_b)
    if total == 0:
        return 1.0
    return 2.0 * len(set_a & set_b) / total


def longest_common_substring_ratio(a: str, b: str) -> float:
    """Longest common substring length divided by the longer string's length.

    Cheap containment signal: a name that is a substring of the other scores
    high, which is what we want for "K. Venkatesh" vs "K. Venkatesh Rao".
    """
    if not a or not b:
        return 0.0
    len_a, len_b = len(a), len(b)
    if len_a > len_b:
        a, b = b, a
        len_a, len_b = len_b, len_a

    previous = [0] * (len_a + 1)
    best = 0
    for j in range(1, len_b + 1):
        current = [0] * (len_a + 1)
        cb = b[j - 1]
        for i in range(1, len_a + 1):
            if a[i - 1] == cb:
                current[i] = previous[i - 1] + 1
                if current[i] > best:
                    best = current[i]
        previous = current
    return best / len_b


# --------------------------------------------------------------------------
# Prefix helpers (used for initials-vs-expanded matching)
# --------------------------------------------------------------------------


def is_prefix_of(short: str, long: str) -> bool:
    """True if ``short`` is a non-empty prefix of ``long`` (or vice versa)."""
    if not short or not long:
        return False
    shorter, longer = (short, long) if len(short) <= len(long) else (long, short)
    return longer.startswith(shorter)


def prefix_ratio(a: str, b: str) -> float:
    """Length of the shared prefix divided by the **shorter** string's length.

    A soft, partial-credit version of :func:`is_prefix_of`. Because the
    denominator is the shorter string, a strict prefix scores 1.0 and a shared
    prefix that runs out before the shorter string does scores in between
    (``"suresh"`` vs ``"sursh"`` shares 3 of 5 characters, so 0.6). It does
    **not** penalise a long token merely for being longer than a short one,
    which is why the caller in :mod:`name_match.algorithms` pairs it with a
    length-ratio check.
    """
    if not a or not b:
        return 0.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    shared = 0
    for ca, cb in zip(shorter, longer):
        if ca != cb:
            break
        shared += 1
    return shared / len(shorter)
