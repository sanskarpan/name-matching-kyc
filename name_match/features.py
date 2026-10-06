"""Feature extraction for the learned combiner.

One place builds the feature vector, and both the learned matcher and the
ablation report read from it, so there is no risk of the model being trained on
features that differ from the ones it is evaluated with. The ablation removes
*columns* from this same matrix rather than recomputing anything, which is what
makes it a clean one-variable-at-a-time comparison.

Design constraint: the feature set must contain nothing that leaks the label.
All features are functions of the two name strings alone.
"""

from __future__ import annotations

from . import phonetics, string_algos
from .normalize import normalize

__all__ = ["FEATURE_NAMES", "extract_features", "extract_matrix"]

#: Ordered feature names. Order is part of the model artefact: it is written
#: into the saved model and checked on load.
FEATURE_NAMES: tuple[str, ...] = (
    "exact_normalized",
    "token_jaccard",
    "token_dice",
    "bigram_jaccard",
    "trigram_jaccard",
    "jaro_winkler_sorted",
    "jaro_winkler_ordered",
    "damerau_ratio",
    "prefix_ratio",
    "lcs_ratio",
    "phonetic_mean",
    "phonetic_max",
    "soundex_full_match",
    "coverage",
    "agreement",
    "order_bonus",
    "char_signal",
    "n_token_diff",
    "has_initial_either",
    "dropped_honorific",
    "dropped_qualifier",
    "length_ratio",
    "same_token_count",
)


def _soundex_collision(left_text: str, right_text: str) -> float:
    """1.0 when the two normalised names share a full-string Soundex code.

    Both codes must be non-empty. `soundex` returns "" for a digit-only string,
    so a naive equality test scored 1.0 for *any* two different digit strings --
    the same phantom-collision class of bug fixed inside
    `phonetic_agreement`, left behind in this one caller.
    """
    left_code = phonetics.soundex(left_text)
    right_code = phonetics.soundex(right_text)
    return float(bool(left_code) and left_code == right_code)


def _token_dice(left: list[str], right: list[str]) -> float:
    """Sorensen-Dice over token *sets*.

    Both the intersection and the total are taken over sets. Dividing a set
    intersection by a list-length sum deflates the coefficient whenever a name
    repeats a token ("Anil Anil Kumar"), and makes this feature disagree with
    the token-set Jaccard computed on the line above it.
    """
    set_left, set_right = set(left), set(right)
    if not set_left and not set_right:
        return 1.0
    total = len(set_left) + len(set_right)
    if total == 0:
        return 1.0
    return 2.0 * len(set_left & set_right) / total


def extract_features(name_a: str, name_b: str) -> tuple[list[float], dict[str, float]]:
    """Return ``(feature_vector, human_readable_components)``.

    The second element is a small dict of the intermediate quantities worth
    looking at when debugging a specific pair.
    """
    from .algorithms import PhoneticJaroWinkler, TokenAlignedScorer

    left = normalize(name_a)
    right = normalize(name_b)

    tokens_a, tokens_b = list(left.canonical), list(right.canonical)
    sorted_a, sorted_b = " ".join(sorted(tokens_a)), " ".join(sorted(tokens_b))

    set_a, set_b = set(tokens_a), set(tokens_b)
    union = set_a | set_b
    jaccard = len(set_a & set_b) / len(union) if union else 0.0

    aligned_result = TokenAlignedScorer().explain(name_a, name_b)
    phonetic_result = PhoneticJaroWinkler().explain(name_a, name_b)

    # --- initial / dropped-token signals ---------------------------------
    # Symmetric on purpose. Splitting this into "initial on side A" and
    # "initial on side B" makes the feature vector -- and therefore the score --
    # depend on which document the caller happened to pass first. The learned
    # combiner is then *not* symmetric, and can flip a match decision when the
    # two documents are swapped, which disqualifies it from use.
    #
    # It is also worse than useless as a pair: the dataset only ever abbreviates
    # the name in slot `a`, so "initial on side B" is a constant-zero column
    # whose weight is pinned at exactly 0.0 by the zero-variance guard in
    # `standardize`, while "initial on side A" learns "abbreviated document =>
    # match". Swapping the arguments then moves the initial between a feature
    # weighted +0.92 and a feature weighted 0.0, which is a large swing on no
    # evidence at all.
    has_initial_either = float(bool(left.initials) or bool(right.initials))

    dropped = set(left.dropped) | set(right.dropped)
    from .normalize import HONORIFICS, RELATIONSHIP_QUALIFIERS, SUFFIXES

    dropped_honorific = float(bool(dropped & (HONORIFICS | SUFFIXES)))
    dropped_qualifier = float(bool(dropped & RELATIONSHIP_QUALIFIERS))

    # --- length / count signals ------------------------------------------
    length_a, length_b = len(left.canonical_text), len(right.canonical_text)
    length_ratio = (min(length_a, length_b) / max(length_a, length_b)) if max(length_a, length_b) else 0.0

    # --- empty-input guard ------------------------------------------------
    if not tokens_a or not tokens_b:
        vector = [0.0] * len(FEATURE_NAMES)
        return vector, {"empty": 1.0}

    # Index 0 (exact_normalized) is filled in after the literal is built, so
    # the feature order below reads left to right in FEATURE_NAMES order.
    vector = [
        0.0,  # exact_normalized -- assigned just below
        jaccard,
        _token_dice(tokens_a, tokens_b),
        string_algos.ngram_jaccard(sorted_a, sorted_b, n=2),
        string_algos.ngram_jaccard(sorted_a, sorted_b, n=3),
        string_algos.jaro_winkler(sorted_a, sorted_b),
        string_algos.jaro_winkler(left.canonical_text, right.canonical_text),
        1.0 - string_algos.damerau_osa(sorted_a, sorted_b) / max(len(sorted_a), len(sorted_b), 1),
        string_algos.prefix_ratio(sorted_a, sorted_b),
        string_algos.longest_common_substring_ratio(sorted_a, sorted_b),
        phonetic_result.components.get("phonetic_mean", 0.0),
        phonetic_result.components.get("phonetic_max", 0.0),
        _soundex_collision(left.canonical_text, right.canonical_text),
        aligned_result.components.get("coverage", 0.0),
        aligned_result.components.get("agreement", 0.0),
        aligned_result.components.get("order_bonus", 1.0),
        aligned_result.components.get("char_signal", 0.0),
        float(abs(len(tokens_a) - len(tokens_b))),
        has_initial_either,
        dropped_honorific,
        dropped_qualifier,
        length_ratio,
        float(len(tokens_a) == len(tokens_b)),
    ]

    # Feature 0 is exact normalised equality. Computed here rather than by
    # instantiating the matcher so feature order stays self-documenting.
    vector[0] = 1.0 if set_a == set_b else 0.0

    # `TokenAlignedScorer.explain` short-circuits to a two-key components dict
    # when no tokens align at all. Reading that dict with `.get(..., 1.0)` for
    # `order_bonus` injects a *positive* 1.0 -- the strongest possible
    # "these two names agree perfectly on order" -- for a pair that shares no
    # tokens at all, and drops the real character-level similarity on the floor.
    # Both are backfilled here with values computed from first principles.
    if aligned_result.components.get("n_aligned", 0.0) == 0.0:
        char_signal = string_algos.jaro_winkler(sorted_a, sorted_b)
        vector[FEATURE_NAMES.index("char_signal")] = char_signal
        vector[FEATURE_NAMES.index("order_bonus")] = 0.0

    components = {
        "jaccard": jaccard,
        "coverage": aligned_result.components.get("coverage", 0.0),
        "agreement": aligned_result.components.get("agreement", 0.0),
        "phonetic_mean": phonetic_result.components.get("phonetic_mean", 0.0),
        "n_left": float(len(tokens_a)),
        "n_right": float(len(tokens_b)),
    }
    return vector, components


def extract_matrix(name_pairs: list[tuple[str, str]]) -> list[list[float]]:
    """Feature matrix for a list of ``(name_a, name_b)`` tuples."""
    return [extract_features(a, b)[0] for a, b in name_pairs]
