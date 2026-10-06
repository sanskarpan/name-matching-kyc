"""The five name-matching algorithms under evaluation.

Every matcher exposes the same contract::

    matcher.score(name_a, name_b) -> float in [0.0, 1.0]

plus :meth:`Matcher.explain`, which returns the intermediate components that
produced the score. The explanation is not decoration: it is what makes a
failure in reports/results.md diagnosable instead of mysterious, and it is what
the feature extractor for the learned model consumes.

The five matchers
-----------------
1. :class:`ExactNormalizedMatch` -- the control. Pure set equality after
   normalisation. Establishes the floor and exposes how much of the problem is
   solved by cleaning alone.
2. :class:`TokenSetJaccard` -- order-invariant token overlap. The most common
   approach in production KYC pipelines, and the one that fails hardest on
   initials.
3. :class:`PhoneticJaroWinkler` -- phonetic codes plus Jaro-Winkler. The
   standard "robust to spelling variation" answer, and the one that produces
   the most confident false positives.
4. :class:`TokenAlignedScorer` -- a hand-weighted, token-aligned scorer that
   models the actual KYC failure modes: order, initials, dropped middle names,
   compound surnames and transliteration.
5. :class:`LearnedCombiner` -- logistic regression over the feature vector
   extracted from the other four. See :mod:`name_match.model`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from . import phonetics, string_algos
from .normalize import NormalizedName, normalize

__all__ = [
    "MatchResult",
    "Matcher",
    "ExactNormalizedMatch",
    "TokenSetJaccard",
    "PhoneticJaroWinkler",
    "TokenAlignedScorer",
    "LearnedCombiner",
    "ALL_MATCHERS",
    "BASELINE_MATCHERS",
    "get_matcher",
    "matcher_names",
]


@dataclass(frozen=True)
class MatchResult:
    """A score plus the evidence behind it."""

    score: float
    components: dict[str, float]

    def __float__(self) -> float:
        return self.score


class Matcher(Protocol):
    """Structural type every matcher satisfies."""

    key: str
    label: str

    def score(self, name_a: str, name_b: str) -> float: ...

    def explain(self, name_a: str, name_b: str) -> MatchResult: ...


def _clamp(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


# ==========================================================================
# 1. Exact normalised equality -- the control
# ==========================================================================


class ExactNormalizedMatch:
    """Set equality on normalised tokens.

    The control condition. It answers the question "how far does cleaning text
    alone get us?", which is the baseline every other algorithm has to beat
    before its cleverness is worth discussing.
    """

    key = "exact_normalized"
    label = "Exact (normalised tokens)"

    def explain(self, name_a: str, name_b: str) -> MatchResult:
        left = normalize(name_a)
        right = normalize(name_b)

        reason = _undecidable_reason(left, right)
        if reason == "unsupported_script":
            return MatchResult(UNDECIDABLE_SCORE,
                               {"unsupported_script": 1.0})
        if reason:
            # Two empty names are not a match. Silently scoring 1.0 here is a
            # classic production bug: a failed extraction looks like agreement.
            return MatchResult(0.0, {"empty_left": float(not left.canonical),
                                      "empty_right": float(not right.canonical)})

        # Set comparison, so token order is ignored.
        equal = set(left.canonical) == set(right.canonical)
        # Partial credit for reporting this algorithm's own view of overlap,
        # which makes its ROC curve non-degenerate and its errors readable.
        union = set(left.canonical) | set(right.canonical)
        jaccard = len(set(left.canonical) & set(right.canonical)) / len(union)

        return MatchResult(
            score=1.0 if equal else 0.0,
            components={"token_jaccard": jaccard, "exact": float(equal)},
        )

    def score(self, name_a: str, name_b: str) -> float:
        return self.explain(name_a, name_b).score


# ==========================================================================
# 2. Token-set Jaccard -- the common production baseline
# ==========================================================================


class TokenSetJaccard:
    """Order-invariant Jaccard overlap over normalised tokens.

    Handles dropped tokens and reordering well. It cannot handle an abbreviated
    given name, because ``{"s", "kumar"}`` and ``{"suresh", "kumar"}`` share
    exactly one of three elements. That failure is structural, not a tuning
    problem, and it is the single most common bug in real KYC name matching.
    """

    key = "token_jaccard"
    label = "Token-set Jaccard"

    def explain(self, name_a: str, name_b: str) -> MatchResult:
        left = normalize(name_a)
        right = normalize(name_b)

        reason = _undecidable_reason(left, right)
        if reason == "unsupported_script":
            return MatchResult(UNDECIDABLE_SCORE, {"unsupported_script": 1.0})
        if reason:
            return MatchResult(0.0, {"empty": 1.0})

        set_a, set_b = set(left.canonical), set(right.canonical)
        union = set_a | set_b
        jaccard = len(set_a & set_b) / len(union) if union else 1.0

        # Order sensitivity, reported but not used in the score: it is the
        # clearest possible demonstration that this family of algorithms is
        # blind to name order.
        same_order = list(left.canonical) == list(right.canonical)

        return MatchResult(
            score=_clamp(jaccard),
            components={
                "jaccard": jaccard,
                "same_order": float(same_order),
                "n_tokens_a": float(len(set_a)),
                "n_tokens_b": float(len(set_b)),
            },
        )

    def score(self, name_a: str, name_b: str) -> float:
        return self.explain(name_a, name_b).score


# ==========================================================================
# 3. Phonetic + Jaro-Winkler
# ==========================================================================


def _no_phonetic_evidence_available(left_tokens, right_tokens) -> bool:
    """True when phonetics cannot speak at all, as opposed to disagreeing.

    Soundex and Metaphone both return "" for a string with no letters, and
    ``phonetic_agreement`` reports 0.0 for that case -- the same value it
    reports when two well-formed codes simply do not match. Only the first is a
    missing measurement, and only the first justifies ignoring the weight the
    phonetic term would otherwise carry.
    """
    tokens = list(left_tokens) + list(right_tokens)
    if not tokens:
        return False
    return all(not any(ch.isalpha() for ch in token) for token in tokens)


class PhoneticJaroWinkler:
    """Best-of phonetic and character similarity, combined additively.

    The pitch of this algorithm is "be robust to spelling variation", and for
    transliteration it genuinely is. The cost is that phonetic codes are
    deliberately lossy, so unrelated surnames that sound alike land in the same
    bucket. It is included precisely so that cost can be measured rather than
    argued about.
    """

    key = "phonetic_jaro"
    label = "Phonetic + Jaro-Winkler"

    #: Weight of the phonetic component in the final blend.
    PHONETIC_WEIGHT = 0.5

    def explain(self, name_a: str, name_b: str) -> MatchResult:
        left = normalize(name_a)
        right = normalize(name_b)

        reason = _undecidable_reason(left, right)
        if reason == "unsupported_script":
            return MatchResult(UNDECIDABLE_SCORE, {"unsupported_script": 1.0})
        if reason:
            return MatchResult(0.0, {"empty": 1.0})

        # --- phonetic agreement over tokens -------------------------------
        left_tokens, right_tokens = left.tokens, right.tokens
        # Aggregation must be *symmetric*. Averaging "best match for each left
        # token" alone is not: it depends on which name happens to be presented
        # first, and a match decision that flips when the two documents are
        # swapped is not usable no matter how good the underlying codes are.
        # The mean of the two directional means fixes that.
        best_per_left = [
            max((phonetics.phonetic_agreement(a, b) for b in right_tokens), default=0.0)
            for a in left_tokens
        ]
        best_per_right = [
            max((phonetics.phonetic_agreement(b, a) for a in left_tokens), default=0.0)
            for b in right_tokens
        ]
        phonetic_max = max(
            (phonetics.phonetic_agreement(a, b)
             for a in left_tokens for b in right_tokens),
            default=0.0,
        )
        phonetic_mean = (
            0.5 * (sum(best_per_left) / len(best_per_left) + sum(best_per_right) / len(best_per_right))
            if left_tokens and right_tokens else 0.0
        )

        # --- character similarity ------------------------------------------
        # Sorted tokens make the string comparison order-insensitive, so this
        # algorithm does not fail on surname-first documents.
        jw = string_algos.jaro_winkler(
            " ".join(sorted(left.canonical)), " ".join(sorted(right.canonical))
        )
        # A second pass on the original order keeps a bonus for the common case.
        jw_ordered = string_algos.jaro_winkler(left.canonical_text, right.canonical_text)
        jw = max(jw, jw_ordered)

        phonetic_component = 0.5 * phonetic_max + 0.5 * phonetic_mean
        score = self.PHONETIC_WEIGHT * phonetic_component + (1 - self.PHONETIC_WEIGHT) * jw
        # No phonetic evidence available at all -- digit-only strings, where
        # Soundex and Metaphone both return the empty code rather than
        # inventing a match. The blend above would then cap the score at 0.5
        # however identical the two strings are, so four matchers call a
        # byte-identical pair a match and this one called it a coin flip. The
        # character signal is not evidence of anything on its own, but it is
        # strictly better evidence than a component that does not exist.
        if phonetic_component == 0.0 and _no_phonetic_evidence_available(
                left_tokens, right_tokens):
            score = jw

        return MatchResult(
            score=_clamp(score),
            components={
                "phonetic_max": phonetic_max,
                "phonetic_mean": phonetic_mean,
                "jaro_winkler": jw,
                "jaro_winkler_ordered": jw_ordered,
            },
        )

    def score(self, name_a: str, name_b: str) -> float:
        return self.explain(name_a, name_b).score


# ==========================================================================
# 4. Token-aligned weighted scorer
# ==========================================================================

#: How much a token type contributes to the final verdict. The family name
#: carries the most weight because it is the most stable part of an Indian
#: name across documents: a person's given name is far more likely to be
#: abbreviated, expanded, misspelled or replaced by an initial than their
#: family name is to change.
_ROLE_WEIGHTS = {
    "family": 1.00,
    "given": 0.85,
    "middle": 0.45,
}

#: Credit given to a bare initial matching the start of a full token. Deliberately
#: below full credit: "S." is consistent with Suresh, Sunil, Suresh Kumar *and*
#: Sandeep. It is evidence, not proof.
INITIAL_CREDIT = 0.70

#: Credit for a prefix relationship, scaled by how much of the token is covered.
PREFIX_CREDIT = 0.85

#: Multiplier applied to a match that relies on a phonetic collision rather than
#: an orthographic one. Soundex says "Sharma", "Sarma" and "Saxena" are all
#: plausible, so a phonetic-only match is weak evidence.
PHONETIC_DISCOUNT = 0.55

#: Credit for two *different* single-letter initials. Small but non-zero: they
#: are weak evidence, and a hard zero would deny the phonetic and character
#: fallbacks any chance to contribute.
INITIAL_MISMATCH_FLOOR = 0.10

#: Weight of the orthographic character-level signal in the final blend. Kept
#: meaningful so that genuine single-character typos are still recoverable even
#: when the token alignment misses.
CHAR_WEIGHT = 0.25

#: Floor on the multiplier applied by the name-order consistency check.
#: ``ORDER_BONUS_WEIGHT + (1 - ORDER_BONUS_WEIGHT) * order_bonus`` spans
#: [0.90, 1.00], so order agreement moves the score by at most 3%. That is
#: deliberately small, and worth being honest about why: positional token
#: weighting already makes "Kumar Suresh" align family-to-family against
#: "Suresh Kumar", so the explicit order term is nearly redundant. It is kept as
#: a weak tie-breaker, not as a load-bearing signal.
ORDER_BONUS_WEIGHT = 0.90


#: Score returned when a name cannot be read at all. Deliberately *neutral*
#: rather than 0.0: a matcher that cannot compare two strings has no evidence
#: that the people are different, and reporting 0.0 would make an unreadable
#: name look like a confident rejection. 0.5 sits above the auto-reject
#: threshold and below the auto-approve one, so such a pair is routed to human
#: review -- which is the only defensible destination for it.
UNDECIDABLE_SCORE = 0.5


def _undecidable_reason(left, right) -> str | None:
    """Name of the policy that makes this pair undecidable, or ``None``."""
    if left.undecidable or right.undecidable:
        return "unsupported_script"
    if not left.canonical or not right.canonical:
        return "empty"
    return None


def _token_similarity(token_a: str, canon_a: str, token_b: str, canon_b: str,
                      is_initial_a: bool, is_initial_b: bool) -> float:
    """Similarity of one aligned token pair, in [0, 1].

    Ordered from strongest evidence to weakest:

    1. identical canonical tokens (lexicon or generic fold resolved them)
    2. initial-vs-expanded, in either direction
    3. prefix containment
    4. phonetic collision, discounted
    5. raw character similarity
    """
    if canon_a == canon_b and token_a == token_b:
        return 1.0
    if canon_a == canon_b:
        # Resolved by the lexicon even though the raw spellings differ.
        return 0.95

    # Initial vs expanded name.
    if is_initial_a and not is_initial_b:
        return INITIAL_CREDIT if string_algos.is_prefix_of(token_a, token_b) else 0.0
    if is_initial_b and not is_initial_a:
        return INITIAL_CREDIT if string_algos.is_prefix_of(token_b, token_a) else 0.0
    if is_initial_a and is_initial_b:
        if token_a[0] == token_b[0]:
            return INITIAL_CREDIT
        # Two different initials are *absence of evidence*, not evidence of
        # mismatch. A hard 0.0 short-circuits the phonetic and character
        # fallbacks below, which is the opposite of what the ordering above
        # documents as the intent.
        return INITIAL_MISMATCH_FLOOR

    # Prefix containment: "venkat" inside "venkatesh".
    #
    # This is a genuinely dangerous signal in Indian names, and the earlier
    # version of this branch made it worse by using a *step function*: a
    # one-or-two-character extension was discounted heavily while a
    # three-character extension received full credit. A one-character difference
    # between two names was therefore treated as more dangerous than a
    # three-character one, which is incoherent, and the branch was
    # non-monotone in the only variable it depends on.
    #
    # What replaced it is a single smooth quantity: how much of the *longer*
    # token the shorter one accounts for. Credit then falls monotonically as the
    # extension grows, with no threshold to tune:
    #
    #     krishna -> krishnan    7/8    0.74
    #     krishn  -> krishnan    6/8    0.64
    #     venkat  -> venkatesh   5/9    0.57
    #     vijay   -> vijayalaxmi 5/11   0.39
    #
    # This does *not* resolve "krishna"/"krishnan", which is a father and a son
    # in the evaluation set. Nothing in the strings can: "venkat"/"venkatesh" is
    # one person and "krishna"/"krishnan" is two, and they are the same shape.
    # The father/son pair ranks below a genuine same-person variant because of
    # the *other* tokens and the coverage penalty, not because of this branch,
    # and the limitation is documented rather than papered over with a constant.
    if string_algos.is_prefix_of(canon_a, canon_b) or \
       string_algos.is_prefix_of(canon_b, canon_a):
        shorter, longer = (canon_a, canon_b) if len(canon_a) <= len(canon_b) else (canon_b, canon_a)
        shared = string_algos.prefix_ratio(canon_a, canon_b)
        accounted_for = len(shorter) / len(longer)
        return PREFIX_CREDIT * (0.5 * shared + 0.5 * accounted_for)

    # Phonetic collision, heavily discounted.
    agreement = phonetics.phonetic_agreement(canon_a, canon_b)
    if agreement > 0.0:
        return PHONETIC_DISCOUNT * agreement

    # Fall back to character similarity, transposition-aware.
    distance = string_algos.damerau_osa(canon_a, canon_b)
    similarity = 1.0 - distance / max(len(canon_a), len(canon_b), 1)

    # A short token that shares only a couple of characters with a longer one
    # should not collect real credit. "sarma" and "saksena" differ by one edit
    # at Damerau level once a common prefix is stripped, and a naive
    # length-normalised ratio calls that ~0.43 similarity between two surnames
    # that are in fact entirely different. Requiring a meaningful shared prefix
    # for the partial-credit band keeps genuinely confusable pairs
    # ("vijay"/"vijayalaxmi") while refusing credit to unrelated ones.
    #
    # No prefix test is needed: this is reached only when neither token contains
    # the other, and `is_prefix_of` is itself symmetric, so an `and` of two calls
    # would always be True here.
    shared_prefix = string_algos.prefix_ratio(canon_a, canon_b)
    shorter = min(len(canon_a), len(canon_b))
    if shorter >= 4 and shared_prefix * shorter < 3:
        # Less than three characters in common: scale credit down hard rather
        # than dropping it, so the ordering is still informative.
        return similarity * 0.35
    if shorter < 4:
        return similarity * 0.5

    return similarity


def _greedy_align(
    left: NormalizedName, right: NormalizedName
) -> list[tuple[int, int, float]]:
    """Greedily align tokens by descending similarity.

    Greedy is an approximation, not an optimal assignment: on 3 of 200,000
    randomly sampled token sets a full optimum scores about 0.01 higher in total
    weighted similarity. The matrices here are 2-5 tokens and near-diagonal, so
    the gap does not move the final score past the fourth decimal -- but it is
    not exact, and the difference is worth stating rather than claiming
    optimality. Greedy was kept because it keeps the alignment inspectable,
    which matters more here than the fourth decimal.
    """
    left_tokens, right_tokens = left.tokens, right.tokens
    candidates: list[tuple[float, int, int]] = []

    for i, (token_a, canon_a) in enumerate(zip(left_tokens, left.canonical)):
        for j, (token_b, canon_b) in enumerate(zip(right_tokens, right.canonical)):
            similarity = _token_similarity(
                token_a, canon_a, token_b, canon_b,
                i in left.initials, j in right.initials,
            )
            if similarity > 0.0:
                candidates.append((similarity, i, j))

    # Sort by similarity descending, then by position, so ties resolve
    # deterministically and prefer left-to-right order.
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_left: set[int] = set()
    used_right: set[int] = set()
    aligned: list[tuple[int, int, float]] = []
    for similarity, i, j in candidates:
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        aligned.append((i, j, similarity))

    aligned.sort(key=lambda item: item[0])
    return aligned


class TokenAlignedScorer:
    """Role-weighted, order-agnostic, token-aligned scorer.

    The design responds to the specific structure of the problem:

    * **Order.** Handled by weighting tokens by position: the trailing token of
      each name is treated as the family name, so "Kumar Suresh" aligns
      family-to-family against "Suresh Kumar" without scoring both readings and
      taking the better. A residual order term breaks remaining ties. (An earlier
      design scored both readings and took the max; it was dropped because
      positional weighting already covers the case and the second alignment
      doubled the cost for no measurable gain.)
    * **Initials.** A bare initial gets partial credit against any token it
      prefixes, weighted below full credit.
    * **Dropped tokens.** Missing tokens are penalised by *coverage* rather than
      by failing outright, so a dropped middle name costs a little while a
      dropped family name costs a lot.
    * **Weighting.** The family name is weighted highest, because it is the most
      stable part of a name across documents.
    * **Phonetics.** Used only as a weak fallback, discounted by
      :data:`PHONETIC_DISCOUNT`, so it can bridge a spelling gap without being
      able to manufacture a match on its own.

    Known limitation: a longer name containing a shorter one earns only partial
    credit, scaled by length ratio. That is what stops "Krishna" and "Krishnan"
    -- a father and a son in the evaluation set -- from scoring as one surname,
    but it is a blunt instrument: nothing here distinguishes a name that was
    *extended* from one that merely *contains* another.
    """

    key = "token_aligned"
    label = "Token-aligned weighted scorer"

    def explain(self, name_a: str, name_b: str) -> MatchResult:
        left = normalize(name_a)
        right = normalize(name_b)

        reason = _undecidable_reason(left, right)
        if reason == "unsupported_script":
            return MatchResult(UNDECIDABLE_SCORE, {"unsupported_script": 1.0})
        if reason:
            return MatchResult(0.0, {"empty": 1.0})

        aligned = _greedy_align(left, right)
        if not aligned:
            # Report the *full* component set, with honest values. An earlier
            # version returned only two keys here, so every consumer using
            # `.get(name, default)` silently substituted its default -- including
            # 1.0 for `order_bonus`, i.e. "these two names agree perfectly on
            # order" for a pair that shares no tokens at all, and 0.0 for
            # `char_signal`, discarding a real 0.37 similarity. A missing
            # measurement must never default to its best value.
            return MatchResult(0.0, {
                "coverage": 0.0,
                "agreement": 0.0,
                # No tokens aligned, so there is no evidence of order agreement.
                "order_bonus": 0.0,
                "char_signal": string_algos.jaro_winkler(
                    " ".join(sorted(left.canonical)),
                    " ".join(sorted(right.canonical))),
                "n_aligned": 0.0,
                "n_left": float(len(left.tokens)),
                "n_right": float(len(right.tokens)),
            })

        # --- weighted agreement over aligned tokens -----------------------
        # Weights come from the family-name position. Under the given-first
        # reading the trailing token is the family name; the aligned family
        # names are the ones that matter most, and "longest common suffix token"
        # is a good enough proxy when order is ambiguous.
        left_count, right_count = len(left.tokens), len(right.tokens)
        total_weight = 0.0
        matched_weight = 0.0
        weighted_similarity = 0.0

        for i, j, similarity in aligned:
            weight_a = _token_weight(i, left_count)
            weight_b = _token_weight(j, right_count)
            weight = min(weight_a, weight_b)
            total_weight += weight
            matched_weight += weight
            weighted_similarity += weight * similarity

        # Unmatched tokens still count against coverage.
        for i in range(left_count):
            if not any(ai == i for ai, _j, _s in aligned):
                total_weight += _token_weight(i, left_count)
        for j in range(right_count):
            if not any(bj == j for _i, bj, _s in aligned):
                total_weight += _token_weight(j, right_count)

        coverage = matched_weight / total_weight if total_weight else 0.0
        agreement = weighted_similarity / matched_weight if matched_weight else 0.0

        # Order resolution: prefer the reading that aligned the trailing token
        # of each name to the trailing token of the other.
        order_bonus = self._order_bonus(aligned, left_count, right_count)

        # Character-level fallback, so an outright token-alignment miss is not
        # fatal when the raw strings are nearly identical.
        char_signal = string_algos.jaro_winkler(
            " ".join(sorted(left.canonical)), " ".join(sorted(right.canonical))
        )

        # The three factors are combined multiplicatively, not additively.
        # Adding an order bonus to coverage*agreement pushes the sum above 1.0
        # and clamps every order-consistent pair to a perfect score, which
        # destroys the ranking information the threshold search depends on.
        # The order term is deliberately weak (10%): name order is already
        # handled by the positional token weighting, which is what lets
        # "Kumar Suresh" align family-to-family in the first place.
        core = coverage * agreement * (ORDER_BONUS_WEIGHT + (1 - ORDER_BONUS_WEIGHT) * order_bonus)
        score = (1 - CHAR_WEIGHT) * core + CHAR_WEIGHT * char_signal

        return MatchResult(
            score=_clamp(score),
            components={
                "coverage": coverage,
                "agreement": agreement,
                "order_bonus": order_bonus,
                "char_signal": char_signal,
                "n_aligned": float(len(aligned)),
                "n_left": float(left_count),
                "n_right": float(right_count),
            },
        )

    @staticmethod
    def _order_bonus(aligned: list[tuple[int, int, float]],
                     left_count: int, right_count: int) -> float:
        """Reward a reading where the two names agree on name order.

        If the last token of each name aligned to the last token of the other,
        the order readings agree. If instead the first tokens aligned, the two
        names are in opposite order, which is still consistent evidence of a
        match -- so the bonus is smaller, not zero.
        """
        if not aligned or left_count < 2 or right_count < 2:
            return 1.0

        pairs = {(i, j) for i, j, _s in aligned}
        same_order = (left_count - 1, right_count - 1) in pairs
        if same_order:
            return 1.0
        if (0, 0) in pairs:
            # Both names put the same token first; that is evidence they share
            # a family name even though the remaining order differs.
            return 0.85
        return 0.70

    def score(self, name_a: str, name_b: str) -> float:
        return self.explain(name_a, name_b).score


def _token_weight(index: int, count: int) -> float:
    """Weight a token by its position: trailing token is the family name."""
    if count <= 1:
        return _ROLE_WEIGHTS["given"]
    if index == 0:
        return _ROLE_WEIGHTS["given"]
    if index == count - 1:
        return _ROLE_WEIGHTS["family"]
    return _ROLE_WEIGHTS["middle"]


# ==========================================================================
# 5. Learned combiner
# ==========================================================================


class LearnedCombiner:
    """Logistic regression over features extracted from the other matchers.

    Thin wrapper around :class:`name_match.model.LogisticRegression` so that
    the evaluation harness can treat it as just another matcher. The model is
    trained by :mod:`name_match.model` and passed in; nothing is fitted here.
    """

    key = "learned"
    label = "Learned combiner (logistic regression)"

    def __init__(self, model=None) -> None:
        if model is None:
            from .model import LogisticRegression

            model = LogisticRegression()
        self._model = model

    @property
    def model(self):
        return self._model

    def explain(self, name_a: str, name_b: str) -> MatchResult:
        from .features import FEATURE_NAMES, extract_features

        # A failed extraction is not a weak match, it is *no information*. An
        # all-zero feature vector pushed through a trained model still produces
        # sigmoid(bias + weighted mean offsets), which is an arbitrary number
        # that drifts with the model's coefficients. Returning 0.0 keeps the
        # "empty input is never agreement" guarantee that every other matcher
        # makes.
        left = normalize(name_a)
        right = normalize(name_b)
        if left.undecidable or right.undecidable:
            return MatchResult(UNDECIDABLE_SCORE, {"unsupported_script": 1.0})
        if left.is_empty() or right.is_empty():
            return MatchResult(0.0, {"empty": 1.0})

        if self._model.weights and tuple(self._model.feature_names) != FEATURE_NAMES:
            raise ValueError("learned model feature order does not match the extractor; refit the model")
        vector, components = extract_features(name_a, name_b)
        probability = self._model.predict_proba(vector)
        return MatchResult(
            score=_clamp(probability),
            components={**components, **{f"f_{k}": v for k, v in zip(FEATURE_NAMES, vector)}},
        )

    def score(self, name_a: str, name_b: str) -> float:
        return self.explain(name_a, name_b).score


# ==========================================================================
# Registry
# ==========================================================================

#: The four hand-written matchers, i.e. everything except the learned one.
#: Used for cross-validation, and reported alongside the learned arm in
#: the ablation (`python3 -m name_match.cli ablate`).
BASELINE_MATCHERS: tuple[Matcher, ...] = (
    ExactNormalizedMatch(),
    TokenSetJaccard(),
    PhoneticJaroWinkler(),
    TokenAlignedScorer(),
)

#: Everything evaluated, including the learned arm.
ALL_MATCHERS: tuple[Matcher, ...] = (*BASELINE_MATCHERS, LearnedCombiner())

_REGISTRY = {matcher.key: matcher for matcher in ALL_MATCHERS}


def get_matcher(key: str) -> Matcher:
    return _REGISTRY[key]


def matcher_names() -> list[str]:
    return [matcher.key for matcher in ALL_MATCHERS]
