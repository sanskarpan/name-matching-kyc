"""Command-line entry point.

Subcommands, in the order a reviewer should run them::

    python3 -m name_match.cli generate    # write data/name_pairs.csv
    python3 -m name_match.cli train       # fit + write data/model.json
    python3 -m name_match.cli evaluate    # run all algorithms, print tables
    python3 -m name_match.cli report      # write reports/results.{md,json}
    python3 -m name_match.cli all         # all of the above, in order

``all`` is the single command the test suite and the README use.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import Sequence

from . import dataset as dataset_module
from . import evaluate as evaluate_module
from .algorithms import (ALL_MATCHERS, BASELINE_MATCHERS, LearnedCombiner,
                         MatchResult, TokenAlignedScorer, get_matcher)
from .dataset import NamePair, build_dataset, read_csv, write_csv
from .features import FEATURE_NAMES, extract_features
from .model import DEFAULT_MODEL_PATH, LogisticRegression

DEFAULT_REPORT_JSON = os.path.join("reports", "results.json")
DEFAULT_REPORT_MD = os.path.join("reports", "results.md")

#: Cost ratios swept in the sensitivity analysis. 1:1 is included to show what
#: would happen if the two error types were (incorrectly) treated as equal.
COST_RATIOS: tuple[float, ...] = (1.0, 3.0, 5.0, 10.0, 25.0, 50.0, 100.0)

#: Precision the auto-approve cut-off is required to reach. Published per
#: algorithm, and whether it was actually achieved.
BAND_TARGET_PRECISION = 0.99

#: Number of cross-validation folds for the learned model. 5 is the smallest
#: split that leaves each fold large enough to be meaningful on a 494-row set.
CV_FOLDS = 5

#: Recall floor for the recall-constrained table. 0.90 is satisfiable for every
#: algorithm here, which is what makes the table readable -- but it is NOT a
#: like-for-like ranking comparison, and the generated report says so. An
#: algorithm whose scores are too coarse to land on the floor is compared at a
#: *different* achieved recall, and its cost per pair is not comparable. The
#: precision-at-recall table is the comparable one.
RECALL_FLOOR = 0.90

#: Recall levels at which precision is reported. This is the
#: genuinely comparable cross-algorithm metric: every algorithm answers the same
#: question at each level, independent of how coarse its score distribution is.
PRECISION_RECALL_LEVELS: tuple[float, ...] = (0.50, 0.70, 0.90, 0.95, 0.99)


# --------------------------------------------------------------------------
# Cross-validation for the learned model
# --------------------------------------------------------------------------


def kfold_indices(n: int, folds: int = CV_FOLDS) -> list[tuple[list[int], list[int]]]:
    """Deterministic k-fold split, assigned by a stride rather than a shuffle.

    Reproducible without an RNG, which is the whole point: two runs of the
    pipeline must produce identical out-of-fold scores.

    It is *not* stratified in the sense of balancing labels by construction --
    it achieves similar positive rates only because the dataset happens to be
    ordered positives-first, and an earlier version of this function declared
    that intent with two counters it never read. The property is asserted
    directly by ``test_folds_have_similar_positive_rates`` instead of being
    claimed here.
    """
    assignments = [[] for _ in range(folds)]

    for index in range(n):
        assignments[index % folds].append(index)

    return [
        (
            [i for fold in range(folds) if fold != target for i in assignments[fold]],
            assignments[target],
        )
        for target in range(folds)
    ]


def cross_validated_scores(pairs: Sequence[NamePair],
                           folds: int = CV_FOLDS) -> tuple[list[float], dict]:
    """Out-of-fold probabilities for every pair.

    Every score comes from a model that never saw that row, which is what makes
    the learned arm comparable to the hand-written ones: scored on its own
    training data it would report an operating threshold tuned on the same rows
    it is evaluated on, which is optimistic in a way that does not survive
    contact with production.

    What this does *not* do, and what no claim in the report may imply: select
    the operating threshold. The threshold is chosen once, afterwards, from the
    pooled out-of-fold scores of the whole dataset (see :func:`analyse`). That
    is the standard "cross-validate the score, then tune the threshold"
    arrangement; it does leak threshold-selection information across folds, and
    a nested scheme would not. With a threshold chosen from this many rows and
    reported alongside its bootstrap interval, the leak is small -- but it is a
    leak, and the honest description is "out-of-fold scores, threshold tuned on
    the pooled scores", not "the threshold was chosen on other folds".
    """
    matrix = [extract_features(row.name_a, row.name_b)[0] for row in pairs]
    labels = [row.label for row in pairs]

    scores = [0.0] * len(pairs)
    fold_losses: list[float] = []

    for train_idx, test_idx in kfold_indices(len(pairs), folds):
        model = LogisticRegression().fit(
            [matrix[i] for i in train_idx],
            [labels[i] for i in train_idx],
            FEATURE_NAMES,
        )
        fold_losses.append(model.final_loss)
        for i in test_idx:
            scores[i] = model.predict_proba(matrix[i])

    full_model = LogisticRegression().fit(matrix, labels, FEATURE_NAMES)
    info = {
        "folds": folds,
        "fold_train_losses": fold_losses,
        "full_model_loss": full_model.final_loss,
        "model": full_model,
    }
    return scores, info


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _fmt(value: float, digits: int = 3) -> str:
    if value != value:  # NaN
        return "n/a"
    return f"{value:.{digits}f}"


def build_score_sets(pairs: Sequence[NamePair],
                     use_cv: bool = True) -> tuple[list[evaluate_module.ScoreSet], dict]:
    """Score every matcher over the dataset.

    The learned matcher always uses out-of-fold probabilities, even when a
    fully-trained model is available, so it is compared on equal footing with
    the hand-written algorithms.
    """
    score_sets: list[evaluate_module.ScoreSet] = []
    extras: dict = {}

    for matcher in BASELINE_MATCHERS:
        scores = [matcher.score(row.name_a, row.name_b) for row in pairs]
        score_sets.append(evaluate_module.ScoreSet(
            key=matcher.key, label=matcher.label,
            scores=scores, labels=[row.label for row in pairs], rows=list(pairs),
        ))

    if use_cv:
        learned_scores, info = cross_validated_scores(pairs)
        extras["cv"] = {k: v for k, v in info.items() if k != "model"}
    else:
        full_model = LogisticRegression().fit(
            [extract_features(r.name_a, r.name_b)[0] for r in pairs],
            [r.label for r in pairs],
            FEATURE_NAMES,
        )
        learned_scores = [full_model.predict_proba(extract_features(r.name_a, r.name_b)[0])
                          for r in pairs]
        extras["cv"] = {"folds": 0, "full_model_loss": full_model.final_loss, "model": full_model}

    score_sets.append(evaluate_module.ScoreSet(
        key="learned", label="Learned combiner (logistic regression, out-of-fold)",
        scores=learned_scores, labels=[row.label for row in pairs], rows=list(pairs),
    ))
    return score_sets, extras


def analyse(score_sets: Sequence[evaluate_module.ScoreSet],
            cost_fp: float = evaluate_module.DEFAULT_COST_FP,
            cost_fn: float = evaluate_module.DEFAULT_COST_FN) -> dict:
    """Full metric bundle for one cost ratio."""
    results: dict[str, dict] = {}

    for score_set in score_sets:
        optimal = evaluate_module.find_operating_point(
            score_set.scores, score_set.labels, cost_fp, cost_fn)
        zero_fp = evaluate_module.zero_fp_point(score_set.scores, score_set.labels)
        constrained = evaluate_module.find_operating_point_at_recall(
            score_set.scores, score_set.labels, RECALL_FLOOR, cost_fp, cost_fn)
        report = evaluate_module.band_report(score_set.scores, score_set.labels,
                                             cost_fp, cost_fn)

        results[score_set.key] = {
            "label": score_set.label,
            "pr_auc": evaluate_module.pr_auc(score_set.scores, score_set.labels),
            "roc_auc": evaluate_module.roc_auc(score_set.scores, score_set.labels),
            "precision_at_recall": {
                str(level): evaluate_module.interpolated_precision_at_recall(
                    score_set.scores, score_set.labels, level)
                for level in PRECISION_RECALL_LEVELS
            },
            "optimal": optimal.as_row(),
            "zero_fp": zero_fp.as_row(),
            "at_recall_floor": constrained.as_row(),
            "bands": evaluate_module.three_band_split(
                score_set.scores, score_set.labels,
                report["reject_below"], report["approve_at_or_above"]),
            "band_thresholds": {"low": report["reject_below"],
                                "high": report["approve_at_or_above"]},
            "band_detail": report,
            "per_category": evaluate_module.per_category_error(score_set, optimal.threshold),
            "score_set": score_set,
        }

    return results


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def _json_safe(value: object) -> object:
    """Recursively replace non-finite floats with ``None``.

    Python's ``json`` emits bare ``NaN`` and ``Infinity`` tokens by default.
    Those are not valid JSON per RFC 8259: ``jq`` and Python accept them, strict
    parsers do not, and this file is advertised as machine-readable output for
    verification. The mapping is lossless in meaning -- a non-finite value says
    "undefined at this operating point", which is precisely what ``null`` means
    in JSON.
    """
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _require_pairs(pairs: list) -> list:
    """Fail loudly on an empty dataset rather than deep inside the model."""
    if not pairs:
        raise SystemExit(
            "no pairs to evaluate. Run `python3 -m name_match.cli generate` "
            "first, or check that --data points at a populated CSV.")
    return pairs


def cmd_generate(args: argparse.Namespace) -> int:
    pairs = build_dataset()
    path = write_csv(pairs, args.data)
    positives = sum(1 for p in pairs if p.label == 1)
    print(f"Wrote {len(pairs)} labelled pairs to {path}")
    print(f"  positives: {positives}   negatives: {len(pairs) - positives}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    pairs = _require_pairs(read_csv(args.data))
    matrix = [extract_features(row.name_a, row.name_b)[0] for row in pairs]
    labels = [row.label for row in pairs]
    model = LogisticRegression().fit(matrix, labels, FEATURE_NAMES)
    path = model.save(args.model)
    print(f"Trained on {len(pairs)} pairs -> {path}")
    print(f"  training log-loss: {model.final_loss:.4f}")
    print("  standardised coefficients (largest first):")
    for name, weight in model.coefficient_report()[:10]:
        print(f"    {name:24} {weight:+.3f}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    pairs = _require_pairs(
        read_csv(args.data) if os.path.exists(args.data) else build_dataset())
    score_sets, extras = build_score_sets(pairs)
    results = analyse(score_sets)

    print()
    print("=" * 96)
    print(f"PRIMARY METRIC: cost at the cost-optimal threshold, "
          f"c_FN = {evaluate_module.DEFAULT_COST_FN:.0f} x c_FP")
    print("=" * 96)
    print(_markdown_table(
        ["algorithm", "cost/pair", "FP", "FN", "precision", "recall", "PR-AUC", "ROC-AUC"],
        [
            [
                results[s.key]["label"],
                _fmt(results[s.key]["optimal"]["cost_per_pair"]),
                results[s.key]["optimal"]["fp"],
                results[s.key]["optimal"]["fn"],
                _fmt(results[s.key]["optimal"]["precision"]),
                _fmt(results[s.key]["optimal"]["recall"]),
                _fmt(results[s.key]["pr_auc"]),
                _fmt(results[s.key]["roc_auc"]),
            ]
            for s in score_sets
        ],
    ))

    print()
    print("PRECISION AT A FIXED RECALL (PR envelope, nothing interpolated): "
          "the comparable metric.")
    print("Every algorithm answers the same question at each recall level, so")
    print("unlike the cost table this cannot be flattered by where a score")
    print("distribution happens to bottom out. Higher is better.")
    print("-" * 96)
    print(_markdown_table(
        ["algorithm"] + [f"P@R={r:.2f}" for r in PRECISION_RECALL_LEVELS] + ["PR-AUC"],
        [
            [results[s.key]["label"]]
            + [_fmt(results[s.key]["precision_at_recall"][f"{r}"]) for r in PRECISION_RECALL_LEVELS]
            + [_fmt(results[s.key]["pr_auc"])]
            for s in score_sets
        ],
    ))

    print()
    print("AT A FIXED RECALL FLOOR (recall >= %.2f), matched on achieved recall"
          % RECALL_FLOOR)
    print("-" * 96)
    print(_markdown_table(
        ["algorithm", "cost/pair", "FP", "recall", "precision"],
        [
            [
                results[s.key]["label"],
                _fmt(results[s.key]["at_recall_floor"]["cost_per_pair"]),
                results[s.key]["at_recall_floor"]["fp"],
                _fmt(results[s.key]["at_recall_floor"]["recall"]),
                _fmt(results[s.key]["at_recall_floor"]["precision"]),
            ]
            for s in score_sets
        ],
    ))
    print("  Note: the achieved recall differs between algorithms because their")
    print("  score distributions have different granularity. Where an algorithm")
    print("  cannot land near the floor, it is compared at the most selective")
    print("  threshold it does have, which is why P@R above is the primary")
    print("  cross-algorithm metric and this table is supporting detail.")

    print()
    print("ZERO-FALSE-POSITIVE OPERATING POINT (what a bank would pick if an")
    print("incorrect auto-approval were unacceptable)")
    print("-" * 96)
    print(_markdown_table(
        ["algorithm", "threshold", "FP", "FN", "recall"],
        [
            [
                results[s.key]["label"],
                _fmt(results[s.key]["zero_fp"]["threshold"]),
                results[s.key]["zero_fp"]["fp"],
                results[s.key]["zero_fp"]["fn"],
                _fmt(results[s.key]["zero_fp"]["recall"]),
            ]
            for s in score_sets
        ],
    ))

    print()
    print("COST SENSITIVITY: cost per pair at the optimal threshold, by c_FN/c_FP ratio")
    print("-" * 96)
    sweep_rows = []
    for ratio in COST_RATIOS:
        row: list[object] = [f"1 : {ratio:.0f}"]
        for score_set in score_sets:
            confusion = evaluate_module.find_operating_point(
                score_set.scores, score_set.labels, 1.0, ratio)
            row.append(_fmt(confusion.cost_per_pair(1.0, ratio)))
        sweep_rows.append(row)
    print(_markdown_table(["c_FN : c_FP"] + [results[s.key]["label"] for s in score_sets],
                         sweep_rows))

    print()
    print("THREE-BAND SPLIT for the recommended algorithm")
    print("-" * 96)
    best_key = min(results, key=lambda k: results[k]["optimal"]["cost_per_pair"])
    best = results[best_key]
    bands = best["bands"]
    detail = best["band_detail"]
    print(f"  algorithm: {best['label']}")
    print(f"  auto-approve  score >= {detail['raw_approve_at_or_above']:.4f}   "
          f"{bands['approve_n']:>4} pairs ({bands['approve_pct']:5.1f}%), "
          f"{_fmt(bands['approve_accuracy'])} of which really are matches"
          + ("" if detail["approve_meets_target"]
             else f"   [does NOT meet the {BAND_TARGET_PRECISION:.0%} target]"))
    print(f"  manual review  {detail['raw_reject_below']:.4f} <= score < "
          f"{detail['raw_approve_at_or_above']:.4f}   {bands['review_n']:>4} pairs "
          f"({bands['review_pct']:5.1f}%)")
    print(f"  auto-reject    score < {detail['raw_reject_below']:.4f}   "
          f"{bands['reject_n']:>4} pairs ({bands['reject_pct']:5.1f}%), "
          f"{_fmt(bands['reject_accuracy'])} of which really are non-matches")
    if detail["bands_collapsed"]:
        print("  NOTE: the precision cut-off and the zero-false-negative cut-off cross, "
              "so there is no score range that is simultaneously safe to auto-approve "
              "and safe to auto-reject. Everything in between must go to review.")
    if not detail["three_band_pipeline_usable"]:
        empty = []
        if detail["approve_band_empty"]:
            empty.append("approve")
        if detail["reject_band_empty"]:
            empty.append("reject")
        print(f"  NOTE: the {' and '.join(empty)} band is empty -- nothing reaches it, "
              "so it never fires. This is a two-band pipeline, not a three-band one.")

    print()
    print("PER-CATEGORY ERRORS for the recommended algorithm")
    print(f"at its cost-optimal threshold {best['optimal']['threshold']:.6g} "
          f"(recall {best['optimal']['recall']:.3f}) -- a different operating "
          f"point from the recall-floored table in results.md")
    print("-" * 96)
    print(_markdown_table(
        ["category", "n", "pos", "neg", "FN", "FP", "FN rate", "FP rate"],
        [
            [row["category"], row["n"], row["positives"], row["negatives"],
             row["false_negatives"], row["false_positives"],
             _fmt(row["fn_rate"]), _fmt(row["fp_rate"])]
            for row in best["per_category"]
        ],
    ))

    print()
    print("PAIRED BOOTSTRAP: recommended vs runner-up (2000 resamples, paired)")
    print("-" * 96)
    ranked = sorted(results.items(), key=lambda kv: kv[1]["optimal"]["cost_per_pair"])
    if len(ranked) >= 2:
        (key_a, res_a), (key_b, res_b) = ranked[0], ranked[1]
        set_a = res_a["score_set"]
        set_b = res_b["score_set"]
        delta = evaluate_module.bootstrap_cost_delta(
            set_a.scores, set_a.labels, set_b.scores,
            res_a["optimal"]["threshold"], res_b["optimal"]["threshold"],
        )
        print(f"  A = {res_a['label']}")
        print(f"  B = {res_b['label']}")
        print(f"  mean cost delta (B - A, per pair): {delta['mean_delta']:+.4f}")
        print(f"  95% CI: [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}]")
        print(f"  P(A cheaper than B) = {delta['p_a_cheaper']:.3f}")
        # delta = cost(B) - cost(A), so positive means A is cheaper.
        if delta["ci_low"] > 0:
            verdict = "A is significantly cheaper; the gap survives resampling"
        elif delta["ci_high"] < 0:
            verdict = "B is significantly cheaper; the headline ranking is wrong"
        else:
            verdict = ("NOT significant: the 95% CI includes zero, so on this "
                       "dataset the two are effectively tied")
        print(f"  verdict: {verdict}")

    return 0


def _cross_algorithm_error_tables(score_sets: Sequence[evaluate_module.ScoreSet],
                                  ordered_keys: Sequence[str],
                                  recall_floor: float = RECALL_FLOOR
                                  ) -> tuple[list[dict], list[dict]]:
    """Per-category FP and FN counts for every algorithm at a common operating point.

    Every algorithm is placed at the most selective threshold that still clears
    the recall floor, then its errors are broken down by dataset category. The
    per-category error counts attached to each *cost-optimal* threshold are not
    comparable across algorithms, because those thresholds sit at different
    points in each score distribution -- comparing them would show differences
    in where an algorithm's scores happen to fall rather than differences in
    quality.
    """
    thresholds: dict[str, float] = {}
    for score_set in score_sets:
        feasible = [
            evaluate_module.cost_at(score_set.scores, score_set.labels, threshold)
            for threshold in evaluate_module._threshold_grid(score_set.scores)
        ]
        feasible = [c for c in feasible if c.recall + 1e-9 >= recall_floor]
        if feasible:
            lowest_recall = min(c.recall for c in feasible)
            matched = [c for c in feasible if c.recall <= lowest_recall + 1e-9]
            thresholds[score_set.key] = min(
                matched, key=lambda c: (c.fp, -c.threshold)).threshold
        else:
            thresholds[score_set.key] = 1.0

    categories = sorted({
        getattr(row, "category", "unknown")
        for score_set in score_sets for row in score_set.rows
    })

    false_positives: list[dict] = []
    false_negatives: list[dict] = []

    for category in categories:
        fp_row: dict = {"category": category, "fp_by_algorithm": []}
        fn_row: dict = {"category": category, "fn_by_algorithm": []}

        for key in ordered_keys:
            score_set = next(s for s in score_sets if s.key == key)
            threshold = thresholds[key]
            fp = fn = 0
            for score, label, row in zip(score_set.scores, score_set.labels,
                                         score_set.rows):
                if getattr(row, "category", "unknown") != category:
                    continue
                if label == 0 and score >= threshold:
                    fp += 1
                elif label == 1 and score < threshold:
                    fn += 1
            fp_row["fp_by_algorithm"].append(fp)
            fn_row["fn_by_algorithm"].append(fn)

        false_positives.append(fp_row)
        false_negatives.append(fn_row)

    return false_positives, false_negatives


def cmd_report(args: argparse.Namespace) -> int:
    pairs = _require_pairs(
        read_csv(args.data) if os.path.exists(args.data) else build_dataset())
    score_sets, extras = build_score_sets(pairs)
    results = analyse(score_sets)
    if not results:
        raise SystemExit("no algorithms were scored; nothing to report")

    # Sensitivity sweep across cost ratios.
    sweep: dict[str, dict[str, float]] = {}
    for ratio in COST_RATIOS:
        per_algorithm = {}
        for score_set in score_sets:
            confusion = evaluate_module.find_operating_point(
                score_set.scores, score_set.labels, 1.0, ratio)
            per_algorithm[score_set.key] = {
                "cost_per_pair": confusion.cost_per_pair(1.0, ratio),
                "fp": confusion.fp,
                "fn": confusion.fn,
                "threshold": confusion.threshold,
            }
        sweep[f"1:{ratio:.0f}"] = per_algorithm

    best_key = min(results, key=lambda k: results[k]["optimal"]["cost_per_pair"])
    ranked = sorted(results.items(), key=lambda kv: kv[1]["optimal"]["cost_per_pair"])
    bootstrap = None
    if len(ranked) >= 2:
        (key_a, res_a), (key_b, res_b) = ranked[0], ranked[1]
        bootstrap = {
            "a": key_a, "b": key_b,
            **evaluate_module.bootstrap_cost_delta(
                res_a["score_set"].scores, res_a["score_set"].labels,
                res_b["score_set"].scores,
                res_a["optimal"]["threshold"], res_b["optimal"]["threshold"],
            ),
        }

    # -- cross-comparison per-category tables ------------------------------
    ordered_keys = [key for key, _res in ranked]
    per_category_fp, per_category_fn = _cross_algorithm_error_tables(
        score_sets, ordered_keys)

    payload = {
        "dataset": {
            "n_pairs": len(pairs),
            "n_positive": sum(1 for p in pairs if p.label == 1),
            "n_negative": sum(1 for p in pairs if p.label == 0),
            "categories": sorted({p.category for p in pairs}),
        },
        "metric": {
            "name": "asymmetric misclassification cost per pair",
            "cost_fp": evaluate_module.DEFAULT_COST_FP,
            "cost_fn": evaluate_module.DEFAULT_COST_FN,
        },
        "algorithms": {
            key: {
                "label": res["label"],
                "pr_auc": res["pr_auc"],
                "roc_auc": res["roc_auc"],
                "precision_at_recall": res["precision_at_recall"],
                "optimal": res["optimal"],
                "zero_fp": res["zero_fp"],
                "at_recall_floor": res["at_recall_floor"],
                "bands": res["bands"],
                "band_thresholds": res["band_thresholds"],
                "band_detail": res["band_detail"],
                "per_category": res["per_category"],
            }
            for key, res in results.items()
        },
        # Keyed by algorithm, not positional. `json.dump(sort_keys=True)` sorts
        # the `algorithms` mapping alphabetically, so a bare list in the
        # cost-ranked column order would be read against the wrong algorithm by
        # anyone zipping the two together -- silently attributing, say, 40 false
        # positives to the algorithm that actually had 1.
        "cross_algorithm_errors_at_recall_floor": {
            "recall_floor": RECALL_FLOOR,
            "algorithm_order": ordered_keys,
            "false_positives": {row["category"]: dict(zip(ordered_keys,
                                                           row["fp_by_algorithm"]))
                                for row in per_category_fp},
            "false_negatives": {row["category"]: dict(zip(ordered_keys,
                                                           row["fn_by_algorithm"]))
                                for row in per_category_fn},
        },
        "cost_sweep": sweep,
        "bootstrap_ranking_comparison": bootstrap,
        "cross_validation": extras.get("cv", {}),
        "recommended": best_key,
    }

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w", encoding="utf-8", newline="\n") as handle:
        # `allow_nan=False` rejects the bare `NaN` token Python emits by default,
        # which is not valid JSON per RFC 8259. Floats are sanitised *before*
        # serialisation rather than in a `default=` hook -- `default` only fires
        # for unserialisable types, so a float NaN reaches the encoder directly
        # and would raise. `Confusion.precision` returns NaN for a threshold
        # that predicts nothing, so this path is live.
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")

    md = render_markdown(payload, pairs, results, sweep, bootstrap,
                         per_category_fp, per_category_fn)
    os.makedirs(os.path.dirname(args.markdown) or ".", exist_ok=True)
    with open(args.markdown, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(md)

    print(f"Wrote {args.json} and {args.markdown}")
    return 0


def _reversals(rows: Sequence[dict[str, float]],
               keys: Sequence[str]) -> set[str]:
    """Algorithms whose standing against another algorithm strictly inverts.

    `rows` holds one dict of comparable values per condition -- a recall level
    or a cost ratio. A pair counts as reversing only when the strict order
    flips between two conditions.

    This is deliberately not a comparison of list positions, and not a "number
    of algorithms strictly better" rank either. An algorithm that ends level
    with the one above it has tied, not overtaken it; both of those schemes
    report the trailing member of every tie as having moved. Reversal is the
    only formulation here that means what "these two changed places" says.
    """
    flipped: set[str] = set()
    for a, b in itertools.combinations(keys, 2):
        signs = set()
        for row in rows:
            if row[a] > row[b]:
                signs.add(1)
            elif row[a] < row[b]:
                signs.add(-1)
            else:
                signs.add(0)
        if len(signs - {0}) > 1:
            flipped.update((a, b))
    return flipped


def render_markdown(payload: dict, pairs: Sequence[NamePair],
                    results: dict, sweep: dict, bootstrap: dict | None,
                    per_category_fp: list[dict] | None = None,
                    per_category_fn: list[dict] | None = None) -> str:
    """Render the human-readable results report."""
    from collections import Counter

    per_category_fp = per_category_fp or []
    per_category_fn = per_category_fn or []

    lines: list[str] = []
    lines.append("# Results: name matching across identity documents")
    lines.append("")
    lines.append("Generated by `python3 -m name_match.cli all`. "
                 "Do not edit by hand; re-run the command instead.")
    lines.append("")

    # -- dataset summary --------------------------------------------------
    lines.append("## Dataset")
    lines.append("")
    positives = sum(1 for p in pairs if p.label == 1)
    lines.append(f"{len(pairs)} labelled pairs: **{positives} positives**, "
                 f"**{len(pairs) - positives} negatives**, derived from "
                 f"{len({p.person_a for p in pairs} | {p.person_b for p in pairs})} distinct identities.")
    lines.append("")
    category_counts = Counter(p.category for p in pairs)
    lines.append(_markdown_table(
        ["category", "pairs", "positives", "negatives"],
        [
            [category, category_counts[category],
             sum(1 for p in pairs if p.category == category and p.label == 1),
             sum(1 for p in pairs if p.category == category and p.label == 0)]
            for category in sorted(category_counts)
        ],
    ))
    lines.append("")

    # -- headline ---------------------------------------------------------
    lines.append("## Headline: cost per pair at the cost-optimal threshold")
    lines.append("")
    lines.append(f"Cost function `C = {evaluate_module.DEFAULT_COST_FP:.0f}*FP + "
                 f"{evaluate_module.DEFAULT_COST_FN:.0f}*FN`, minimised over the threshold. "
                 "Lower is better; this is the metric the recommendation is based on.")
    lines.append("")
    ordered = sorted(results.items(), key=lambda kv: kv[1]["optimal"]["cost_per_pair"])
    lines.append(_markdown_table(
        ["rank", "algorithm", "cost/pair", "FP", "FN", "precision", "recall", "F1", "PR-AUC"],
        [
            [index, res["label"],
             _fmt(res["optimal"]["cost_per_pair"]),
             res["optimal"]["fp"], res["optimal"]["fn"],
             _fmt(res["optimal"]["precision"]), _fmt(res["optimal"]["recall"]),
             _fmt(res["optimal"]["f1"]), _fmt(res["pr_auc"])]
            for index, (_key, res) in enumerate(ordered, start=1)
        ],
    ))
    optimal_recalls = {res_key: res["optimal"]["recall"]
                       for res_key, res in ordered}
    at_full = [res["label"] for res_key, res in ordered
               if res["optimal"]["recall"] >= 1.0 - 1e-9]
    short = [(res["label"], optimal_recalls[res_key]) for res_key, res in ordered
             if res["optimal"]["recall"] < 1.0 - 1e-9]
    short_list = ", ".join(f"{label} at {recall:.3f}" for label, recall in short)
    lines.append("")
    if at_full and short:
        verb = "reaches" if len(at_full) == 1 else "reach"
        lines.append(
            f"Note the shape of this table. At `c_FN = 25` a missed match is expensive "
            f"enough that lowering the threshold keeps paying, so the cost-optimal "
            f"point sits at or near full recall -- but not exactly at it. "
            f"{len(at_full)} of {len(ordered)} algorithms {verb} recall 1.000 "
            f"({', '.join(at_full)}); the rest stop short ({short_list}). The "
            f"recall column is in the table, and it is why the cost column alone "
            f"cannot be read as a ranking: each algorithm is priced at a slightly "
            f"different operating point. That is what the next two tables exist to "
            f"fix."
        )
    elif at_full:
        lines.append(
            f"Note the shape of this table: at `c_FN = 25` the cost-optimal point "
            f"sits at full recall for every algorithm, because lowering the threshold "
            f"is always worth it when a missed match costs "
            f"{evaluate_module.DEFAULT_COST_FN:.0f}x a false "
            f"one. The column is therefore a clean false-positive count and only a "
            f"weak *ranking* comparison, which is why the next two tables exist."
        )
    else:
        lines.append(
            "Note the shape of this table: at `c_FN = 25` the cost-optimal point "
            "does not reach full recall for any algorithm, and each stops at a "
            "different one (see the recall column). The cost column is therefore not "
            "a like-for-like comparison, which is why the next two tables exist."
        )
    lines.append("")
    lines.append("Thresholds:")
    lines.append("")
    lines.append(_markdown_table(
        ["algorithm", "optimal threshold", "zero-FP threshold"],
        [[res["label"], _fmt(res["optimal"]["threshold"]), _fmt(res["zero_fp"]["threshold"])]
         for _key, res in ordered],
    ))
    lines.append("")

    # -- precision at recall ----------------------------------------------
    lines.append("## Precision at a fixed recall (the comparable metric)")
    lines.append("")
    lines.append(
        "This is the cross-algorithm comparison to trust, and it exists because the "
        "cost table above cannot be read as a ranking on its own. At "
        f"`c_FN = {evaluate_module.DEFAULT_COST_FN:.0f}` each algorithm's "
        "cost-optimal threshold sits wherever that cost function puts it -- as the "
        "recall column above shows, at five different operating points rather than "
        "one -- so the costs are not comparable to each other."
    )
    lines.append("")
    lines.append(
        "Precision at a common recall level removes that problem: every algorithm "
        "answers the identical question, regardless of how coarse its scores are. The "
        "value is the standard PR envelope `P(R) = max{ precision(r) : r >= R }`, so "
        "two properties hold by construction. Every published value is attained by "
        "some real threshold -- nothing is interpolated into existence -- and the row "
        "is non-increasing in recall, because demanding more recall can only admit "
        "more items and therefore only add false positives."
    )
    lines.append("")
    lines.append(_markdown_table(
        ["algorithm"] + [f"P@R={level:.2f}" for level in PRECISION_RECALL_LEVELS]
        + ["PR-AUC"],
        [
            [res["label"]]
            + [_fmt(res["precision_at_recall"][f"{level}"])
               for level in PRECISION_RECALL_LEVELS]
            + [_fmt(res["pr_auc"])]
            for _key, res in ordered
        ],
    ))
    lines.append("")
    widest_level = max(
        (f"{level}" for level in PRECISION_RECALL_LEVELS),
        key=lambda level: (
            results["learned"]["precision_at_recall"][level]
            - results["token_jaccard"]["precision_at_recall"][level]
        ),
    )
    lines.append(
        f"The `P@R=0.99` column is the operationally interesting one: it is the "
        f"precision available at the recall level a KYC pipeline needs if missing a "
        f"genuine customer is expensive. The learned combiner's margin over the "
        f"standard token-set baseline is widest at P@R={widest_level}."
    )
    # Which algorithms change relative position across the published recall
    # levels is a property of the data, so it is computed rather than asserted.
    keys = [res_key for res_key, _ in ordered]

    crossing = [
        key for key in keys
        if key in _reversals(
            [{k: results[k]["precision_at_recall"][f"{level}"] for k in keys}
             for level in PRECISION_RECALL_LEVELS],
            keys)]
    if crossing:
        lines.append("")
        lines.append(
            "Note that the field does not settle into a single order across these "
            "recall levels. These algorithms change position somewhere in the "
            "table: "
            + ", ".join(dict(ordered)[key]["label"] for key in crossing)
            + ". An exact tie counts as a tie, not as one overtaking the "
            "other. A gap that small should not decide which algorithm you "
            "ship."
        )
    lines.append("")

    # -- recall-floored comparison ----------------------------------------
    lines.append(f"## Cost at a matched recall floor (recall >= {RECALL_FLOOR:.2f})")
    lines.append("")
    lines.append(
        "Supporting detail for the table above. Each algorithm is placed at the most "
        "selective threshold that still clears the recall floor. The achieved recall "
        "differs between rows because score granularity differs -- an algorithm that "
        "emits few distinct scores cannot land exactly on the floor -- so read the FP "
        "column alongside it, and use the P@R table for the actual ranking."
    )
    lines.append("")
    lines.append(_markdown_table(
        ["algorithm", "cost/pair", "FP", "recall", "precision", "threshold"],
        [
            [res["label"], _fmt(res["at_recall_floor"]["cost_per_pair"]),
             res["at_recall_floor"]["fp"], _fmt(res["at_recall_floor"]["recall"]),
             _fmt(res["at_recall_floor"]["precision"]),
             _fmt(res["at_recall_floor"]["threshold"])]
            for _key, res in ordered
        ],
    ))
    lines.append("")

    # -- zero FP ----------------------------------------------------------
    lines.append("## The cost of insisting on zero false positives")
    lines.append("")
    lines.append("A false positive here is not a rounding error: it is a document "
                 "that does not belong to the applicant being accepted as if it did. "
                 "This table is the price of a hard no-false-positive policy.")
    lines.append("")
    lines.append(_markdown_table(
        ["algorithm", "threshold", "FP", "FN", "recall", "precision", "cost/pair (c_FN=25)"],
        [
            [res["label"], _fmt(res["zero_fp"]["threshold"]), res["zero_fp"]["fp"],
             res["zero_fp"]["fn"], _fmt(res["zero_fp"]["recall"]),
             _fmt(res["zero_fp"]["precision"]),
             _fmt(res["zero_fp"]["cost_per_pair"])]
            for _key, res in ordered
        ],
    ))
    lines.append("")
    # "No zero-FP threshold exists" means no threshold *in the score range*
    # achieves it -- not that the published (degenerate) threshold has an FP,
    # because the degenerate threshold is above every score by construction.
    no_zero_fp = []
    for _key, _res in ordered:
        _ss = _res["score_set"]
        attainable = any(
            evaluate_module.cost_at(_ss.scores, _ss.labels, _t).fp == 0
            and evaluate_module.cost_at(_ss.scores, _ss.labels, _t).predicted > 0
            for _t in evaluate_module._threshold_grid(_ss.scores)
        )
        if not attainable:
            no_zero_fp.append(_key)
    lines.append(
        "The last column is the whole argument against shipping a hard zero-FP "
        "policy on name evidence alone. Read it together with the `recall` column, "
        "because the two move together: a zero-false-positive threshold has to sit "
        "at the bottom of the positive score distribution, so every genuine match "
        "it lets through is one it also had to catch, and the recall it achieves is "
        "the price."
    )
    if no_zero_fp:
        lines.append("")
        lines.append(
            f"**{len(no_zero_fp)} of {len(ordered)} algorithms have no zero-false-positive "
            f"threshold at all.** Not an expensive trade-off -- an *infeasible* one: "
            f"some negative in the dataset outscores every positive, so no cut-off "
            f"separates them. Those algorithms are marked `FP = 0` at the degenerate "
            f"reject-everything threshold, which is the only FP-free point they have, "
            f"and it has recall 0.000 by construction. The distinction that matters "
            f"for a policy is *zero false positives among pairs that reach manual "
            f"review*, which is achievable; *zero false positives overall* is not, "
            f"as long as two different people can share a name."
        )
    lines.append("")
    lines.append("")

    # -- sweep ------------------------------------------------------------
    lines.append("## Cost sensitivity to the c_FN : c_FP ratio")
    lines.append("")
    lines.append(
        "Every algorithm is measured with the same cost function at seven different "
        "cost ratios, from treating the two error types as equally bad to treating a "
        "missed match as a hundred times worse. The purpose is to show how much the "
        "conclusion depends on the specific ratio quoted above."
    )
    lines.append("")
    keys = [res_key for res_key, _ in ordered]

    by_ratio = [{key: values[key]["cost_per_pair"] for key in keys}
                for values in sweep.values()]
    winners = {min(row, key=lambda k: (row[k], k)) for row in by_ratio}
    flipped = _reversals(by_ratio, keys)
    distinct = {tuple(sorted(row.items(), key=lambda kv: kv[1]))
                for row in by_ratio}
    if len(winners) == 1:
        winner_label = dict(ordered)[next(iter(winners))]["label"]
        lines.append(
            f"The cheapest algorithm is {winner_label} at every ratio in the sweep, "
            "so the recommendation does not rest on the choice of "
            f"{evaluate_module.DEFAULT_COST_FN:.0f}. What the ratio changes is "
            "*which threshold* is optimal -- the operating point, not the identity "
            "of the winner."
        )
    else:
        lines.append(
            "**The cheapest algorithm is not the same at every ratio**: "
            + ", ".join(sorted(dict(ordered)[k]["label"] for k in winners))
            + f" each win somewhere in the sweep. Any recommendation that depends "
            f"on the choice of `c_FN = {evaluate_module.DEFAULT_COST_FN:.0f}` must "
            f"therefore name the ratio it assumes."
        )
    if len(distinct) == 1:
        lines.append("")
        lines.append(
            "The full ordering is stable across the sweep as well, which is a "
            "stronger result than a stable winner: nothing in the tail ranking is "
            "an artefact of the ratio either."
        )
    else:
        moved = sorted(key for key in keys if key in flipped)
        lines.append("")
        lines.append(
            "The *full* ordering is not stable, which is worth stating plainly "
            "rather than rounding to \"the ranking holds\". These algorithms "
            "change position somewhere in the sweep: "
            + ", ".join(dict(ordered)[key]["label"] for key in moved)
            + ". Only the leading algorithm is safe to rank on; the middle of the "
            "field is not separable at a ratio you have not been told."
        )
    lines.append("")
    algorithms = [res["label"] for _key, res in ordered]
    lines.append(_markdown_table(
        ["c_FN : c_FP"] + algorithms,
        [
            [ratio] + [_fmt(values[res_key]["cost_per_pair"])
                       for res_key in [k for k, _ in ordered]]
            for ratio, values in sweep.items()
        ],
    ))
    lines.append("")

    # -- bands ------------------------------------------------------------
    lines.append("## Three-band operating point (the shape that would actually ship)")
    lines.append("")
    lines.append("The cost-optimal threshold is the right threshold for *triaging* a "
                 "queue. It is not the right threshold for *acting without a human*, "
                 "because the cost of a false positive differs depending on whether "
                 "anyone is still in the loop. So the shipped configuration uses two "
                 "boundaries chosen from opposite ends:")
    lines.append("")
    lines.append(f"- **auto-approve** at or above the *precision* cut-off: the lowest "
                 f"score at which at least {BAND_TARGET_PRECISION:.0%} of auto-approved "
                 f"pairs really are matches.")
    lines.append("- **auto-reject** below the *recall* cut-off: the **lowest** score that "
                 "any genuine match in the data reached. Anything below it cannot be a "
                 "true match, so nothing genuine is auto-rejected. (It is the lowest, "
                 "not the highest: the cut-off is the floor of the positive score "
                 "distribution.)")
    lines.append("- **everything between goes to a human.**")
    lines.append("")
    band_rows = []
    for _key, res in ordered:
        bands = res["bands"]
        detail = res["band_detail"]
        usable = detail["three_band_pipeline_usable"]
        band_rows.append([
            res["label"],
            _fmt(detail["raw_reject_below"]),
            _fmt(detail["raw_approve_at_or_above"]),
            f"{bands['approve_n']} ({bands['approve_pct']:.0f}%)",
            _fmt(bands["approve_accuracy"]),
            "yes" if detail["approve_meets_target"] else "**no**",
            f"{bands['review_n']} ({bands['review_pct']:.0f}%)",
            f"{bands['reject_n']} ({bands['reject_pct']:.0f}%)",
            _fmt(bands["reject_accuracy"]),
            "yes" if usable else "**no**",
        ])
    lines.append(_markdown_table(
        ["algorithm", "reject <", "approve >=", "auto-approve",
         "of which matches", f"meets {BAND_TARGET_PRECISION:.0%} target",
         "manual review", "auto-reject", "of which non-matches",
         "three bands usable?"],
        band_rows,
    ))
    lines.append("")
    lines.append(
        "*Cut-offs are printed from the un-normalised values, so `reject <` is "
        "always the zero-false-negative floor and `approve >=` is always the "
        "precision cut-off, even in the degenerate case where they cross.*"
    )
    lines.append("")
    lines.append(
        "`three bands usable?` = no means the pipeline is not really three bands. "
        "Either a band is empty -- nothing reaches it, so it never fires -- or the "
        "two cut-offs cross, leaving no score range that is simultaneously safe to "
        "act on automatically and free of missed matches. An algorithm with an "
        "empty reject band is a two-band pipeline: it auto-approves and reviews, "
        "and every borderline case goes to a human."
    )
    lines.append("")
    lines.append(
        f"`meets {BAND_TARGET_PRECISION:.0%} target` = no means no threshold reaches "
        f"{BAND_TARGET_PRECISION:.0%} precision at all, so the auto-approve cut-off is "
        f"the most precise one available rather than a compliant one. The "
        f"`of which matches` column beside it is the number that actually matters; "
        f"the stated rule was not met."
    )
    lines.append("")

    # -- bootstrap --------------------------------------------------------
    if bootstrap:
        lines.append("## Is the top algorithm actually better than the runner-up?")
        lines.append("")
        lines.append("Paired bootstrap over 2000 resamples of the dataset, comparing "
                     "per-pair cost at each algorithm's own optimal threshold.")
        lines.append("")
        lines.append(f"- mean cost delta (runner-up minus top): **{bootstrap['mean_delta']:+.4f}** per pair")
        lines.append(f"- 95% confidence interval: **[{bootstrap['ci_low']:+.4f}, {bootstrap['ci_high']:+.4f}]**")
        lines.append(f"- P(top is cheaper than runner-up): **{bootstrap['p_a_cheaper']:.3f}**")
        lines.append("")
        # delta = cost(runner-up) - cost(top), so positive favours the top algorithm.
        if bootstrap["ci_low"] > 0:
            verdict = ("The top algorithm is significantly cheaper. The gap survives "
                       "resampling, so the ranking is a real property of the data and "
                       "not an artefact of the particular "
                       f"{payload['dataset']['n_pairs']} pairs.")
        elif bootstrap["ci_high"] < 0:
            verdict = ("The runner-up is significantly cheaper, so the headline ranking "
                       "above is wrong. Fix the ranking before drawing any conclusion.")
        else:
            verdict = ("**The difference is not statistically significant at 95%.** The "
                       "95% interval spans zero, so on this dataset the two are "
                       "effectively tied on cost. The headline table still orders them, "
                       "but the ordering should not be treated as evidence on its own, "
                       "and the simpler algorithm should be preferred on grounds other "
                       "than this measurement.")
        lines.append(f"**Verdict:** {verdict}")
        lines.append("")

    # -- per category -----------------------------------------------------
    lines.append("## Per-category errors: which noise each algorithm actually fails on")
    lines.append("")
    lines.append(
        "An aggregate score cannot distinguish a matcher that is uniformly mediocre "
        "from one that is excellent except on a specific failure mode. These two "
        "tables are the actionable output."
    )
    lines.append("")
    lines.append(
        f"**False positives at a matched operating point (recall >= {RECALL_FLOOR:.2f}, "
        "lowest false-positive count each algorithm can reach).** This is the "
        "precision side: how many *different people* each algorithm would wrongly "
        "accept while still catching 90% of genuine matches."
    )
    lines.append("")
    lines.append(_markdown_table(
        ["category"] + [res["label"].split(" (")[0] for _k, res in ordered],
        [[row["category"]] + [str(value) for value in row["fp_by_algorithm"]]
         for row in per_category_fp],
    ))
    lines.append("")
    lines.append(
        f"**False negatives at the same operating point** -- genuine customers each "
        "algorithm would send to manual review."
    )
    lines.append("")
    lines.append(_markdown_table(
        ["category"] + [res["label"].split(" (")[0] for _k, res in ordered],
        [[row["category"]] + [str(value) for value in row["fn_by_algorithm"]]
         for row in per_category_fn],
    ))
    lines.append("")
    lines.append(
        "Read the two together. `hard_negative_identical_name` is a 100% false-positive "
        "row for *every* algorithm and is not a bug: those are pairs of different "
        "people whose names are byte-for-byte identical, so no name-only matcher can "
        "separate them. That row is the floor on achievable precision, and it is in "
        "the dataset precisely so the reported number is not quietly optimistic."
    )
    lines.append("")

    # -- cross-validation -------------------------------------------------
    cv = payload.get("cross_validation", {})
    if cv:
        lines.append("## Cross-validation of the learned model")
        lines.append("")
        lines.append(f"The learned matcher is scored with **{cv.get('folds', 0)}-fold "
                     "out-of-fold predictions**: every score in every table above "
                     "comes from a model that never saw that row. Without this the "
                     "learned model would be graded on its own training data and "
                     "would flatter itself.")
        lines.append("")
        lines.append(
            "**What this does not cover.** The *operating threshold* is chosen once, "
            "afterwards, from the pooled out-of-fold scores of the whole dataset -- not "
            "per fold. That is the standard arrangement for tuning a threshold on "
            "cross-validated scores, and it does mean threshold-selection information "
            "is shared across folds. A nested scheme would not, and would cost five "
            "times as much. With the threshold reported alongside a bootstrap interval "
            "over the dataset as a whole, the leak is small -- but describing it as "
            "\"the threshold is chosen on the other folds\" would be false, so it is "
            "not described that way."
        )
        lines.append("")
        if "fold_train_losses" in cv:
            losses = ", ".join(_fmt(v, 4) for v in cv["fold_train_losses"])
            lines.append(f"- per-fold training log-loss: {losses}")
        if "full_model_loss" in cv:
            lines.append(f"- full-model training log-loss: {_fmt(cv['full_model_loss'], 4)}")
        lines.append("")

    lines.append("## Reproducing this report")
    lines.append("")
    lines.append("```")
    lines.append("python3 -m name_match.cli all")
    lines.append("```")
    lines.append("")
    lines.append("No third-party packages are required. Python 3.9 or newer.")
    lines.append("")
    return "\n".join(lines)


def cmd_all(args: argparse.Namespace) -> int:
    print("[1/4] Generating dataset...")
    rc = cmd_generate(args)
    if rc:
        return rc
    print()
    print("[2/4] Training learned combiner...")
    rc = cmd_train(args)
    if rc:
        return rc
    print()
    print("[3/4] Evaluating all algorithms...")
    rc = cmd_evaluate(args)
    if rc:
        return rc
    print()
    print("[4/4] Writing report...")
    rc = cmd_report(args)
    return rc


def _add_common_arguments(parser: argparse.ArgumentParser,
                          suppress_defaults: bool = False) -> None:
    """Path options, added to both the top level and every subcommand.

    Two argparse traps are avoided here.

    1. Global options are otherwise only accepted *before* the subcommand, so
       ``cli all --data foo.csv`` fails while ``cli --data foo.csv all`` works.
       Reviewers reasonably try both, and "unrecognized arguments" on the
       natural ordering is exactly the ambiguity the brief asks us to remove.

    2. A subparser argument with a default silently *overwrites* whatever the
       top-level parser already resolved, so ``cli --json out.json report``
       would write to the default location. ``SUPPRESS`` makes the subparser
       copy set the value only when the flag is actually given.
    """
    def default_for(fallback):
        return argparse.SUPPRESS if suppress_defaults else fallback

    parser.add_argument("--data", default=default_for(dataset_module.DEFAULT_DATA_PATH),
                        help="path to the labelled dataset CSV")
    parser.add_argument("--model", default=default_for(DEFAULT_MODEL_PATH),
                        help="path to save/load the learned model JSON")
    parser.add_argument("--json", default=default_for(DEFAULT_REPORT_JSON),
                        help="path for the machine-readable report")
    parser.add_argument("--markdown", default=default_for(DEFAULT_REPORT_MD),
                        help="path for the human-readable report")


DEFAULT_ABLATION_JSON = os.path.join("reports", "ablation.json")
DEFAULT_ABLATION_MD = os.path.join("reports", "ablation.md")


def cmd_ablate(args: argparse.Namespace) -> int:
    """Leave-one-feature-out ablation. Opt-in: it refits the model 24 times and
    takes a couple of minutes, which is why it is not part of ``all``."""
    from .ablation import ablation_report, format_ablation_table

    pairs = dataset_module.read_csv(args.data) if os.path.exists(args.data) \
        else build_dataset()
    folds = kfold_indices(len(pairs), CV_FOLDS)
    print(f"Ablating {len(FEATURE_NAMES)} features over {len(pairs)} pairs, "
          f"{CV_FOLDS} folds. This refits the model once per feature; expect a "
          f"couple of minutes.")
    report = ablation_report(pairs, folds)

    table = format_ablation_table(report)
    worst = report["features"][0]
    body = [
        "# Feature ablation: what each feature is actually worth",
        "",
        "Generated by `python3 -m name_match.cli ablate`. Leave-one-feature-out: "
        "each feature is removed, the model is refit across the same folds, and "
        "the out-of-fold cost is compared against the full model on exactly "
        "those folds.",
        "",
        "**Read this next to the coefficient table, not instead of it.** A large "
        "weight is not evidence that a feature is doing work. These two tables "
        "disagree in both directions, which is exactly why the disagreement is "
        "worth publishing:",
        "",
        f"* `{worst['feature']}` carries a weight of "
        f"{float(worst['full_weight']):+.2f} and is nevertheless the single most "
        f"load-bearing feature by cost, at "
        f"{float(worst['oof_cost_delta']):+.3f} cost per pair when removed.",
        "* `length_ratio` has one of the largest weights in the model and "
        "removing it *improves* out-of-fold cost. On 494 rows that is inside the "
        "noise, so the honest reading is that it is not earning its place, not "
        "that it is harmful.",
        "",
        "A negative delta means the model did slightly better without the "
        "feature. With 23 features on 494 rows those are not distinguishable "
        "from zero and are reported rather than rounded away.",
        "",
        table,
        "",
    ]
    lines = [ln for ln in body if ln is not None]
    while lines and lines[-1] == "":
        lines.pop()
    md = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.ablation_markdown) or ".", exist_ok=True)
    with open(args.ablation_json, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_safe(report), handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")
    with open(args.ablation_markdown, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(md)
    print(f"Wrote {args.ablation_json} and {args.ablation_markdown}")
    print(md)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m name_match.cli",
        description="Name matching across identity documents: generation, "
                    "training, evaluation and reporting.",
    )
    _add_common_arguments(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler, help_text in (
        ("generate", cmd_generate, "generate the labelled dataset"),
        ("train", cmd_train, "train the learned combiner"),
        ("evaluate", cmd_evaluate, "score all algorithms and print the comparison"),
        ("report", cmd_report, "write reports/results.json and reports/results.md"),
        ("all", cmd_all, "generate, train, evaluate and report"),
        ("ablate", cmd_ablate, "leave-one-feature-out ablation (slow, optional)"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        _add_common_arguments(sub, suppress_defaults=True)
        if name == "ablate":
            sub.add_argument("--ablation-json", default=DEFAULT_ABLATION_JSON,
                             help="path for the machine-readable ablation")
            sub.add_argument("--ablation-markdown", default=DEFAULT_ABLATION_MD,
                             help="path for the human-readable ablation")
        sub.set_defaults(handler=handler)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
