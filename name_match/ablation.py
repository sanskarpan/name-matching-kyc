"""Leave-one-feature-out ablation for the learned combiner.

The coefficient table in `NOTES.md` says which features carry a large weight.
That is not the same question as which features carry the *result*: a feature
can be heavily weighted and nearly useless, because a correlated partner is
present to do the work. This module answers the second question directly -- what
does the report look like if this one feature does not exist? -- by dropping each
column in turn and refitting across the same folds.

Every number is a *difference* against the full model evaluated on exactly the
same folds, so it is a marginal contribution and not an absolute score. A
negative delta means the model did slightly better without the feature, which
for 23 features on 494 rows is well inside the noise and is reported as such
rather than being rounded to zero or quietly dropped.

This is also the reason the ablation is worth its runtime: a coefficient table
alone invites the reader to treat magnitude as importance, and on a
correlated feature set that inference is wrong often enough to matter.
"""

from __future__ import annotations

from typing import Sequence

from .features import FEATURE_NAMES
from .model import LogisticRegression

__all__ = ["ablate_features", "ablation_report", "format_ablation_table"]


def _cross_validated(matrix: Sequence[Sequence[float]],
                     labels: Sequence[int],
                     folds: Sequence[tuple[Sequence[int], Sequence[int]]],
                     cost_ratio: float) -> dict[str, float]:
    """Out-of-fold scores at one column set, then a single operating point.

    ``feature_names`` is passed to ``fit`` because the model validates its
    weight count against it; the names are not used for anything else, so the
    caller may pass an abridged list.
    """
    from .evaluate import find_operating_point

    out_of_fold = [0.0] * len(labels)
    for train_idx, test_idx in folds:
        model = LogisticRegression()
        model.fit([matrix[i] for i in train_idx], [labels[i] for i in train_idx],
                  list(FEATURE_NAMES)[:len(matrix[0])])
        for i in test_idx:
            out_of_fold[i] = model.predict_proba(matrix[i])
    point = find_operating_point(out_of_fold, labels, 1.0, cost_ratio)
    return {"cost": point.cost_per_pair(1.0, cost_ratio),
            "recall": point.recall,
            "fp": float(point.fp),
            "fn": float(point.fn)}


def ablate_features(matrix: Sequence[Sequence[float]],
                    labels: Sequence[int],
                    folds: Sequence[tuple[Sequence[int], Sequence[int]]],
                    cost_ratio: float = 25.0) -> list[dict[str, object]]:
    """Refit with each feature removed in turn and record the damage.

    Sorts by cost delta, most load-bearing feature first. Ties keep the order
    ``FEATURE_NAMES`` declares, so the output is deterministic.
    """
    width = len(FEATURE_NAMES)
    baseline = _cross_validated(matrix, labels, folds, cost_ratio)
    rows: list[dict[str, object]] = []
    for index, name in enumerate(FEATURE_NAMES):
        keep = [i for i in range(width) if i != index]
        without = _cross_validated(
            [[row[i] for i in keep] for row in matrix], labels, folds, cost_ratio)
        rows.append({
            "feature": name,
            "oof_cost_with": baseline["cost"],
            "oof_cost_without": without["cost"],
            "oof_cost_delta": without["cost"] - baseline["cost"],
            "oof_fp_delta": without["fp"] - baseline["fp"],
            "oof_fn_delta": without["fn"] - baseline["fn"],
            "oof_recall_delta": without["recall"] - baseline["recall"],
        })
    rows.sort(key=lambda row: (-float(row["oof_cost_delta"]),
                               list(FEATURE_NAMES).index(str(row["feature"]))))
    return rows


def ablation_report(pairs, folds, cost_ratio: float = 25.0) -> dict[str, object]:
    """Run the ablation over a dataset and return a JSON-ready payload."""
    from .features import extract_features

    matrix = [extract_features(p.name_a, p.name_b)[0] for p in pairs]
    labels = [p.label for p in pairs]
    rows = ablate_features(matrix, labels, folds, cost_ratio)
    weights = dict(LogisticRegression().fit(
        matrix, labels, FEATURE_NAMES).coefficient_report())
    for row in rows:
        row["full_weight"] = weights[str(row["feature"])]
    return {
        "cost_ratio": cost_ratio,
        "baseline_oof_cost": _cross_validated(matrix, labels, folds, cost_ratio)["cost"],
        "n_features": len(FEATURE_NAMES),
        "features": rows,
    }


def format_ablation_table(report: dict[str, object]) -> str:
    """Markdown table, most load-bearing feature first.

    ``delta`` is out-of-fold cost per pair *increase* when the feature is
    removed. Positive means the feature was carrying weight that nothing else
    replaced.
    """
    lines = ["| feature | weight in full model | OOF cost/pair without it | delta |",
             "|---|---|---|---|"]
    for row in report["features"]:
        lines.append(
            f"| `{row['feature']}` | {float(row['full_weight']):+.2f} | "
            f"{float(row['oof_cost_without']):.3f} | "
            f"{float(row['oof_cost_delta']):+.3f} |")
    return "\n".join(lines)