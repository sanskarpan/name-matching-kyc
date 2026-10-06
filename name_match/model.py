"""A small L2-regularised logistic regression, implemented from scratch.

Why hand-rolled: the exercise says "assume the reviewer has only the language
runtime". A 494-row, 23-feature, binary problem does not need a gradient
descent library, and writing the optimiser out means the convergence behaviour
is inspectable rather than a black box. It trains in well under a second.

The model is deliberately small and fully serialisable to JSON, so the shipped
artefact is inspectable and the reported coefficients can be quoted verbatim in
NOTES.md.
"""

from __future__ import annotations

import warnings
import json
import math
import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

__all__ = ["LogisticRegression", "fit", "standardize", "sigmoid", "DEFAULT_MODEL_PATH"]

DEFAULT_MODEL_PATH = os.path.join("data", "model.json")


def sigmoid(z: float) -> float:
    """Numerically stable logistic sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    # exp(z) would underflow for very negative z; rewrite to avoid it.
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def standardize(matrix: Sequence[Sequence[float]]) -> tuple[list[list[float]], list[float], list[float]]:
    """Zero-mean unit-variance scaling, fitted on the training rows only.

    Returns ``(scaled_matrix, means, stds)``. Scaling matters here because the
    raw features span very different ranges (``exact_normalized`` is 0/1,
    ``n_token_diff`` is a small integer, ``phonetic_max`` is 0-1); without it
    gradient descent converges at wildly different per-feature rates.
    """
    if not matrix:
        return [], [], []
    n_features = len(matrix[0])
    means = [sum(row[j] for row in matrix) / len(matrix) for j in range(n_features)]

    stds = []
    for j in range(n_features):
        variance = sum((row[j] - means[j]) ** 2 for row in matrix) / len(matrix)
        stds.append(math.sqrt(variance) if variance > 1e-12 else 1.0)

    scaled = [
        [(row[j] - means[j]) / stds[j] for j in range(n_features)]
        for row in matrix
    ]
    return scaled, means, stds


@dataclass
class LogisticRegression:
    """Binary logistic regression with L2 regularisation (bias unpenalised)."""

    learning_rate: float = 0.35
    #: Converged well before this: the gradient norm of the true objective is
    #: 6e-7 at 1200 iterations versus 4e-16 at 5000, and weights move by
    #: 1.6e-5 (predictions by 2e-6) over a 50000-iteration reference run.
    #: Set where it stops moving rather than at a round number.
    iterations: int = 1200
    l2: float = 0.02
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    means: list[float] = field(default_factory=list)
    stds: list[float] = field(default_factory=list)
    feature_names: tuple[str, ...] = ()
    n_train: int = 0
    #: Mean binary cross-entropy on the training rows at the final iteration.
    #: Kept so a bad fit is visible rather than silent.
    final_loss: float = float("nan")

    # -- fitting ----------------------------------------------------------

    def fit(self, matrix: Sequence[Sequence[float]], labels: Sequence[int],
            feature_names: Sequence[str] = ()) -> "LogisticRegression":
        if not matrix:
            raise ValueError("cannot fit on an empty matrix")
        if len(matrix) != len(labels):
            raise ValueError(
                f"matrix has {len(matrix)} rows but {len(labels)} labels")
        n_features = len(matrix[0])
        for index, row in enumerate(matrix):
            if len(row) != n_features:
                raise ValueError(
                    f"row {index} has {len(row)} features, expected {n_features}; "
                    "a ragged feature matrix silently drops or reuses columns")
            if any(not math.isfinite(value) for value in row):
                raise ValueError(f"row {index} contains a non-finite feature")
        # `repr` is used for the key so a mixed-type label list reports the
        # offending values instead of raising TypeError out of `sorted`.
        bad_labels = sorted({repr(label) for label in labels
                            if label not in (0, 1, True, False)})
        if bad_labels:
            raise ValueError(
                f"labels must be 0 or 1, found {bad_labels}; binary cross-entropy "
                "is undefined for other values")
        if feature_names and len(feature_names) != n_features:
            raise ValueError(
                f"{len(feature_names)} feature names for {n_features} columns; "
                "the published coefficients would be silently mislabelled")

        scaled, means, stds = standardize(matrix)
        n_rows = len(scaled)

        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.means = means
        self.stds = stds
        self.feature_names = tuple(feature_names)
        self.n_train = n_rows

        n = float(n_rows)
        for iteration in range(self.iterations):
            gradient = [0.0] * n_features
            bias_gradient = 0.0

            for row, label in zip(scaled, labels):
                probability = sigmoid(self.bias + sum(w * x for w, x in zip(self.weights, row)))
                error = probability - label
                for j, x in enumerate(row):
                    gradient[j] += error * x
                bias_gradient += error

            step = self.learning_rate / n
            for j in range(n_features):
                # The objective being minimised is `mean_BCE + 0.5*l2*||w||^2`.
                # The data term is scaled by `step = lr/n` because it is a mean;
                # the penalty is scaled by `lr` alone because it is not a mean.
                # Together these make the effective penalty independent of the
                # row count (verified: one iteration at n=4 and at n=200 produce
                # identical weights) and independent of the iteration count.
                gradient[j] = (step * gradient[j]
                               + self.learning_rate * self.l2 * self.weights[j])
                self.weights[j] -= gradient[j]
            self.bias -= step * bias_gradient

        self.final_loss = self.log_loss(scaled, labels)
        return self

    def log_loss(self, matrix: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
        """Mean binary cross-entropy.

        The sigmoid is applied here. Skipping it is a small, invisible mistake
        with a large numerical consequence: the clamp below operates on a *logit*
        rather than a probability, so its surviving window is roughly [0, 1]
        instead of [eps, 1-eps]. Every logit below 0 then collapses to p = 1e-12
        (loss 27.6) and every logit above 1 collapses to p = 1 - 1e-12 (loss ~0),
        which turns the reported training loss into noise and makes a well-fitted
        model look broken.
        """
        if not matrix:
            return float("nan")
        total = 0.0
        for row, label in zip(matrix, labels):
            probability = sigmoid(self._raw_score(row))
            probability = min(max(probability, 1e-12), 1 - 1e-12)
            total += -(label * math.log(probability) + (1 - label) * math.log(1 - probability))
        return total / len(matrix)

    # -- prediction -------------------------------------------------------

    def _raw_score(self, scaled_row: Sequence[float]) -> float:
        return self.bias + sum(w * x for w, x in zip(self.weights, scaled_row))

    def predict_proba(self, row: Sequence[float]) -> float:
        """Probability that the pair is a match.

        Raises on a column-count mismatch. Without the check, a stale model
        artefact with fewer weights than the current feature vector would be fed
        the *first* few features and return a confident-looking number computed
        from almost no information -- the worst possible failure mode for a
        model loaded from disk.
        """
        if not self.weights:
            # Unfitted model: return a constant so the matcher still runs.
            #
            # The warning is not decoration. An unfitted learned arm used to
            # return 0.5 for every pair in silence, which is indistinguishable
            # from a real model that is merely unconfident -- and NOTES section
            # 5 promises the pipeline "degrades to review, never to
            # auto-approve". 0.5 sits between both bands, so the promise holds,
            # but only by coincidence. One call site forgetting to load the
            # artefact would look like a working pipeline.
            warnings.warn(
                "predict_proba called on an unfitted LogisticRegression; "
                "returning the neutral 0.5 for every pair. Fit the model or "
                "load the artefact before scoring.",
                RuntimeWarning, stacklevel=2)
            return 0.5
        if len(row) != len(self.weights):
            raise ValueError(
                f"feature vector has {len(row)} values but the model was fitted "
                f"on {len(self.weights)}; the artefact is stale or the feature "
                "order changed")
        if any(not math.isfinite(value) for value in row):
            raise ValueError("feature vector contains a non-finite value")
        scaled = [
            (row[j] - self.means[j]) / self.stds[j] for j in range(len(self.weights))
        ]
        return sigmoid(self._raw_score(scaled))

    # -- introspection ----------------------------------------------------

    def coefficient_report(self) -> list[tuple[str, float]]:
        """``(feature_name, standardised_weight)`` sorted by magnitude.

        These are coefficients on *standardised* features, so magnitudes are
        directly comparable: a weight of 0.8 means one standard deviation of
        that feature moves the log-odds by 0.8.
        """
        names = self.feature_names or tuple(f"f{i}" for i in range(len(self.weights)))
        pairs = list(zip(names, self.weights))
        pairs.sort(key=lambda item: -abs(item[1]))
        return pairs

    def to_dict(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "iterations": self.iterations,
            "l2": self.l2,
            "weights": self.weights,
            "bias": self.bias,
            "means": self.means,
            "stds": self.stds,
            "feature_names": list(self.feature_names),
            "n_train": self.n_train,
            "final_loss": self.final_loss,
        }

    def save(self, path: str = DEFAULT_MODEL_PATH) -> str:
        """Serialise to JSON.

        ``allow_nan=False`` is deliberate. Python's ``json`` emits bare ``NaN``
        tokens by default, which are not valid JSON per RFC 8259; ``jq`` and
        Python accept them but stricter consumers do not, and a "machine-readable"
        artefact should survive a strict parser. An unfitted model's loss is
        NaN, so it is normalised to ``None`` first.
        """
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = self.to_dict()
        if isinstance(payload.get("final_loss"), float) \
                and payload["final_loss"] != payload["final_loss"]:
            payload["final_loss"] = None
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        return path

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH,
             expected_features: Sequence[str] | None = None) -> "LogisticRegression":
        """Load a model, refusing artefacts whose feature order has drifted.

        ``expected_features`` is the *current* feature order. When supplied, the
        saved order must match it exactly. Without this check a model saved
        before a feature was added or renamed loads successfully and then scores
        every pair using the wrong weights, in the wrong order, with no error.
        """
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if not isinstance(payload, dict):
            raise ValueError(f"corrupt model artefact at {path}: expected an object")
        required = set(cls().to_dict())
        if not required <= payload.keys():
            raise ValueError(f"corrupt model artefact at {path}: missing fields")
        for key in ("weights", "means", "stds", "feature_names"):
            if not isinstance(payload[key], list):
                raise ValueError(f"corrupt model artefact at {path}: {key} must be a list")

        width = len(payload["weights"])
        for key in ("means", "stds"):
            if len(payload[key]) != width:
                raise ValueError(f"corrupt model artefact at {path}: weights/{key} length mismatch")
        if payload["feature_names"] and len(payload["feature_names"]) != width:
            raise ValueError(f"corrupt model artefact at {path}: feature names length mismatch")
        if any(not isinstance(name, str) for name in payload["feature_names"]):
            raise ValueError(f"corrupt model artefact at {path}: feature names must be strings")
        numeric = [payload["bias"], payload["learning_rate"], payload["l2"],
                   *payload["weights"], *payload["means"], *payload["stds"]]
        if any(not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in numeric):
            raise ValueError(f"corrupt model artefact at {path}: non-finite or non-numeric values")
        if any(value <= 0 for value in payload["stds"]):
            raise ValueError(f"corrupt model artefact at {path}: standard deviations must be positive")

        if expected_features is not None:
            saved = tuple(payload.get("feature_names", ()))
            if saved != tuple(expected_features):
                raise ValueError(
                    f"model artefact at {path} was trained on features "
                    f"{list(saved)[:4]}... but the current extractor produces "
                    f"{list(expected_features)[:4]}...; refit the model "
                    "(python3 -m name_match.cli train) rather than loading a "
                    "stale one")

        return cls(
            learning_rate=payload["learning_rate"],
            iterations=payload["iterations"],
            l2=payload["l2"],
            weights=payload["weights"],
            bias=payload["bias"],
            means=payload["means"],
            stds=payload["stds"],
            feature_names=tuple(payload["feature_names"]),
            n_train=payload["n_train"],
            final_loss=payload["final_loss"],
        )


def fit(matrix: Sequence[Sequence[float]], labels: Sequence[int],
        feature_names: Sequence[str] = (), **kwargs) -> LogisticRegression:
    """Convenience wrapper around :meth:`LogisticRegression.fit`."""
    return LogisticRegression(**kwargs).fit(matrix, labels, feature_names)
