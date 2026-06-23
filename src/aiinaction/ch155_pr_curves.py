"""Precision-Recall curves and Average Precision from scratch.

A small, well-validated reference implementation of the Precision-Recall (PR)
curve and its scalar summaries for binary classification. The public API mirrors
the Julia (`AIInAction.Ch155PrCurves`) and Rust (`aiinaction::ch155_pr_curves`)
implementations one-to-one; the cross-language parity tests assert that all three
agree to within floating-point tolerance on shared fixtures.

Definitions
-----------
Given binary labels ``y_i in {0, 1}`` and real-valued scores ``s_i`` (larger means
more positive), a threshold ``tau`` predicts ``yhat_i = 1[s_i >= tau]``. Sweeping
``tau`` over the distinct score values (from high to low) traces the PR curve as a
sequence of points ``(recall, precision)`` where

    precision = TP / (TP + FP),   recall = TP / P,

with ``P`` the number of actual positives. ``precision`` is defined as ``1.0`` at a
threshold that predicts nothing positive (the conventional limiting value).

Average precision (AP) is the rank-based estimator

    AP = (1 / P) * sum_{i : y_i = 1} Precision@k_i,

where ``k_i`` is the rank of the ``i``-th positive in the score-sorted list. This is
exactly the estimator used by scikit-learn's ``average_precision_score`` and is the
recommended summary because it does not interpolate across the curve's sawtooth.

``auprc_trapezoid`` integrates precision against recall by the trapezoidal rule over
the curve points produced here. Because of the sawtooth, the trapezoidal estimate
generally differs from AP and is mildly optimistic; AP is preferred in practice.
"""
from __future__ import annotations

from collections.abc import Sequence

__all__ = ["PRCurve", "pr_curve", "average_precision", "auprc_trapezoid"]


class PRCurve:
    """A precision-recall curve, one point per distinct threshold.

    Attributes
    ----------
    precision:
        Precision at each operating point, in order of decreasing threshold.
    recall:
        Recall at each operating point, in order of decreasing threshold. This
        sequence is non-decreasing.
    thresholds:
        The distinct score thresholds, in decreasing order. ``thresholds[k]`` is
        the value ``tau`` at which point ``k`` is realized (predict positive iff
        ``score >= tau``).
    """

    __slots__ = ("precision", "recall", "thresholds")

    def __init__(
        self,
        precision: list[float],
        recall: list[float],
        thresholds: list[float],
    ) -> None:
        self.precision = precision
        self.recall = recall
        self.thresholds = thresholds

    def __len__(self) -> int:
        return len(self.thresholds)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PRCurve(n_points={len(self)}, "
            f"recall=[{self.recall[0]:.4g}..{self.recall[-1]:.4g}])"
        )


def _validate(y_true: Sequence[int], scores: Sequence[float]) -> tuple[list[int], list[float]]:
    yt = [int(v) for v in y_true]
    sc = [float(v) for v in scores]
    if len(yt) != len(sc):
        raise ValueError(
            f"length mismatch: len(y_true)={len(yt)} != len(scores)={len(sc)}"
        )
    if not yt:
        raise ValueError("inputs must be non-empty")
    for v in yt:
        if v not in (0, 1):
            raise ValueError(f"y_true must contain only 0/1 labels, found {v}")
    for v in sc:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("scores contains non-finite values (nan or inf)")
    if sum(yt) == 0:
        raise ValueError("y_true must contain at least one positive (label 1)")
    return yt, sc


def pr_curve(y_true: Sequence[int], scores: Sequence[float]) -> PRCurve:
    """Compute the precision-recall curve over distinct score thresholds.

    Parameters
    ----------
    y_true:
        Binary labels, each ``0`` or ``1``. Must contain at least one positive.
    scores:
        Real-valued classifier scores, one per label. Larger means more positive.

    Returns
    -------
    PRCurve
        One ``(recall, precision)`` point per distinct threshold, ordered by
        decreasing threshold. Recall is non-decreasing along the sequence.

    Examples
    --------
    >>> c = pr_curve([1, 0, 1], [0.9, 0.4, 0.8])
    >>> [round(p, 4) for p in c.precision]
    [1.0, 1.0, 0.6667]
    >>> c.recall
    [0.5, 1.0, 1.0]
    """
    yt, sc = _validate(y_true, scores)
    p_total = sum(yt)

    # Distinct thresholds in decreasing order.
    thresholds = sorted(set(sc), reverse=True)

    precision: list[float] = []
    recall: list[float] = []
    for tau in thresholds:
        tp = 0
        fp = 0
        for label, score in zip(yt, sc):
            if score >= tau:
                if label == 1:
                    tp += 1
                else:
                    fp += 1
        predicted_pos = tp + fp
        prec = 1.0 if predicted_pos == 0 else tp / predicted_pos
        precision.append(prec)
        recall.append(tp / p_total)

    return PRCurve(precision, recall, list(thresholds))


def average_precision(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Rank-based average precision (the scikit-learn ``AP`` estimator).

    Sorts instances by descending score (ties broken by original order) and
    averages the precision observed at each rank where a true positive appears:

        AP = (1 / P) * sum over positives of Precision@k.

    Returns
    -------
    float
        Average precision in ``[0, 1]``.

    Examples
    --------
    >>> round(average_precision([1, 0, 1, 1, 0, 1, 0, 0],
    ...                          [0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.51]), 10)
    0.7708333333
    """
    yt, sc = _validate(y_true, scores)
    p_total = sum(yt)

    # Sort indices by descending score; ties keep the original (ascending index)
    # order so the result is deterministic and matches the other languages.
    order = sorted(range(len(yt)), key=lambda i: (-sc[i], i))

    ap = 0.0
    tp = 0
    seen = 0
    for i in order:
        seen += 1
        if yt[i] == 1:
            tp += 1
            ap += tp / seen
    return ap / p_total


def auprc_trapezoid(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the PR curve by the trapezoidal rule over :func:`pr_curve` points.

    Integrates precision against recall using the trapezoidal rule on the curve
    points produced by :func:`pr_curve`. Because the PR curve is a sawtooth, this
    estimate generally differs from (and is mildly optimistic relative to)
    :func:`average_precision`, which is the preferred summary.

    Returns
    -------
    float
        Trapezoidal area under the precision-recall curve.

    Examples
    --------
    >>> round(auprc_trapezoid([1, 0, 1, 1, 0, 1, 0, 0],
    ...                        [0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.51]), 10)
    0.48125
    """
    curve = pr_curve(y_true, scores)
    rec = curve.recall
    prec = curve.precision
    area = 0.0
    for k in range(1, len(rec)):
        width = abs(rec[k] - rec[k - 1])
        area += width * (prec[k] + prec[k - 1]) / 2.0
    return area
