"""Label tensors and pairwise disagreement for the RQ2 decoding-vs-formulation study."""
import numpy as np
import pandas as pd


def _variability_tensor(path):
    g = pd.read_csv(path)
    y = g["label"].map(lambda v: v is True or v == "True" or v == "correct").astype(np.int8)
    g = g.assign(y=y)
    Q, F, R = g["idx"].nunique(), g["formulation_id"].nunique(), g["repeat_id"].nunique()
    g = g.sort_values(["idx", "formulation_id", "repeat_id"])
    assert len(g) == Q * F * R, f"{path.name}: unbalanced ({len(g)} != {Q}x{F}x{R})"
    return g["y"].to_numpy().reshape(Q, F, R), g["verified"].mean()


def _pairwise_disagreement(Y):
    """Per-inquiry within-formulation and cross-formulation pairwise label disagreement. Y: (Q, F, R)."""
    Q, F, R = Y.shape
    c = Y.sum(-1)  # (Q, F)
    within_dis = (c * (R - c)).sum(-1)
    within_pairs = F * R * (R - 1) / 2
    C, n = c.sum(-1), F * R
    cross_dis = C * (n - C) - within_dis
    cross_pairs = n * (n - 1) / 2 - within_pairs
    return within_dis / within_pairs, cross_dis / cross_pairs
