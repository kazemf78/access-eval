"""Breadth / repeatability decomposition of mean accessibility: mu = E[A], effective breadth B = E[A]^2 / E[A^2],
repeatability D = E[A^2] / E[A], so mu = B * D. E[A^2] estimator (M2_ESTIMATOR): "plugin" (paper) = mean over
inquiries of A_hat(q)^2; "pairwise" = mean of c(c-1)/(n(n-1)), unbiased for finite n formulations."""
import numpy as np

M2_ESTIMATOR = "plugin"


def bd_metrics(a, n, estimator=None):
    """(mu, B, D); mu = B * D. estimator: "plugin" = mean(A_hat^2), "pairwise" = mean c(c-1)/(n(n-1))."""
    a, n = np.asarray(a, float), np.asarray(n, float)
    c = np.rint(a * n)
    mu = (c / n).mean()
    if (estimator or M2_ESTIMATOR) == "plugin":
        m2 = ((c / n) ** 2).mean()
    else:
        m2 = (c * (c - 1) / (n * (n - 1))).mean()
    if mu <= 0 or m2 <= 0:
        return mu, np.nan, np.nan
    return mu, mu * mu / m2, m2 / mu
