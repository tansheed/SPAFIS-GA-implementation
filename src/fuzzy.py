"""
Fuzzy-logic layer built on scikit-fuzzy.

Membership : skfuzzy.gaussmf(x, c, s) = exp(-(x - c)^2 / (2 s^2))
Aggregation: product / geometric mean / min of the feature-wise memberships,
             evaluated on LOG memberships so that products of many values
             (d ~ 100 features) do not underflow to 0.
"""
import numpy as np
import skfuzzy as fuzz

MIN_SPREAD = 1e-6          # enforced lower bound on any spread (avoids /0)
_TINY = 1e-300             # avoids log(0)

#   product        : prod_j mu_j            -> sum of logs
#   geometric_mean : (prod_j mu_j)^(1/d)    -> mean of logs  (default, scale-free in d)
#   min            : min_j mu_j             -> min of logs   (Goedel t-norm)
AGGREGATIONS = {
    "product": lambda lm, axis: lm.sum(axis=axis),
    "geometric_mean": lambda lm, axis: lm.mean(axis=axis),
    "min": lambda lm, axis: lm.min(axis=axis),
}


def gaussian_membership(x, center, spread):
    """Feature-wise Gaussian membership in (0,1] (scikit-fuzzy `gaussmf`)."""
    return fuzz.gaussmf(x, center, np.maximum(spread, MIN_SPREAD))


def log_activation(X, centers, spreads, aggregation="geometric_mean", max_elements=8_000_000):
    """
    Log firing strength of K prototypes for n samples.

    X       : (n, d)   samples
    centers : (K, d)   prototype centres
    spreads : (K, d)   prototype spreads (already multiplied by spread_scaling)
    returns : (n, K)   log activation (<= 0); activation = exp(result)
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"Unknown aggregation '{aggregation}'")
    X = np.atleast_2d(X)
    n, d = X.shape
    K = centers.shape[0]
    if K == 0:
        return np.zeros((n, 0))
    agg = AGGREGATIONS[aggregation]
    out = np.empty((n, K))
    step = max(1, int(max_elements // (K * d)))          # chunk rows to bound memory
    for a in range(0, n, step):
        mu = gaussian_membership(X[a:a + step, None, :], centers[None], spreads[None])
        out[a:a + step] = agg(np.log(np.maximum(mu, _TINY)), axis=2)
    return out
