"""A single SPAFIS prototype == a single fuzzy rule."""
from dataclasses import dataclass
import numpy as np
from .fuzzy import log_activation


@dataclass
class Prototype:
    """
    center       : (d,) point in feature space the prototype is centred on
    spread       : (d,) Gaussian width per feature (the fuzzy 'region size')
    weight       : rule weight in (0,1]; multiplies the firing strength
    class_label  : consequent of the rule (THEN class = class_label)
    count        : number of training samples absorbed by this prototype
    """
    center: np.ndarray
    spread: np.ndarray
    weight: float
    class_label: int
    count: int = 1

    def activation(self, x, aggregation="geometric_mean", spread_scaling=1.0):
        """Firing strength in [0,1] of this rule for sample x (before weighting)."""
        la = log_activation(np.atleast_2d(x), self.center[None, :],
                            self.spread[None, :] * spread_scaling, aggregation)
        return float(np.exp(la[0, 0]))

    def describe(self, feature_names=None, top=5):
        """Human readable rule (the `top` features with the narrowest spread)."""
        d = len(self.center)
        names = feature_names if feature_names is not None else [f"f{j}" for j in range(d)]
        idx = np.argsort(self.spread)[:top]
        cond = " AND ".join(f"{names[j]} ~ {self.center[j]:.3f} (±{self.spread[j]:.3f})" for j in idx)
        return f"IF {cond} ... THEN class={self.class_label}  [w={self.weight:.3f}, n={self.count}]"
