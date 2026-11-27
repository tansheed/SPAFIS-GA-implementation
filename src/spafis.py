"""
SPAFIS - Soft Prototype-based Autonomous Fuzzy Inference System
(scikit-learn compatible estimator: fit / predict / get_params / set_params / clone).

Inference:  x -> log-activation of every prototype (fuzzy rule, skfuzzy Gaussian)
              -> weighted firing strength
              -> class aggregation (scipy logsumexp over the rules of each class, or max)
              -> argmax class

Autonomous learning (single online pass, no pre-specified rule base):
    for each training sample (x, y):
        look only at prototypes of class y;  a* = highest activation among them
        if a* >= prototype_threshold (or the per-class cap is hit):  UPDATE winner j*
              c   <- c + lr * (x - c)
              s^2 <- (1-lr) * s^2 + lr * (x - c_old)^2          (running spread)
        else: CREATE a new prototype (centre = x, spread = initial_spread)
    finally: rule weights  w_p = (n_p / max n in its class) ^ rule_weight
"""
import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_X_y

from .fuzzy import log_activation, MIN_SPREAD
from .prototype import Prototype


class _ClassBank:
    """Growable arrays holding all prototypes of ONE class (fast online updates)."""

    def __init__(self, d, capacity):
        self.centers = np.zeros((capacity, d))
        self.spreads = np.zeros((capacity, d))
        self.counts = np.zeros(capacity, dtype=int)
        self.n = 0

    def add(self, x, spread0):
        self.centers[self.n], self.spreads[self.n], self.counts[self.n] = x, spread0, 1
        self.n += 1


class SPAFIS(BaseEstimator, ClassifierMixin):
    """
    Tunable parameters (= GA chromosome genes)
    ------------------------------------------
    prototype_threshold : min activation for a sample to be absorbed by an existing prototype
    initial_spread      : Gaussian spread of a newly created prototype
    spread_scaling      : global multiplier applied to all spreads in the membership function
    learning_rate       : step of centre/spread update
    rule_weight         : exponent gamma in  w_p = (n_p / n_max)^gamma
    attack_bias         : constant added to the log class score of every class except
                          `normal_class`.  0 = plain SPAFIS.  Larger values make the
                          classifier flag more traffic as attack (fewer false negatives).

    Structural settings (not optimised)
    -----------------------------------
    aggregation, class_aggregation, max_prototypes_per_class, spread_floor_ratio, n_epochs
    normal_class : label of the "normal" class (None disables attack_bias)
    random_state : seed for the order in which samples are presented
    """

    def __init__(self, prototype_threshold=0.5, initial_spread=0.5, spread_scaling=1.0,
                 learning_rate=0.1, rule_weight=1.0, aggregation="geometric_mean",
                 class_aggregation="sum", max_prototypes_per_class=150,
                 spread_floor_ratio=0.25, n_epochs=1, attack_bias=0.0, normal_class=None,
                 random_state=42):
        self.prototype_threshold = prototype_threshold
        self.initial_spread = initial_spread
        self.spread_scaling = spread_scaling
        self.learning_rate = learning_rate
        self.rule_weight = rule_weight
        self.aggregation = aggregation
        self.class_aggregation = class_aggregation
        self.max_prototypes_per_class = max_prototypes_per_class
        self.spread_floor_ratio = spread_floor_ratio
        self.n_epochs = n_epochs
        self.attack_bias = attack_bias
        self.normal_class = normal_class
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit(self, X, y):
        """Build the rule base from (X, y). The training arrays are never modified."""
        X, y = check_X_y(X, y, dtype=float)
        n, d = X.shape
        self.classes_ = np.unique(y)
        cap, lr = self.max_prototypes_per_class, self.learning_rate
        banks = {c: _ClassBank(d, cap) for c in self.classes_}
        floor = max(MIN_SPREAD, self.spread_floor_ratio * self.initial_spread)
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_epochs):
            for i in rng.permutation(n):
                x, bank = X[i], banks[y[i]]
                if bank.n == 0:
                    bank.add(x, self.initial_spread)
                    continue
                la = log_activation(x[None, :], bank.centers[:bank.n],
                                    bank.spreads[:bank.n] * self.spread_scaling,
                                    self.aggregation)[0]
                j = int(np.argmax(la))
                if np.exp(la[j]) >= self.prototype_threshold or bank.n >= cap:
                    diff = x - bank.centers[j]
                    bank.centers[j] += lr * diff                              # move centre
                    var = (1 - lr) * bank.spreads[j] ** 2 + lr * diff ** 2
                    bank.spreads[j] = np.maximum(np.sqrt(var), floor)         # adapt spread
                    bank.counts[j] += 1
                else:
                    bank.add(x, self.initial_spread)                          # new rule

        self.prototypes_ = []
        for c in self.classes_:
            b = banks[c]
            nmax = max(1, b.counts[:b.n].max()) if b.n else 1
            for k in range(b.n):
                w = (b.counts[k] / nmax) ** self.rule_weight
                self.prototypes_.append(Prototype(b.centers[k].copy(), b.spreads[k].copy(),
                                                  float(w), int(c), int(b.counts[k])))
        # stacked matrices for vectorised inference
        self._C = np.array([q.center for q in self.prototypes_])
        self._S = np.array([q.spread for q in self.prototypes_]) * self.spread_scaling
        self._logW = np.log(np.array([q.weight for q in self.prototypes_]) + 1e-300)
        self._cls = np.array([q.class_label for q in self.prototypes_])
        return self

    # ------------------------------------------------------------ inference
    def decision_function(self, X):
        """(n, n_classes) log class activations (higher = more likely)."""
        X = check_array(X, dtype=float)
        la = log_activation(X, self._C, self._S, self.aggregation) + self._logW[None, :]
        out = np.full((X.shape[0], len(self.classes_)), -np.inf)
        for k, c in enumerate(self.classes_):
            cols = la[:, self._cls == c]
            if cols.shape[1]:
                out[:, k] = logsumexp(cols, axis=1) if self.class_aggregation == "sum" else cols.max(axis=1)
        if self.normal_class is not None and self.attack_bias:
            out[:, self.classes_ != self.normal_class] += self.attack_bias   # lean towards "attack"
        return out

    def predict(self, X):
        """Predicted class = class with highest aggregated rule activation."""
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    @property
    def n_rules(self):
        return len(self.prototypes_)
