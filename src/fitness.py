"""
Fitness evaluation.

chromosome (vector of parameters) -> sklearn `clone` of SPAFIS with those parameters
-> fit on TRAIN -> score on VALIDATION.   The test set is never passed to this module.
"""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, fbeta_score, precision_score, recall_score)


def chromosome_to_params(chrom, bounds):
    """Map a gene vector to {parameter_name: value} (order = order of `bounds`)."""
    return {k: float(v) for k, v in zip(bounds.keys(), chrom)}


def params_to_chromosome(params, bounds):
    return np.array([params[k] for k in bounds.keys()], dtype=float)


def fitness_metrics(y_true, y_pred, binary, beta=2.0):
    """Metrics available to the fitness function (scikit-learn implementations)."""
    avg = "binary" if binary else "macro"
    kw = dict(average=avg, zero_division=0)
    m = {"accuracy": accuracy_score(y_true, y_pred),
         "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
         "f1": f1_score(y_true, y_pred, **kw),
         "recall": recall_score(y_true, y_pred, **kw),
         "precision": precision_score(y_true, y_pred, **kw),
         "fbeta": fbeta_score(y_true, y_pred, beta=beta, **kw)}
    if binary:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        m.update({"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)})
    return m


def score_from_metrics(m, cfg):
    """
    Scalar fitness (higher = better).  See FITNESS_CONFIG in config.py.
      fbeta        : F-beta, beta>1 punishes false negatives more than false positives
      recall_floor : recall - penalty * max(0, min_precision - precision)
      composite    : sum_k weight_k * metric_k
      other        : the metric itself
    """
    t = cfg["type"]
    if t == "composite":
        return float(sum(w * m[k] for k, w in cfg["weights"].items()))
    if t == "recall_floor":
        return float(m["recall"] - cfg["penalty"] * max(0.0, cfg["min_precision"] - m["precision"]))
    return float(m[t])


class FitnessEvaluator:
    """
    Callable: chromosome -> scalar fitness.   Holds TRAIN and VALIDATION data only.
    Results are cached (fitness is deterministic), so re-evaluating elites is free.
    """

    def __init__(self, base_model, X_train, y_train, X_val, y_val, bounds, fitness_cfg, binary=True):
        self.base, self.bounds, self.cfg, self.binary = base_model, bounds, fitness_cfg, binary
        self.Xtr, self.ytr, self.Xv, self.yv = X_train, y_train, X_val, y_val
        self._cache = {}

    def metrics(self, chrom):
        key = tuple(np.round(chrom, 12))
        if key not in self._cache:
            model = clone(self.base).set_params(**chromosome_to_params(chrom, self.bounds))
            model.fit(self.Xtr, self.ytr)
            self._cache[key] = fitness_metrics(self.yv, model.predict(self.Xv), self.binary,
                                               self.cfg.get("beta", 2.0))
        return self._cache[key]

    def __call__(self, chrom):
        return score_from_metrics(self.metrics(chrom), self.cfg)

    def objectives(self, chrom):
        """Two objectives for NSGA-II/III selection: [recall, precision] (both maximised)."""
        m = self.metrics(chrom)
        return [float(m["recall"]), float(m["precision"])]
