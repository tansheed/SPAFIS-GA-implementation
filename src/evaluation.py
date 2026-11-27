"""Metrics, comparison table and plots."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)


def compute_metrics(y_true, y_pred, binary=True, labels=None):
    """
    Returns (metrics dict, confusion matrix).
    Binary : precision/recall/F1 of the ATTACK class (label 1) + TP/TN/FP/FN.
    Multi  : precision/recall/F1 are macro averages; weighted_f1 also reported.
    """
    labels = [0, 1] if binary else (labels if labels is not None else np.unique(y_true))
    avg = "binary" if binary else "macro"
    kw = dict(average=avg, zero_division=0)
    m = {"accuracy": accuracy_score(y_true, y_pred),
         "precision": precision_score(y_true, y_pred, **kw),
         "recall": recall_score(y_true, y_pred, **kw),
         "f1": f1_score(y_true, y_pred, **kw),
         "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)}
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if binary:
        tn, fp, fn, tp = cm.ravel()
        m.update({"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)})
    else:
        m.update({"macro_precision": m["precision"], "macro_recall": m["recall"],
                  "macro_f1": m["f1"],
                  "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0)})
    return m, cm


def comparison_table(metrics_a, metrics_b, name_a="SPAFIS", name_b="GA-SPAFIS"):
    """DataFrame (rows = metrics) with the difference column."""
    df = pd.DataFrame({name_a: metrics_a, name_b: metrics_b})
    df["difference"] = df[name_b] - df[name_a]
    return df


def plot_convergence(history, path):
    plt.figure(figsize=(7, 4.5))
    plt.plot(history["generation"], history["best_fitness"], "o-", ms=3, label="Best fitness")
    plt.plot(history["generation"], history["average_fitness"], "s--", ms=3, label="Average fitness")
    plt.xlabel("Generation"); plt.ylabel("Fitness (validation)")
    plt.title("GA convergence"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def plot_confusion_matrices(cm_a, cm_b, class_names, path, name_a="SPAFIS", name_b="GA-SPAFIS"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, cm, title in zip(axes, (cm_a, cm_b), (name_a, name_b)):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f"{title} (test set)"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_comparison(metrics_a, metrics_b, path, name_a="SPAFIS", name_b="GA-SPAFIS"):
    keys = ["accuracy", "precision", "recall", "f1", "balanced_accuracy"]
    x = np.arange(len(keys)); w = 0.38
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - w / 2, [metrics_a[k] for k in keys], w, label=name_a)
    plt.bar(x + w / 2, [metrics_b[k] for k in keys], w, label=name_b)
    plt.xticks(x, ["Accuracy", "Precision", "Recall", "F1", "Bal. Acc."])
    plt.ylim(0, 1.05); plt.ylabel("Score"); plt.title("Test-set performance")
    plt.legend(); plt.grid(axis="y", alpha=.3); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def plot_summary(df, baseline_fn, path):
    """Bar chart of test-set false negatives per GA run (lower is better) + baseline line."""
    d = df[df["status"] == "ok"]
    if d.empty:
        return
    plt.figure(figsize=(max(7, 0.45 * len(d) + 3), 5))
    plt.bar(range(len(d)), d["test_FN"], color="tab:blue")
    plt.axhline(baseline_fn, color="red", ls="--", label=f"default SPAFIS (FN={baseline_fn})")
    plt.xticks(range(len(d)), d["run_id"].str.slice(0, 3) + " " + d["operators"].str.slice(0, 40),
               rotation=90, fontsize=6)
    plt.ylabel("False negatives (test)"); plt.title("Missed attacks per GA configuration")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
