"""
SPAFIS vs GA-optimised SPAFIS for network intrusion detection (goal: fewer false negatives).

    python main.py                                   # default operators, one run
    python main.py --quick                           # tiny GA for a smoke test
    python main.py --mode one_at_a_time              # sweep every operator in ga_operators.json
    python main.py --mode grid --max-runs 20         # cartesian product (can be huge!)
    python main.py --mode plan                       # explicit "experiments" list in the JSON
    python main.py --mode one_at_a_time --only rws,sbx,gaussian --list    # dry run, show plan
    python main.py --ga-json other.json --dataset data/KDDTrain.csv --target label --seed 7

Every GA configuration writes to its own sub-folder  results/<run_id>/ ;
results/baseline/ holds the default SPAFIS and results/summary.csv compares all runs.
"""
import argparse
import json
import random
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.base import clone

from src import config as C
from src.evaluation import (comparison_table, compute_metrics, plot_comparison,
                            plot_confusion_matrices, plot_convergence, plot_summary)
from src.fitness import (FitnessEvaluator, chromosome_to_params, fitness_metrics,
                         params_to_chromosome, score_from_metrics)
from src.ga_json import CATEGORIES, load_catalog, plan_experiments
from src.genetic_algorithm import print_ga_setup, run_ga
from src.preprocessing import prepare_data
from src.spafis import SPAFIS


class Tee:
    """Write console output to the screen and to a log file at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-selection", default=None)
    ap.add_argument("--crossover", default=None)
    ap.add_argument("--mutation", default=None)
    ap.add_argument("--survivor-selection", default=None)
    ap.add_argument("--dataset", default=str(C.DATASET_PATH))
    ap.add_argument("--target", default=C.TARGET_COLUMN)
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--population", type=int, default=None)
    ap.add_argument("--multiclass", action="store_true", help="keep all attack classes")
    ap.add_argument("--quick", action="store_true", help="population 8, 5 generations")
    ap.add_argument("--ga-json", default=str(C.GA_JSON_PATH), help="operator catalogue / plan")
    ap.add_argument("--mode", default=None, choices=["single", "one_at_a_time", "grid", "plan"],
                    help="default: 'plan' if the JSON has an 'experiments' list, else 'single'")
    ap.add_argument("--only", default=None, help="comma separated operator names to include")
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--list", action="store_true", help="print the run plan and exit")
    ap.add_argument("--fitness", default=None,
                    choices=["fbeta", "recall_floor", "composite", "recall", "f1", "accuracy", "balanced_accuracy"],
                    help="override FITNESS_CONFIG['type']")
    ap.add_argument("--beta", type=float, default=None, help="beta of the F-beta fitness (default 2)")
    ap.add_argument("--results-dir", default=None, help="output folder (default: results/)")
    return ap.parse_args()


def verdict(m_base, m_ga, eps=0.002):
    """Honest textual conclusion based on TEST metrics (single run!)."""
    caveat = "(single split/seed - repeat with several seeds before drawing conclusions)"
    if "FN" in m_base:
        dfn, dfp = m_ga["FN"] - m_base["FN"], m_ga["FP"] - m_base["FP"]
        head = (f"False negatives {m_base['FN']} -> {m_ga['FN']} ({dfn:+d}), false positives "
                f"{m_base['FP']} -> {m_ga['FP']} ({dfp:+d}), recall {m_base['recall']:.4f} -> "
                f"{m_ga['recall']:.4f}, precision {m_base['precision']:.4f} -> {m_ga['precision']:.4f}.")
        if dfn < 0:
            s = "GA-SPAFIS missed FEWER attacks than default SPAFIS. "
        elif dfn > 0:
            s = "GA-SPAFIS missed MORE attacks than default SPAFIS (no improvement on the main goal). "
        else:
            s = "No change in false negatives. "
        return s + head + " " + caveat
    d = m_ga["recall"] - m_base["recall"]
    s = ("Macro recall improved" if d > eps else "Macro recall did not improve" if d < -eps
         else "No meaningful change in macro recall")
    return f"{s} ({d:+.4f}). " + caveat


def save_run(out, ops_names, ops, hist, best_params, m_base, cm_base, m_ga, cm_ga, class_names,
             best_fit, fit_base, msg, note, seconds):
    out.mkdir(parents=True, exist_ok=True)
    hist.to_csv(out / "ga_history.csv", index=False)
    comparison_table(m_base, m_ga).to_csv(out / "metrics.csv", index_label="metric")
    with open(out / "best_params.json", "w") as fh:
        json.dump({"best_params": best_params, "default_params": C.DEFAULT_SPAFIS_PARAMS,
                   "val_fitness_default": fit_base, "val_fitness_best": best_fit,
                   "conclusion": msg}, fh, indent=2)
    with open(out / "run_config.json", "w") as fh:
        json.dump({"operators": {c: {**ops[c]} for c in CATEGORIES}, "note": note,
                   "seconds": round(seconds, 1), "fitness": C.FITNESS_CONFIG,
                   "ga_settings": {k: v for k, v in C.GA_CONFIG.items() if k != "operators"}},
                  fh, indent=2)
    plot_convergence(hist, out / "ga_convergence.png")
    plot_confusion_matrices(cm_base, cm_ga, class_names, out / "confusion_matrix.png")
    plot_comparison(m_base, m_ga, out / "performance_comparison.png")


def main():
    args = parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    ga_cfg = dict(C.GA_CONFIG, random_seed=seed)
    ga_cfg["operators"] = dict(ga_cfg["operators"])      # don't mutate the shared default
    for cat, val in (("parent_selection", args.parent_selection), ("crossover", args.crossover),
                 ("mutation", args.mutation), ("survivor_selection", args.survivor_selection)):
        if val:
            ga_cfg["operators"][cat] = val
    if args.quick:
        ga_cfg.update(population_size=8, generations=5)
    if args.generations:
        ga_cfg["generations"] = args.generations
    if args.population:
        ga_cfg["population_size"] = args.population
    binary = C.BINARY_TASK and not args.multiclass
    if args.fitness:
        C.FITNESS_CONFIG["type"] = args.fitness
    if args.beta:
        C.FITNESS_CONFIG["beta"] = args.beta
    print("Fitness:", C.FITNESS_CONFIG["type"],
          {k: v for k, v in C.FITNESS_CONFIG.items() if k in ("beta", "min_precision", "penalty")})
    out_root = Path(args.results_dir) if args.results_dir else C.RESULTS_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- which GA configurations to run (parsed from JSON) -----------------------------
    catalog, plan = load_catalog(args.ga_json)
    mode = args.mode or ("plan" if plan else "single")
    only = set(args.only.split(",")) if args.only else None
    runs = plan_experiments(catalog, ga_cfg["operators"], mode, plan, only, args.max_runs)
    print(f"== GA run plan: mode = {mode}, {len(runs)} run(s), catalogue = {args.ga_json} ==")
    for r in runs:
        print(f"  {r['run_id']}")
    if args.list:
        return

    # ---- data -----------------------------------------------------------------------------
    print("\n== Loading and preprocessing ==")
    data = prepare_data(args.dataset, args.target, C.DROP_COLUMNS, binary, C.NORMAL_LABELS,
                        C.SPLIT, seed, C.MAX_TRAIN_SAMPLES, C.MAX_VAL_SAMPLES)
    print(data.info)
    labels = list(range(len(data.class_names)))
    lower = [n.lower() for n in data.class_names]
    normal_class = 0 if binary else (lower.index("normal") if "normal" in lower else None)

    # ---- baseline SPAFIS (computed once, shared by every run) ----------------------
    print("\n== Baseline SPAFIS (default parameters) ==")
    template = SPAFIS(**C.DEFAULT_SPAFIS_PARAMS, **C.SPAFIS_STRUCT, normal_class=normal_class,
                      random_state=seed)
    base = clone(template).fit(data.X_train, data.y_train)
    fit_base = score_from_metrics(
        fitness_metrics(data.y_val, base.predict(data.X_val), binary, C.FITNESS_CONFIG.get("beta", 2.0)),
        C.FITNESS_CONFIG)
    m_base, cm_base = compute_metrics(data.y_test, base.predict(data.X_test), binary, labels)
    print(f"rules = {base.n_rules}, validation fitness = {fit_base:.4f}")
    print("TEST:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m_base.items()})
    bdir = out_root / "baseline"
    bdir.mkdir(exist_ok=True)
    pd.Series(m_base).to_csv(bdir / "metrics.csv", header=["default_SPAFIS"], index_label="metric")
    pd.DataFrame(cm_base).to_csv(bdir / "confusion_matrix.csv", index=False)

    evaluator = FitnessEvaluator(template, data.X_train, data.y_train, data.X_val, data.y_val,
                                 C.PARAMETER_BOUNDS, C.FITNESS_CONFIG, binary)
    seeds = ([params_to_chromosome(C.DEFAULT_SPAFIS_PARAMS, C.PARAMETER_BOUNDS)]
             if ga_cfg["include_default_in_population"] else [])

    # ---- one GA run per configuration ------------------------------------------------------
    summary = []
    for i, run in enumerate(runs, 1):
        ops, names = run["ops"], run["names"]
        opdesc = " | ".join(f"{c}={names[c]}" for c in CATEGORIES)
        out = out_root / run["run_id"]
        out.mkdir(parents=True, exist_ok=True)
        log = open(out / "run_log.txt", "w", encoding="utf-8")
        real_stdout, sys.stdout = sys.stdout, Tee(sys.stdout, log)
        row = {"run_id": run["run_id"], "operators": opdesc, "status": "ok"}
        note = ""
        if names["mutation"] in ("swap", "inversion", "scramble"):
            note = ("permutation-type mutation applied to real-valued genes: values are exchanged "
                    "between different parameters and then clipped to the bounds")
        try:
            print(f"\n===== [{i}/{len(runs)}] {opdesc} =====")
            print("\n== Genetic algorithm ==")
            print_ga_setup(ops, C.PARAMETER_BOUNDS, ga_cfg)
            t0 = time.time()
            best_chrom, best_fit, hist = run_ga(C.PARAMETER_BOUNDS, evaluator, ga_cfg, ops, seeds)
            secs = time.time() - t0
            best_params = chromosome_to_params(best_chrom, C.PARAMETER_BOUNDS)
            print(f"GA finished in {secs:.1f}s; best validation fitness = {best_fit:.4f}")
            print("Best parameters:", json.dumps({k: round(v, 4) for k, v in best_params.items()}))
            opt = clone(template).set_params(**best_params).fit(data.X_train, data.y_train)
            m_ga, cm_ga = compute_metrics(data.y_test, opt.predict(data.X_test), binary, labels)
            msg = verdict(m_base, m_ga)
            print(f"\n{'Metric':<20}{'SPAFIS':>12}{'GA-SPAFIS':>12}")
            for k in m_base:
                f = (lambda v: f"{int(v):>12d}") if k in ("TP", "TN", "FP", "FN") else (lambda v: f"{v:>12.4f}")
                print(f"{k:<20}{f(m_base[k])}{f(m_ga[k])}")
            print("CONCLUSION:", msg)
            save_run(out, names, ops, hist, best_params, m_base, cm_base, m_ga, cm_ga,
                     data.class_names, best_fit, fit_base, msg, note, secs)
            row.update({"val_fitness": best_fit, "rules": opt.n_rules, "seconds": round(secs, 1),
                        "note": note, **{f"test_{k}": v for k, v in m_ga.items()},
                        **{f"param_{k}": v for k, v in best_params.items()}})
            if "FN" in m_base:
                row["delta_FN_vs_default"] = m_ga["FN"] - m_base["FN"]
        except Exception as exc:                       # keep the sweep going, record the failure
            traceback.print_exc(file=sys.stdout)
            row.update({"status": f"failed: {type(exc).__name__}: {str(exc)[:120]}", "note": note})
        finally:
            sys.stdout = real_stdout
            log.close()
        summary.append(row)
        pd.DataFrame(summary).to_csv(out_root / "summary.csv", index=False)   # saved after every run

    df = pd.DataFrame(summary)
    print("\n== SUMMARY (test set; default SPAFIS shown first) ==")
    cols = [c for c in ("run_id", "status", "test_FN", "test_FP", "test_recall", "test_precision",
                        "test_f1", "val_fitness") if c in df.columns]
    show = df[cols].copy()
    base_row = {"run_id": "default SPAFIS", "status": "-", "test_FN": m_base.get("FN"),
                "test_FP": m_base.get("FP"), "test_recall": m_base["recall"],
                "test_precision": m_base["precision"], "test_f1": m_base["f1"], "val_fitness": fit_base}
    show = pd.concat([pd.DataFrame([{k: v for k, v in base_row.items() if k in cols}]), show])
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if "FN" in m_base and "test_FN" in df.columns:
        plot_summary(df, m_base["FN"], out_root / "summary_fn.png")
    print(f"\nResults saved to {out_root}/  (one sub-folder per GA configuration + summary.csv)")


if __name__ == "__main__":
    main()
