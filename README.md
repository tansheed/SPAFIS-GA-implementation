# SPAFIS + Genetic Algorithm for Network Intrusion Detection

A research-oriented implementation of a **Soft Prototype-based Autonomous Fuzzy Inference
System (SPAFIS)** whose **parameters** are tuned by a **Genetic Algorithm (GA)**, built on
established libraries.

| Library | Used for |
|---|---|
| **PyGAD** | GA engine: population, tournament selection, elitism, mutation, generation loop |
| **scikit-fuzzy** | Gaussian membership function (`skfuzzy.gaussmf`) |
| **scikit-learn** | SPAFIS is a `BaseEstimator`/`ClassifierMixin` (`clone`, `set_params`), preprocessing (`ColumnTransformer`, imputers, `OneHotEncoder`, `MinMaxScaler`), stratified splits, all metrics |
| **SciPy** | `logsumexp` for class aggregation, ARFF loading |

The SPAFIS rule logic itself (prototype creation/update, rule weights) is custom code, since no
library implements it. Blend crossover is a ~10-line custom function passed to PyGAD, because PyGAD has no
arithmetic crossover built in (built-in types can be chosen via `GA_CONFIG["crossover_type"]`).

> The GA optimises **model parameters**, never the data. Training, validation and test samples
> are never generated, edited or re-labelled.

## 1. Quick start

```bash
pip install -r requirements.txt
python main.py                               # default operators, one GA run (~5 min on 1 CPU)
python main.py --mode one_at_a_time --list   # show the sweep plan (36 runs) without running
python main.py --mode one_at_a_time          # run every operator in ga_operators.json
python main.py --fitness recall_floor        # push harder on false negatives
python main.py --quick                       # smoke test (population 8, 5 generations)
python main.py --dataset data/KDDTrain.csv --target class --seed 7
```

Outputs: `results/baseline/` (default SPAFIS), one sub-folder per GA configuration
`results/<NNN>__ps-..__cx-..__mut-..__sv-../` (with `ga_history.csv`, `metrics.csv`, `best_params.json`,
`run_config.json`, `run_log.txt`, `ga_convergence.png`, `confusion_matrix.png`,
`performance_comparison.png`) and `results/summary.csv` + `summary_fn.png` comparing all runs.

**Using a real dataset** (NSL-KDD, UNSW-NB15, CIC-IDS...): set `DATASET_PATH`, `TARGET_COLUMN`
(default: last column), `DROP_COLUMNS`, and `NORMAL_LABELS` in `src/config.py`.
Columns are not hard-coded: numeric columns are detected by dtype, everything else is one-hot encoded.
`BINARY_TASK=True` maps normal -> 0 and every attack type -> 1; set it to `False`
(or pass `--multiclass`) to keep all classes (macro/weighted metrics are then reported).

## 2. Project layout

| File                       | Role                                                                                                          |
| -------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `src/config.py`            | all settings: dataset, split, default parameters, **bounds**, fitness, GA                                     |
| `src/preprocessing.py`     | load, impute, encode, scale, stratified split (fit on train only)                                             |
| `src/fuzzy.py`             | scikit-fuzzy Gaussian membership + log-domain aggregation (`product`, `geometric_mean`, `min`)                |
| `src/prototype.py`         | `Prototype` dataclass = one fuzzy rule                                                                        |
| `src/spafis.py`            | sklearn-compatible `SPAFIS` estimator: prototype creation/update, rule weights, inference                     |
| `src/fitness.py`           | chromosome -> SPAFIS -> validation score                                                                      |
| `src/genetic_algorithm.py` | thin PyGAD wrapper: `run_ga(...)`, history logging, blend crossover                                           |
| `src/evaluation.py`        | metrics, comparison table, plots                                                                              |
| `main.py`                  | the full experiment                                                                                           |
| `archive`                  | NSL-KDD dataset. This is gitignored, you can donwload it here:https://www.kaggle.com/datasets/hassan06/nslkdd |

## 3. The SPAFIS algorithm

Each **prototype** `p` stores `center` (d-vector), `spread` (d-vector), `weight`, `class_label`.
It is one fuzzy rule: *IF x is similar to center_p THEN class = class_label*.

**Membership** (Gaussian, `skfuzzy.gaussmf`):

```
mu_j(x) = exp( -(x_j - c_j)^2 / (2 * s_j^2) )        s_j >= MIN_SPREAD = 1e-6
```

**Rule activation** = aggregation of the d memberships. Default is the geometric mean
`(prod_j mu_j)^(1/d)` because a plain product underflows for ~100 features. Everything is computed
as sums of logs, so no underflow occurs. Switch with `SPAFIS_STRUCT["aggregation"]`.

**Class activation**: `score_c(x) = sum_{p in class c} w_p * a_p(x)` (or `max`), computed with
`scipy.special.logsumexp`. Prediction = `argmax_c score_c(x)`.

**Autonomous learning** (one online pass in a seeded random order):
for sample `(x, y)` look at prototypes of class `y` only. If the best activation is
`>= prototype_threshold` (or the per-class cap is reached) the winner is updated:

```
c   <- c + lr * (x - c)
s^2 <- (1 - lr) * s^2 + lr * (x - c_old)^2          (never below spread_floor_ratio * initial_spread)
```

otherwise a **new prototype** is created with `center = x`, `spread = initial_spread`.
After the pass the rule weight is `w_p = (n_p / n_max,class)^rule_weight`.

> This is a transparent, simplified student implementation of the soft-prototype idea. It is not a
> line-by-line reproduction of any published SPAFIS paper; adapt `spafis.py` if you need the exact
> published update equations.

## 4. The chromosome

One gene per tunable SPAFIS parameter (order = `PARAMETER_BOUNDS`):

```
[ prototype_threshold, initial_spread, spread_scaling, learning_rate, rule_weight, attack_bias ]
```

| Gene | Effect |
|---|---|
| `prototype_threshold` | how similar a sample must be to reuse a prototype -> number of rules |
| `initial_spread` | width of a freshly created prototype |
| `spread_scaling` | global multiplier of all spreads in the membership function |
| `learning_rate` | speed of centre/spread adaptation |
| `rule_weight` | exponent that turns prototype support counts into rule weights |
| `attack_bias` | constant added to the log score of every non-normal class: 0 = plain SPAFIS, larger = flag more traffic as attack (**the main lever against false negatives**) |

Delete the `attack_bias` line from `PARAMETER_BOUNDS` to remove it from the search. No sample, label or
index is ever a gene.

## 5. The Genetic Algorithm (PyGAD) and the operator catalogue

`ga_operators.json` lists the available operators in four categories: `parent_selection`, `crossover`,
`mutation`, `survivor_selection`. Every entry has a `pygad` flag:

* `"pygad": true`  -> the name is passed to PyGAD (for survivor selection, the `parameter` field says which PyGAD
  setting is used: `keep_elitism` or `keep_parents`).
* `"pygad": false` -> custom implementation in `src/ga_operators.py` (crossover: arithmetic, blx_alpha, intermediate,
  heuristic; mutation: gaussian, uniform, non_uniform, creep; survivor: fitness_based, tournament, random, crowding,
  age_based, plus_selection, comma_selection, diversity_based). Survivor schemes run as a hook after PyGAD builds
  the next population and choose among (previous generation + new generation); the best individual is always kept.

`src/ga_json.py` parses the file and builds the run list. **Modes** (`--mode`):

| Mode | Runs |
|---|---|
| `single` (default) | only `GA_CONFIG["operators"]` |
| `one_at_a_time` | defaults + each other catalogue entry, changing one category at a time (36 runs for the shipped JSON) |
| `grid` | full cartesian product - huge; limit with `--max-runs` |
| `plan` | the optional `"experiments"` list in the JSON, e.g. `[{"name":"rws_sbx","parent_selection":"rws","crossover":"sbx"}]` (unspecified categories use the defaults) |

`--only rws,sbx,gaussian` restricts the sweep to the named operators; `--list` prints the plan and exits.
Each run writes to its own sub-folder; a failing run is recorded in `summary.csv` and the sweep continues.

Notes: NSGA-II/III parent selection is multi-objective, so the GA optimises `[recall, precision]` for those runs
(the logged fitness is still the scalar fitness). `swap`, `inversion`, `scramble` mutation are permutation operators:
on real-valued genes they exchange values between different parameters, which are then clipped to the bounds
(flagged in `run_config.json`). Fitness values are memoised.

## 6. The fitness function (focus: minimise false negatives)

For every chromosome: `clone` the SPAFIS estimator with the chromosome's parameters, train it on the **training**
split, predict the **validation** split, score. Types (`FITNESS_CONFIG["type"]`, or `--fitness`, `--beta`):

* `fbeta` (default): F-beta with beta = 2, recall counts 4x as much as precision. Raise beta to push harder.
* `recall_floor`: `recall - penalty * max(0, min_precision - precision)`: maximise recall but forbid the trivial
  "flag everything" solution. Stricter than `fbeta` on false negatives.
* `composite` (`0.4*F1 + 0.3*BalAcc + 0.3*Recall`), `accuracy`, `f1`, `balanced_accuracy`, `recall`.

The test set is never passed to the GA or the fitness evaluator.

## 7. Why the GA does not modify the training data

```
Training data (fixed) --> SPAFIS(theta) --> validation score --> GA proposes new theta
```

The training matrix `X_train` and labels `y_train` are read-only inputs to `SPAFIS.fit`
(they are copied/indexed but never written to). The GA only produces and mutates the parameter
vector `theta`. Hence a chromosome = *candidate SPAFIS parameter set*, **not** a modified
dataset. The optional `MAX_TRAIN_SAMPLES` setting only *selects* a fixed stratified subset once, before
any model is built; values and labels are untouched and every model sees the same subset.

## 8. Leakage control

* Split first (stratified 70/15/15, seeded); imputers, one-hot encoder and min-max scaler are fitted
  on the training split only (`clip=True` keeps unseen values in [0,1]).
* Training split -> SPAFIS training; validation split -> GA fitness; test split -> final evaluation only.
* Default SPAFIS and GA-SPAFIS are compared on the identical test set.

## 9. Interpreting the results

`results/metrics.csv` lists every metric for SPAFIS, GA-SPAFIS and their difference (binary: attack is
the positive class, with TP/TN/FP/FN; multi-class: macro precision/recall/F1 and weighted F1).

* `ga_convergence.png`: best fitness is non-decreasing (elitism). A plateau means the search converged.
* Compare **test** metrics, not validation fitness: validation fitness is what the GA optimised, so it
  is optimistically biased. A large validation gain with no test gain indicates overfitting to the
  validation set.
* On imbalanced data prefer F1, recall and balanced accuracy over accuracy.
* `main.py` prints a conclusion that says "improved" only if both test F1 and balanced accuracy rise by
  more than 0.002. A single run is one split and one seed: repeat with several `--seed` values
  (and ideally report mean +- std) before claiming GA helps.
* The GA-SPAFIS model is built on a *different* parameter set, so it may also use many more rules
  (`Rules:` line in the output): consider this a cost in model complexity and inference time.

## 10. Reproducibility

`RANDOM_SEED` seeds `random`, `numpy`, `train_test_split`, SPAFIS sample ordering and PyGAD (`random_seed`). The same
configuration gives identical results (fitness evaluation is deterministic).

## 11. Synthetic demo data

`data/generate_synthetic_dataset.py` writes a KDD-like toy dataset. Results obtained on it say nothing
about real intrusion-detection performance; use a real benchmark for research claims.
