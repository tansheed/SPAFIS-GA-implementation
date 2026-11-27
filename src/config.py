"""
Central configuration for the SPAFIS + Genetic Algorithm experiment.

Everything that a student might want to change lives here: dataset location,
split ratios, SPAFIS default parameters, parameter bounds (= gene ranges),
fitness definition and GA hyper-parameters.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
DATASET_PATH = ROOT / "archive/KDDTrain+.arff"  # .csv or .arff
RESULTS_DIR = ROOT / "results"
TARGET_COLUMN = None          # None -> last column is the label
DROP_COLUMNS = []             # e.g. ["id", "difficulty"] for NSL-KDD
BINARY_TASK = True            # True: normal(0) vs attack(1); False: keep every class
# Label strings (case-insensitive) that mean "normal traffic" in binary mode.
NORMAL_LABELS = ("normal", "benign", "0", "0.0", "legitimate")

RANDOM_SEED = 42              # numpy / random / train_test_split / SPAFIS
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

# Speed control: SPAFIS is trained ~1500 times inside the GA, so the training
# split can be stratified-subsampled ONCE (same subset for every model).
# Samples are never modified. Use None to use the full split.
MAX_TRAIN_SAMPLES = 4000
MAX_VAL_SAMPLES = 2000

# ----------------------------------------------------------------------------
# SPAFIS: tunable parameters (the GA chromosome) -- see README section 3
# ----------------------------------------------------------------------------
DEFAULT_SPAFIS_PARAMS = {
    "prototype_threshold": 0.50,   # min activation to reuse an existing prototype
    "initial_spread": 0.50,        # spread given to a newly created prototype
    "spread_scaling": 1.00,        # multiplies all spreads when computing memberships
    "learning_rate": 0.10,         # step size of centre / spread update
    "rule_weight": 1.00,           # exponent gamma in w_p = (n_p / n_max)^gamma
    "attack_bias": 0.00,           # log-score bonus for every non-normal class (0 = off);
                                   # raising it trades false negatives for false positives
}

# Valid range of every gene (mutation and initialisation respect these).
PARAMETER_BOUNDS = {
    "prototype_threshold": (0.05, 0.95),
    "initial_spread": (0.05, 2.00),
    "spread_scaling": (0.20, 3.00),
    "learning_rate": (0.001, 0.50),
    "rule_weight": (0.00, 2.00),
    "attack_bias": (0.00, 2.00),   # remove this line to stop the GA from tuning the bias
}

# Structural (NOT optimised) settings of SPAFIS.
SPAFIS_STRUCT = {
    "aggregation": "geometric_mean",   # "geometric_mean" | "product" | "min"
    "class_aggregation": "sum",        # "sum" | "max"
    "max_prototypes_per_class": 150,   # safety cap on number of rules per class
    "spread_floor_ratio": 0.25,        # spreads never shrink below ratio*initial_spread
    "n_epochs": 1,                     # passes over the training data
}

# ----------------------------------------------------------------------------
# Fitness
# ----------------------------------------------------------------------------
# Goal: MINIMISE FALSE NEGATIVES (attacks classified as normal).
#   "fbeta"        : F-beta with beta > 1 weights recall more than precision
#                    (beta=2 -> recall counts 4x as much as precision). DEFAULT.
#   "recall_floor" : recall, minus  penalty * max(0, min_precision - precision)
#                    (maximise recall but forbid the 'flag everything' solution)
#   "composite"    : weighted sum of metrics (weights below)
#   "accuracy" | "f1" | "balanced_accuracy" | "recall"
FITNESS_CONFIG = {
    "type": "fbeta",
    "beta": 2.0,
    "min_precision": 0.60,
    "penalty": 2.0,
    "weights": {"f1": 0.4, "balanced_accuracy": 0.3, "recall": 0.3},
}

# ----------------------------------------------------------------------------
# Genetic algorithm
# ----------------------------------------------------------------------------
# Implemented with PyGAD (https://pygad.readthedocs.io): tournament selection,
# elitism (keep_elitism), random-replacement mutation inside gene_space bounds.
GA_JSON_PATH = ROOT / "ga_operators.json"   # operator catalogue (see ga_operators.json)

# Operators used when no experiment plan is requested ("single" mode).
# Names must exist in the JSON catalogue.
GA_CONFIG = {
    "operators": {
        "parent_selection": "tournament",
        "crossover": "arithmetic",
        "mutation": "gaussian",
        "survivor_selection": "keep_elitism",
    },
    "population_size": 30,
    "generations": 50,
    "crossover_rate": 0.8,
    "mutation_rate": 0.1,
    "tournament_size": 3,              # K for tournament-type selections
    "elite_count": 2,                  # keep_elitism / elitism inside custom survivor schemes
    "keep_parents": 2,                 # used by survivor_selection = keep_parents
    "nsga3_num_divisions": 4,          # used by nsga3 parent selections
    "random_seed": RANDOM_SEED,
    "include_default_in_population": True,  # seed one individual with the defaults
    "n_jobs": 1,                       # >1: PyGAD parallel_processing=["thread", n_jobs]
}
