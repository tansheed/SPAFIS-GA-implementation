"""
Genetic algorithm layer built on PyGAD (https://pygad.readthedocs.io).

A chromosome is a vector of SPAFIS *parameters* (one gene per entry of `bounds`).
The GA never sees training samples or labels: it only receives a fitness
function  chromosome -> float.

Operators come from the JSON catalogue.  Entry flag "pygad": true  -> the name is passed
to PyGAD; "pygad": false -> the custom implementation in ga_operators.py is used.
"""
import numpy as np
import pandas as pd
import pygad

from .ga_operators import make_crossover, make_mutation, make_survivor
from .ga_json import CATEGORIES

NSGA = ("nsga2", "tournament_nsga2", "nsga3", "tournament_nsga3")


def print_ga_setup(ops, bounds, cfg):
    """Print the GA operators and settings once, before the run starts."""
    n = cfg["population_size"]
    print(f"Population size      : {n}  (initial: uniform random inside the bounds"
          f"{' + default parameters seeded' if cfg['include_default_in_population'] else ''})")
    print(f"Generations          : {cfg['generations']}")
    print(f"Chromosome           : {list(bounds.keys())}")
    for k, (lo, hi) in bounds.items():
        print(f"    gene {k:<22} range [{lo}, {hi}]")
    tags = {c: f"{ops[c]['full_name'] if 'full_name' in ops[c] else ops[c]['name']} "
               f"[{'PyGAD' if ops[c]['pygad'] else 'custom'}]" for c in CATEGORIES}
    print(f"Parent selection     : {tags['parent_selection']}, K = {cfg['tournament_size']}, "
          f"{max(2, n // 2)} parents per generation")
    print(f"Crossover            : {tags['crossover']}, rate = {cfg['crossover_rate']}")
    print(f"Mutation             : {tags['mutation']}, per-gene probability = {cfg['mutation_rate']}")
    print(f"Survivor selection   : {tags['survivor_selection']}, elites = {cfg['elite_count']}, "
          f"keep_parents = {cfg['keep_parents']}")
    print(f"Random seed          : {cfg['random_seed']}")


def run_ga(bounds, fitness_fn, cfg, ops, initial_individuals=None, verbose=True):
    """
    Run the GA and return (best_chromosome, best_fitness, history DataFrame).

    bounds     : {gene_name: (low, high)}
    fitness_fn : FitnessEvaluator (callable chromosome -> float; .objectives() for NSGA)
    cfg        : GA_CONFIG dict
    ops        : {category: catalogue entry}   (see ga_json.plan_experiments)
    """
    names = list(bounds.keys())
    lo = np.array([bounds[k][0] for k in names], dtype=float)
    hi = np.array([bounds[k][1] for k in names], dtype=float)
    n_pop, seed = cfg["population_size"], cfg["random_seed"]
    ps, cx, mu, sv = (ops[c] for c in CATEGORIES)
    multi = ps["name"] in NSGA

    # ---- initial population: uniform inside the bounds (+ optional seeded individuals)
    rng = np.random.default_rng(seed)
    init_pop = lo + rng.random((n_pop, len(names))) * (hi - lo)
    for i, ind in enumerate((initial_individuals or [])[:n_pop]):
        init_pop[i] = np.clip(ind, lo, hi)

    clip = lambda c: np.clip(np.asarray(c, dtype=float), lo, hi)   # out-of-range genes (e.g. swap) are clipped
    scalar = lambda pop: np.array([fitness_fn(clip(c)) for c in pop])

    def fitness_func(ga_instance, solution, solution_idx):
        c = clip(solution)
        return [float(v) for v in fitness_fn.objectives(c)] if multi else float(fitness_fn(c))

    # ---- crossover
    crossover = (cx["name"] if cx["pygad"] else
                 make_crossover(cx["name"], cfg["crossover_rate"], lo, hi, seed, lambda c: fitness_fn(clip(c))))

    # ---- mutation
    mut_prob = cfg["mutation_rate"]
    if mu["pygad"]:
        mutation = mu["name"]
        if mutation == "adaptive":                 # [prob for low-fitness, prob for high-fitness] solutions
            mut_prob = [min(1.0, 2 * cfg["mutation_rate"]), cfg["mutation_rate"] / 2]
    else:
        mutation = make_mutation(mu["name"], cfg["mutation_rate"], lo, hi, seed, cfg["generations"])

    # ---- survivor selection
    keep_elitism, keep_parents, survivor_fn = 0, 0, None
    if sv["pygad"]:
        if sv.get("parameter", sv["name"]) == "keep_parents":
            keep_parents = cfg["keep_parents"]
        else:
            keep_elitism = cfg["elite_count"]
    else:
        survivor_fn = make_survivor(sv["name"], lo, hi, seed, cfg["tournament_size"])
        if sv["name"] != "comma_selection":
            keep_elitism = cfg["elite_count"]      # PyGAD builds the next population, hook then filters

    rows, state = [], {}

    def on_fitness(ga_instance, population_fitness):
        """After every population evaluation -> experiment log."""
        pop = ga_instance.population
        fit = scalar(pop)
        state["pop"], state["fit"] = pop.copy(), fit
        b = int(np.argmax(fit))
        chrom = clip(pop[b])
        rows.append({"generation": len(rows) + 1, "best_fitness": fit[b], "average_fitness": fit.mean(),
                     **{f"gene_{k}": v for k, v in zip(names, chrom)},
                     "best_chromosome": np.round(chrom, 6).tolist()})
        if verbose:
            print(f"Generation {len(rows):3d}: best = {fit[b]:.4f}  avg = {fit.mean():.4f}")

    def on_generation(ga_instance):
        """After PyGAD built the next population: apply custom survivor selection."""
        if survivor_fn is None:
            return
        new_pop = ga_instance.population.copy()
        pop, _ = survivor_fn(state["pop"], state["fit"], new_pop, scalar(new_pop))
        ga_instance.population = pop

    extra = {}
    if ps["name"].endswith("nsga3"):
        extra["nsga3_num_divisions"] = cfg["nsga3_num_divisions"]
    n_jobs = cfg.get("n_jobs", 1)
    ga = pygad.GA(
        num_generations=cfg["generations"],
        num_parents_mating=max(2, n_pop // 2),
        sol_per_pop=n_pop,
        num_genes=len(names),
        initial_population=init_pop,
        gene_space=[{"low": float(l), "high": float(h)} for l, h in zip(lo, hi)],
        gene_type=float,
        fitness_func=fitness_func,
        parent_selection_type=ps["name"],
        K_tournament=cfg["tournament_size"],
        keep_elitism=keep_elitism,
        keep_parents=keep_parents,
        crossover_type=crossover,
        crossover_probability=cfg["crossover_rate"],
        mutation_type=mutation,
        mutation_probability=mut_prob,
        on_fitness=on_fitness,
        on_generation=on_generation,
        parallel_processing=["thread", n_jobs] if n_jobs and n_jobs > 1 else None,
        random_seed=seed,
        suppress_warnings=True,
        **extra,
    )
    ga.run()

    hist = pd.DataFrame(rows)
    b = int(hist["best_fitness"].idxmax())       # best individual seen in the log
    best = np.array([hist.loc[b, f"gene_{k}"] for k in names])
    return clip(best), float(hist.loc[b, "best_fitness"]), hist
