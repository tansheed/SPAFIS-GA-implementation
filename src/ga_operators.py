"""
Custom GA operators for the entries of ga_operators.json that PyGAD does NOT
provide ("pygad": false).  Everything here works on real-valued chromosomes
(vectors of SPAFIS parameters) bounded by [lo, hi].

Crossover  : callable(parents, offspring_size, ga_instance) -> offspring
Mutation   : callable(offspring, ga_instance)               -> offspring
Survivor   : callable(prev_pop, prev_fit, new_pop, new_fit) -> (population, fitness)
             applied after PyGAD has built the next generation; it decides which
             individuals of (previous generation U new generation) survive.
"""
import numpy as np

# ============================================================== crossover
CUSTOM_CROSSOVERS = ("arithmetic", "blx_alpha", "intermediate", "heuristic")


def make_crossover(name, rate, lo, hi, seed, fitness_lookup):
    """
    arithmetic   : child = a*p1 + (1-a)*p2, ONE a ~ U(0,1) per child
    intermediate : per-gene a ~ U(-0.25, 1.25)   (may leave the parents' interval)
    blx_alpha    : per gene, uniform in [min - 0.5*I, max + 0.5*I],  I = |p1 - p2|
    heuristic    : child = best + r*(best - worst), r ~ U(0,1)   (moves away from the worse parent)
    Children are clipped to the gene bounds.  With probability 1-rate the child copies parent 1.
    """
    rng = np.random.default_rng(seed)

    def crossover(parents, offspring_size, ga_instance):
        n_off, n_genes = offspring_size
        out = np.empty(offspring_size)
        for k in range(n_off):
            p1, p2 = parents[k % len(parents)], parents[(k + 1) % len(parents)]
            if rng.random() >= rate:
                out[k] = p1
                continue
            if name == "arithmetic":
                a = rng.random()
                child = a * p1 + (1 - a) * p2
            elif name == "intermediate":
                a = rng.uniform(-0.25, 1.25, n_genes)
                child = a * p1 + (1 - a) * p2
            elif name == "blx_alpha":
                low, high = np.minimum(p1, p2), np.maximum(p1, p2)
                span = high - low
                child = rng.uniform(low - 0.5 * span, high + 0.5 * span)
            elif name == "heuristic":
                better, worse = (p1, p2) if fitness_lookup(p1) >= fitness_lookup(p2) else (p2, p1)
                child = better + rng.random() * (better - worse)
            else:
                raise ValueError(f"No custom crossover '{name}'")
            out[k] = np.clip(child, lo, hi)
        return out

    return crossover


# =============================================================== mutation
CUSTOM_MUTATIONS = ("gaussian", "uniform", "non_uniform", "creep")


def make_mutation(name, rate, lo, hi, seed, total_generations):
    """
    Each gene mutates with probability `rate`:
    gaussian    : g + N(0, 0.1*(hi-lo))
    uniform     : g <- U(lo, hi)                       (replacement)
    creep       : g + U(-0.05, 0.05)*(hi-lo)           (small local step)
    non_uniform : Michalewicz: step shrinks as generations progress,
                  g +/- (bound-g)*(1 - r^((1-t/T)^5))
    Results are clipped to [lo, hi].
    """
    rng = np.random.default_rng(seed + 1)
    span = hi - lo

    def mutation(offspring, ga_instance):
        shape = offspring.shape
        mask = rng.random(shape) < rate
        if name == "gaussian":
            new = offspring + rng.normal(0.0, 0.1 * span, shape)
        elif name == "uniform":
            new = rng.uniform(lo, hi, shape)
        elif name == "creep":
            new = offspring + rng.uniform(-0.05, 0.05, shape) * span
        elif name == "non_uniform":
            t = getattr(ga_instance, "generations_completed", 0)
            shrink = 1.0 - rng.random(shape) ** ((1.0 - t / max(1, total_generations)) ** 5)
            up = rng.random(shape) < 0.5
            new = np.where(up, offspring + (hi - offspring) * shrink,
                           offspring - (offspring - lo) * shrink)
        else:
            raise ValueError(f"No custom mutation '{name}'")
        return np.clip(np.where(mask, new, offspring), lo, hi)

    return mutation


# ====================================================== survivor selection
# keep_elitism / keep_parents are PyGAD parameters (handled in genetic_algorithm.py).
# comma_selection = plain generational replacement (no elites/parents kept) -> no hook.
CUSTOM_SURVIVORS = ("fitness_based", "tournament", "random", "crowding", "age_based",
                    "plus_selection", "comma_selection", "diversity_based")


def _key(c):
    return tuple(np.round(c, 10))


def _unique(pop, fit):
    seen, idx = set(), []
    for i, c in enumerate(pop):
        if _key(c) not in seen:
            seen.add(_key(c))
            idx.append(i)
    return pop[idx], fit[idx]


def make_survivor(name, lo, hi, seed, tournament_k=3, max_age=3):
    """
    Returns fn(prev_pop, prev_fit, new_pop, new_fit) -> (pop, fit) of the SAME size as prev_pop,
    or None when no hook is needed (comma_selection).  The overall best individual is always kept.
      plus_selection  : best mu of (parents U children)                      (mu + lambda)
      fitness_based   : roulette-wheel sampling (no replacement) from the union
      tournament      : repeated k-tournaments over the union (no replacement)
      random          : uniform random sample of the union
      crowding        : deterministic crowding - a child replaces its NEAREST parent if fitter
      age_based       : youngest individuals survive (age > max_age removed first); ties -> fitter
      diversity_based : greedy: best first, then maximise 0.5*fitness + 0.5*distance to chosen set
    """
    if name == "comma_selection":
        return None
    rng = np.random.default_rng(seed + 2)
    span = np.maximum(hi - lo, 1e-12)
    ages = {}

    def dist(U, c):
        return np.linalg.norm((U - c) / span, axis=1)

    def survivor(prev, pf, new, nf):
        n = len(prev)
        U, F = _unique(np.vstack([prev, new]), np.concatenate([pf, nf]))
        m = len(U)
        if name == "crowding":
            pop, fit = prev.copy(), pf.copy()
            for c, fc in zip(new, nf):
                j = int(np.argmin(dist(prev, c)))
                if fc > fit[j]:
                    pop[j], fit[j] = c, fc
            return pop, fit
        if name == "plus_selection":
            idx = np.argsort(-F)[:n]
        elif name == "fitness_based":
            p = F - F.min() + 1e-9
            idx = rng.choice(m, size=min(n, m), replace=False, p=p / p.sum())
        elif name == "random":
            idx = rng.choice(m, size=min(n, m), replace=False)
        elif name == "tournament":
            left, idx = list(range(m)), []
            while left and len(idx) < n:
                cand = rng.choice(left, size=min(tournament_k, len(left)), replace=False)
                w = int(cand[np.argmax(F[cand])])
                idx.append(w)
                left.remove(w)
        elif name == "age_based":
            for c in prev:
                ages[_key(c)] = ages.get(_key(c), 0) + 1       # survivors get older
            for c in new:
                ages.setdefault(_key(c), 0)                     # children are age 0
            age = np.array([ages[_key(c)] for c in U])
            order = sorted(range(m), key=lambda i: (age[i] > max_age, age[i], -F[i]))
            idx = order[:n]
        elif name == "diversity_based":
            fn_ = (F - F.min()) / (np.ptp(F) + 1e-12)
            idx = [int(np.argmax(F))]
            mind = dist(U, U[idx[0]])
            while len(idx) < min(n, m):
                score = 0.5 * fn_ + 0.5 * mind / (mind.max() + 1e-12)
                score[idx] = -np.inf
                j = int(np.argmax(score))
                idx.append(j)
                mind = np.minimum(mind, dist(U, U[j]))
        else:
            raise ValueError(f"No custom survivor selection '{name}'")
        idx = list(idx)
        if int(np.argmax(F)) not in idx:                        # never lose the best individual
            idx[int(np.argmin(F[idx]))] = int(np.argmax(F))
        while len(idx) < n:                                      # pad if the union was too small
            idx.append(int(rng.integers(m)))
        return U[idx], F[idx]

    return survivor
