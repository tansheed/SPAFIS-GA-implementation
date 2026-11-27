"""
Reads the operator catalogue (ga_operators.json) and turns it into a list of GA runs.

Catalogue format (one list per category, every entry has name / full_name / pygad):
    parent_selection, crossover, mutation, survivor_selection
Optional key "experiments" (explicit plan):
    [{"name": "my_run", "parent_selection": "rws", "crossover": "sbx"}, ...]
    -> unspecified categories fall back to GA_CONFIG["operators"].

Modes
    single         : only the default operators (GA_CONFIG["operators"])
    one_at_a_time  : the defaults + every other catalogue entry, varied one category at a time
    grid           : full cartesian product (can be huge; use --max-runs)
    plan           : the "experiments" list of the JSON
"""
import itertools
import json
import re

CATEGORIES = ("parent_selection", "crossover", "mutation", "survivor_selection")
SHORT = {"parent_selection": "ps", "crossover": "cx", "mutation": "mut", "survivor_selection": "sv"}


def load_catalog(path):
    """Returns ({category: {name: entry}}, experiments_or_None)."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    catalog = {}
    for cat in CATEGORIES:
        entries = raw.get(cat, [])
        for e in entries:
            if "name" not in e or "pygad" not in e:
                raise ValueError(f"{cat}: every entry needs 'name' and 'pygad': {e}")
        catalog[cat] = {e["name"]: e for e in entries}
    return catalog, raw.get("experiments")


def _slug(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", s)


def _run(catalog, names, label=None):
    ops = {}
    for cat in CATEGORIES:
        if names[cat] not in catalog[cat]:
            raise KeyError(f"'{names[cat]}' not found in catalogue category '{cat}'")
        ops[cat] = catalog[cat][names[cat]]
    return {"label": label, "names": dict(names), "ops": ops}


def plan_experiments(catalog, base, mode="single", plan=None, only=None, max_runs=None):
    """Return a list of runs: {run_id, label, names, ops}."""
    runs, seen = [], set()

    def add(names, label=None):
        key = tuple(names[c] for c in CATEGORIES)
        if key in seen:
            return
        seen.add(key)
        runs.append(_run(catalog, names, label))

    add(base, "default")
    if mode == "one_at_a_time":
        for cat in CATEGORIES:
            for name in catalog[cat]:
                if only and name not in only and name != base[cat]:
                    continue
                add({**base, cat: name})
    elif mode == "grid":
        pools = [[n for n in catalog[c] if not only or n in only or n == base[c]] for c in CATEGORIES]
        for combo in itertools.product(*pools):
            add(dict(zip(CATEGORIES, combo)))
    elif mode == "plan":
        if not plan:
            raise ValueError("mode 'plan' needs an 'experiments' list in the JSON file")
        runs, seen = [], set()
        for ex in plan:
            add({**base, **{c: ex[c] for c in CATEGORIES if c in ex}}, ex.get("name"))
    elif mode != "single":
        raise ValueError(f"unknown mode '{mode}'")
    if max_runs:
        runs = runs[:max_runs]
    for i, r in enumerate(runs, 1):
        tag = "__".join(f"{SHORT[c]}-{r['names'][c]}" for c in CATEGORIES)
        r["run_id"] = _slug(f"{i:03d}__{tag}" if not r["label"] or r["label"] == "default"
                            else f"{i:03d}__{r['label']}")
    return runs
