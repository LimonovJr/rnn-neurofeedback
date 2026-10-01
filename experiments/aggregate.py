"""Turn the per-run CSVs into the tables the report quotes. Aggregation is deliberately separate
from the runs, so every table can be rebuilt without retraining anything.

    python -m experiments.aggregate headroom
    python -m experiments.aggregate nf-context --reps 0,1
"""

import argparse
import statistics as st
from collections import defaultdict

import numpy as np

from src import behaviour, runs, stats
from src.signals import LR

# what a repeat of the same measurement moves by, from `resolutions` on the networks and heads the
# experiments run on: two standard deviations over ten repeats. The campaign was read against 0.05,
# 0.0086 and 0.060, which came from an earlier checkpoint whose crossnobis scale was ten times
# larger; every table below uses the measured ones.
RES_BEHAVIOUR = 0.068
RES_CROSSNOBIS = 0.0015
RES_PR = 0.11

SETTINGS = ["c1", "c1r1", "c1r3", "patch"]


def curve(row):
    return runs.split(row["curve_upd"], int), runs.split(row["curve_dprime"])


def by_net(rows, value):
    out = defaultdict(list)
    for r in rows:
        out[int(r["net"])].append(value(r))
    return out


def headroom(args):
    print("setting        n   C_A    C_B    H(med)   SEM      p   H(max)   H/C_B  res")
    for setting in SETTINGS:
        rows = runs.read(f"{args.results}/headroom_{setting}_*_net*.csv")
        arms = {a: by_net([r for r in rows if r["arm"] == a], lambda r: float(r["C_mean"]))
                for a in ("A", "B")}
        nets = sorted(set(arms["A"]) & set(arms["B"]))
        if not nets:
            continue

        med = [st.median(arms["B"][n]) - st.median(arms["A"][n]) for n in nets]
        mx = [max(arms["B"][n]) - max(arms["A"][n]) for n in nets]
        fraction = [(st.median(arms["B"][n]) - st.median(arms["A"][n])) / st.median(arms["B"][n])
                    for n in nets]
        c_a = np.mean([st.median(arms["A"][n]) for n in nets])
        c_b = np.mean([st.median(arms["B"][n]) for n in nets])
        mean, sem, p = stats.one_sample(med)
        res = float(np.median([float(r["se_C"]) for r in rows]))
        print(f"{setting:<12}{len(nets):>3}{c_a:>7.3f}{c_b:>7.3f}{mean:>+9.3f}{sem:>7.3f}"
              f"{p:>8.4f}{np.mean(mx):>+8.3f}{np.mean(fraction):>8.3f}{res:>7.3f}  "
              f"{stats.verdict(mean, p, res)}")

    rows = runs.read(f"{args.results}/headroom_c1_*_net*.csv")
    plus = by_net([r for r in rows if r["arm"] == "Aplus"], lambda r: float(r["C_mean"]))
    linear = by_net([r for r in rows if r["arm"] == "A"], lambda r: float(r["C_mean"]))
    if plus:
        nets = sorted(set(plus) & set(linear))
        mean, sem, p = stats.paired([st.median(plus[n]) for n in nets],
                                    [st.median(linear[n]) for n in nets])
        print(f"\nnonlinear read-out on the same frozen dynamics: "
              f"C = {np.mean([st.median(plus[n]) for n in nets]):.3f}, "
              f"{mean:+.3f} +/- {sem:.3f} over the linear read-out, p = {p:.4f}")

    drift = [float(r["wrec_drift"]) for r in runs.read(f"{args.results}/headroom_*_A*_net*.csv")]
    print(f"frozen-arm drift: max ||dW_rec||inf = {max(drift):.1e}")


def control(args):
    rows = runs.read(f"{args.results}/control_net*.csv")
    arms = {a: by_net([r for r in rows if r["arm"] == a], lambda r: float(r["reeval"]))
            for a in ("A", "B")}
    nets = sorted(set(arms["A"]) & set(arms["B"]))

    print(f"context reversal, {len(nets)} networks")
    print(f"{'net':<5}{'theta0':>8}{'C_A med':>10}{'C_B med':>10}{'gap med':>10}{'gap max':>10}")
    med, mx = [], []
    for net in nets:
        theta0 = float(next(r["theta0_d"] for r in rows if int(r["net"]) == net))
        med.append(st.median(arms["B"][net]) - st.median(arms["A"][net]))
        mx.append(max(arms["B"][net]) - max(arms["A"][net]))
        print(f"{net:<5}{theta0:>+8.3f}{st.median(arms['A'][net]):>10.3f}"
              f"{st.median(arms['B'][net]):>10.3f}{med[-1]:>+10.3f}{mx[-1]:>+10.3f}")

    for name, gaps in (("median", med), ("max", mx)):
        mean, sem, p = stats.one_sample(gaps)
        print(f"  gap by {name:<7}= {mean:+.3f} +/- {sem:.3f}, p = {p:.4f}")
    print("  read for its order of magnitude, not its statistics: three networks, and the arms "
          "carry unequal restart counts")


def degradation(args):
    rows = runs.read(f"{args.results}/degradation_net*.csv")
    ladder = [r for r in rows if r["knob"] == "input" and r["sigma"] != ""]
    intact = {int(r["net"]): float(r["dprime"]) for r in ladder if float(r["sigma"]) == 0.0}
    print("sigma_in    d' (mean over nets)    s = 1 - d/d(intact)")
    for sigma in sorted({float(r["sigma"]) for r in ladder}):
        cell = [r for r in ladder if float(r["sigma"]) == sigma]
        d = [float(r["dprime"]) for r in cell]
        s = [1 - float(r["dprime"]) / intact[int(r["net"])] for r in cell if int(r["net"]) in intact]
        loss = f"{np.mean(s):>22.3f}" if s else " " * 22
        print(f"{sigma:>6.2f}{np.mean(d):>18.3f}{loss}    n = {len(cell)}")

    untrained = [float(r["dprime"]) for r in rows if r["knob"] == "untrained"]
    if untrained:
        print(f"\nuntrained networks sit at d' = {np.mean(untrained):.3f}, which is the clamp "
              f"rather than a point on the axis")


def patch(args):
    rows = runs.read(f"{args.results}/patch_net*.csv")
    print("jitter  T_avg   d' (mean over nets)   kind")
    for kind in sorted({r["kind"] for r in rows}):
        for key in sorted({(float(r["jitter"]), float(r["t_avg"])) for r in rows
                           if r["kind"] == kind}):
            d = [float(r["dprime"]) for r in rows if r["kind"] == kind
                 and (float(r["jitter"]), float(r["t_avg"])) == key]
            print(f"{key[0]:>6.2f}{key[1]:>7.0f}{np.mean(d):>15.3f}   {kind}")

    gains = defaultdict(dict)
    for r in rows:
        if r["kind"] == "patch":
            gains[float(r["jitter"])][float(r["t_avg"])] = gains[float(r["jitter"])].get(
                float(r["t_avg"]), []) + [float(r["dprime"])]
    print("\naveraging gain (T_avg 20 minus T_avg 100), per jitter level:")
    for jitter, by_t in sorted(gains.items()):
        if 20.0 in by_t and 100.0 in by_t:
            mean, sem, p = stats.paired(by_t[20.0], by_t[100.0])
            print(f"  jitter {jitter:.2f}: {mean:+.3f} +/- {sem:.3f}, p = {p:.4f}")


def resolutions(args):
    """Two standard deviations over repeat measurements of one frozen model, per quantity."""
    rows = runs.read(f"{args.results}/resolution_*_net*.csv")
    for task in sorted({r["task"] for r in rows}):
        task_rows = [r for r in rows if r["task"] == task]
        print(f"{task}: {len(task_rows)} repeats")
        for key in ("dprime", "crossnobis", "pr", "context", "match", "xor"):
            values = [float(r[key]) for r in task_rows if r.get(key) not in (None, "")]
            if len(values) > 1:
                print(f"  {key:<12}mean {np.mean(values):>8.4f}   resolution (2 SD) "
                      f"{stats.resolution(values):>8.4f}")
        paired = [(float(r["dprime"]), float(r["dprime_paired"])) for r in task_rows
                  if r.get("dprime_paired")]
        if len(paired) > 1:
            diffs = [a - b for a, b in paired]
            print(f"  evaluation SD at the reporting sample size: "
                  f"{np.std(diffs, ddof=1) / np.sqrt(2):.4f} over {len(diffs)} paired evaluations")


def competence(args):
    """Where behaviour leaves the floor, whether that point reproduces, and whether the
    representation moves before it does."""
    rows = runs.read(f"{args.results}/competence_net*_r*.csv")
    crossings = defaultdict(dict)
    above = total = 0
    for r in rows:
        above += abs(float(r["crossnobis"])) > RES_CROSSNOBIS
        total += 1
    for key in sorted({(int(r["net"]), int(r["rep"])) for r in rows}):
        net, rep = key
        run_rows = sorted([r for r in rows if (int(r["net"]), int(r["rep"])) == key],
                          key=lambda r: int(r["upd"]))
        first = next((int(r["upd"]) for r in run_rows if float(r["dprime"]) > 0.0), None)
        crossings[net][rep] = first
        print(f"net {net} rep {rep}: first update with behaviour above the floor = {first}")

    shifts = [abs(v[0] - v[1]) for v in crossings.values()
              if len(v) > 1 and None not in (v.get(0), v.get(1))]
    if shifts:
        print(f"crossing shift between repeats: mean {np.mean(shifts):.0f}, max {max(shifts):.0f} updates")
    print(f"crossnobis above its resolution: {above}/{total} sampled points")


def labelfree(args):
    """The label-free question: does the positional term help a reward-only learner? reward_aux
    against reward on the same networks at the same budget, level and updates-to-level."""
    reps = [int(s) for s in args.reps.split(",")]
    for task in ("c1", "context"):
        rows = [r for r in runs.read(f"{args.results}/nf_{task}_reward*_net*.csv")
                if int(r["eval_n"]) == 10000 and published(r, target=r.get("target"))]
        base = levels([r for r in rows if r["cond"] == "reward"], reps)
        if not base:
            continue
        print(f"\n{task}: reward-only level {np.mean(list(base.values())):.3f} over {len(base)} networks")
        for lam, target in sorted({(r["lam"], r.get("target", "target")) for r in rows
                                   if r["cond"] == "reward_aux"}):
            aux = levels([r for r in rows if r["cond"] == "reward_aux"
                          and (r["lam"], r.get("target", "target")) == (lam, target)], reps)
            nets = sorted(set(aux) & set(base))
            if len(nets) < 2:
                continue
            mean, sem, p = stats.paired([aux[n] for n in nets], [base[n] for n in nets])
            print(f"  + positional term (lam {lam}, {target}): level "
                  f"{np.mean([aux[n] for n in nets]):.3f}, {mean:+.3f} +/- {sem:.3f} against "
                  f"reward-only, p = {p:.4f}, {stats.verdict(mean, p, RES_BEHAVIOUR)}")


def manifold(args):
    """Relearning after a within- or outside-manifold perturbation of the decision read-out: the
    immediate drop, the level recovered, and the update at which the network is back within
    resolution of its intact self."""
    rows = runs.read(f"{args.results}/manifold_*_net*.csv")
    print(f"{'kind':<9}{'cond':<9}{'angle':>6}{'steps':>6}{'n':>4}{'intact':>8}{'dropped':>9}"
          f"{'recovered':>11}   back within resolution")
    for key in sorted({(r["kind"], r["cond"], float(r["angle"]), int(r["steps"])) for r in rows}):
        cell = [r for r in rows if (r["kind"], r["cond"], float(r["angle"]), int(r["steps"])) == key]
        recovered, back = [], []
        for r in cell:
            upds, values = curve(r)
            recovered.append(float(np.mean(values[-3:])))
            target = float(r["d_intact"]) - RES_BEHAVIOUR
            back.append(next((u for u, v in zip(upds, values) if v >= target), None))
        settled = [b for b in back if b is not None]
        when = f"{np.median(settled):.0f} updates ({len(settled)}/{len(cell)})" if settled else "never"
        print(f"{key[0]:<9}{key[1]:<9}{key[2]:>6g}{key[3]:>6}{len(cell):>4}"
              f"{np.mean([float(r['d_intact']) for r in cell]):>8.3f}"
              f"{np.mean([float(r['d_perturbed']) for r in cell]):>9.3f}"
              f"{np.mean(recovered):>11.3f}   {when}")


def nf_c1(args):
    """Signals against the equal-budget control on c1, at the evaluation size given by --eval-n:
    1000 is the published stage with its representation panel, 10000 the properly measured
    behaviour. Conditions that have no runs at that size are left out."""
    rows = runs.read(f"{args.results}/nf_c1_*_n{args.eval_n}_net*.csv")
    rows = [r for r in rows if published(r)]
    for r in rows:                      # the expansion arms differ only in the gain they ask for
        if r["cond"] == "expand":
            r["cond"] = f"expand k{float(r.get('kappa') or 1.5):g}"
    conds = [c for c in ("control", "position", "joint", "reward", "reward_aux", "gated")
             if any(r["cond"] == c for r in rows)]
    conds += sorted({r["cond"] for r in rows if r["cond"].startswith("expand")})
    if "control" not in conds or len(conds) < 2:
        print("no control at this evaluation size, or nothing to compare it with")
        return

    coupling, level, cross, pr, moved = {}, {}, {}, {}, {}
    for cond in conds:
        cond_rows = [r for r in rows if r["cond"] == cond and (cond != "joint" or r["lam"] in ("1", "1.0"))]
        by = by_net(cond_rows, lambda r: r)
        coupling[cond], level[cond], cross[cond], pr[cond], moved[cond] = {}, {}, {}, {}, {}
        for net, group in by.items():
            values = []
            for r in group:
                upds, curve_values = curve(r)
                values.append((np.median([v for u, v in zip(upds, curve_values) if u >= 600]),
                               np.mean(curve_values[-6:])))
            coupling[cond][net] = float(np.mean([v[0] for v in values]))
            level[cond][net] = float(np.mean([v[1] for v in values]))
            if all(r.get("dW_rec") for r in group):
                moved[cond][net] = float(np.mean([float(r["dW_rec"]) for r in group]))
            if all(r["crossnobis"] for r in group):
                cross[cond][net] = float(np.mean([float(r["crossnobis"]) for r in group]))
                pr[cond][net] = float(np.mean([float(r["pr"]) for r in group]))

    print(f"evaluation on {args.eval_n} trials; per-network values averaged over repeats")
    print(f"{'quantity':<14}" + "".join(f"{c:>26}" for c in conds if c != "control") + "   resolution")
    for name, data, res in (("level", level, RES_BEHAVIOUR), ("coupling", coupling, RES_BEHAVIOUR),
                            ("crossnobis", cross, RES_CROSSNOBIS), ("PR", pr, RES_PR)):
        cells = []
        for cond in conds:
            if cond == "control":
                continue
            nets = sorted(set(data[cond]) & set(data["control"]))
            if len(nets) < 2:
                cells.append(f"{'(no data)':>26}")
                continue
            mean, sem, p = stats.paired([data[cond][n] for n in nets], [data["control"][n] for n in nets])
            cells.append(f"{mean:+.4f} p={p:.3f} {stats.verdict(mean, p, res)[:3]:>3}")
        print(f"{name:<14}" + "".join(f"{c:>26}" for c in cells) + f"   {res}")

    if any(moved[c] for c in conds):
        print("\nhow far the recurrent weights moved, relative to their starting norm:")
        for cond in conds:
            if moved[cond]:
                print(f"  {cond:<12}{np.mean(list(moved[cond].values())):.4f}")

    print("\nupdates to reach the coupling level, per condition:")
    for cond in conds:
        reached = []
        cond_rows = [r for r in rows if r["cond"] == cond and (cond != "joint" or r["lam"] in ("1", "1.0"))]
        for r in sorted(cond_rows, key=lambda r: (int(r["net"]), int(r["rep"]))):
            upds, values = curve(r)
            target = float(np.median([v for u, v in zip(upds, values) if u >= 600]))
            reached.append(stats.first_within(upds[1:], values[1:], target, RES_BEHAVIOUR))
        print(f"  {cond:<11}{reached}")

    start = [float(r["d0"]) for r in rows if r["cond"] == "control" and r["d0"]]
    if start and "reward" in coupling:
        nets = sorted(set(coupling["reward"]) & set(coupling["control"]))
        print(f"\nreward against a zero-update baseline: "
              f"{np.mean([coupling['reward'][n] for n in nets]) - np.mean(start):+.3f} "
              f"(against the equal-budget control: "
              f"{np.mean([coupling['reward'][n] - coupling['control'][n] for n in nets]):+.3f})")


BUDGET = {"c1": 1200, "context": 8000}
EVAL_N = {"c1": 1000, "context": 10000}

# Every column a run writes says what the run is, what it measured, or how it was run. The last
# kind is compared against its default, and a column that is in none of these lists counts as a
# changed protocol: a knob added later keeps its runs out of every table until a table names it.
IDENTITY = {"task", "cond", "net", "rep", "_file"}
OUTCOMES = {"d0", "clipped", "kl_task", "crossnobis", "pr", "crossnobis_start", "pr_start",
            "axis_in_manifold", "imbalance_late"}
OUTCOME_PREFIXES = ("pretrain_", "patch_", "axis_auc_", "axis_m_", "crit_", "auc_", "roc_", "dW_",
                    "curve_", "offset_traj", "imbalance_traj", "hit_rate_traj", "abort_traj")
DEFAULTS = {"lam": "1", "updates": lambda r: BUDGET[r["task"]], "lr": lambda r: LR[r["cond"]],
            "eval_n": lambda r: EVAL_N[r["task"]], "train_decoder": "0", "target": "target",
            "anchor": "0", "kappa": "1.5", "shuffle": "0", "adapt_decoder": "0", "rule": "bptt",
            "explore": "1", "explore_k": "0", "rec_noise_scale": "1", "optimizer": "adam",
            "all_trials": "0", "fa_cost": "1", "fa_penalty": "", "operant_reward": "position",
            "cosine": "", "axis_increment": "", "refit_decoder": "0", "entropy": "0.05",
            "save_at": "", "track": "", "track_rate": "", "yoke_from": "", "baseline": "batch",
            "batch": "48", "temperature": "1", "track_every": "", "track_step": "",
            "baseline_decay": "", "focus_theta": "", "focus_width": ""}
# what the published tables were built from, and may pool
PUBLISHED = {"lam", "eval_n", "train_decoder", "kappa", "shuffle", "explore", "explore_k",
             "optimizer", "all_trials", "fa_cost", "fa_penalty", "operant_reward", "cosine",
             "axis_increment", "target"}
SWEPT = PUBLISHED | {"updates", "lr", "anchor", "adapt_decoder", "rule", "rec_noise_scale"}


def same(value, default):
    if value in ("", None):
        return True
    try:
        return float(value) == float(default)
    except (TypeError, ValueError):
        return str(value) == str(default)


def protocol(r):
    """The knobs a run changed from their defaults."""
    out = set()
    for k, v in r.items():
        if k in IDENTITY or k in OUTCOMES or k.startswith(OUTCOME_PREFIXES):
            continue
        default = DEFAULTS.get(k)
        if default is None:
            if v not in ("", None):
                out.add(k)
        elif not same(v, default(r) if callable(default) else default):
            out.add(k)
    return out


def published(r, target="target"):
    """The registered protocol: the stimulus target, no anchor, no decoder co-adaptation,
    back-propagation, the condition's own learning rate, the full budget, and nothing a later
    block varied. The repeats run for other questions (swept thresholds, weight movement,
    retention) are left out too, so the published tables stay what they were."""
    return (protocol(r) <= PUBLISHED and r.get("target") in (target, "", None)
            and "_move_" not in r["_file"] and not r.get("crit_d_end") and not r.get("kl_task"))


def levels(rows, reps, k=6):
    """Level at the budget per network: mean of the last k re-evaluations, averaged over repeats."""
    out = defaultdict(list)
    for r in rows:
        if int(r["rep"]) not in reps:
            continue
        _, values = curve(r)
        out[int(r["net"])].append(stats.level(values, k)[0])
    return {net: float(np.mean(v)) for net, v in out.items()}


def nf_context(args):
    rows = [r for r in runs.read(f"{args.results}/nf_context_*_lam1_*_net*.csv") if published(r)]
    conds = ("control", "position", "joint", "reward")
    reps = [int(s) for s in args.reps.split(",")]
    table = {c: levels([r for r in rows if r["cond"] == c], reps) for c in conds}
    nets = sorted(set.intersection(*[set(v) for v in table.values()]))
    if not nets:
        print("no network has all four conditions yet")
        return

    print("net " + "".join(f"{c:>10}" for c in conds))
    for net in nets:
        print(f"{net:<4}" + "".join(f"{table[c][net]:>10.3f}" for c in conds))
    print("mean" + "".join(f"{np.mean([table[c][n] for n in nets]):>10.3f}" for c in conds))

    # the threshold is the replicate spread, which is coarser than the within-model resolution
    print("\nruns whose residual bias is below 0.0634, and the update the level settles at:")
    for cond in conds:
        cond_rows = [r for r in rows if r["cond"] == cond and int(r["rep"]) in reps]
        settled, biases = [], []
        for r in cond_rows:
            upds, values = curve(r)
            level, _, _, bias = stats.level(values)
            biases.append(abs(bias) < 0.0634)
            settled.append(stats.plateau(upds, values, level, 0.0634))
        print(f"  {cond:<10}{sum(biases)}/{len(biases)}   settles at {settled}")

    print("\ncontrast                          value     SEM       p")
    for name, a, b in (("joint minus control", "joint", "control"),
                       ("reward minus control", "reward", "control"),
                       ("position minus joint", "position", "joint")):
        for label, subset in ((name, nets),
                              ("  the same, excluding net 1", [n for n in nets if n != 1])):
            mean, sem, p = stats.paired([table[a][n] for n in subset],
                                        [table[b][n] for n in subset])
            print(f"{label:<32}{mean:>+8.3f}{sem:>8.3f}{p:>8.4f}   "
                  f"{stats.verdict(mean, p, RES_BEHAVIOUR)}")


def lam(args):
    for task in ("context", "c1"):
        rows = runs.read(f"{args.results}/nf_{task}_*_lam*_net*.csv")
        rows = [r for r in rows if int(r["eval_n"]) == 10000 and published(r)]
        if not rows:
            continue
        print(f"\n{task}: level at the budget by lambda")
        control_rows = [r for r in rows if r["cond"] == "control"]
        base = levels(control_rows, range(10))
        for value in sorted({float(r["lam"]) for r in rows if r["cond"] == "joint"}):
            joint = levels([r for r in rows if r["cond"] == "joint" and float(r["lam"]) == value],
                           range(10))
            nets = sorted(set(joint) & set(base))
            if not nets:
                continue
            mean, sem, p = stats.paired([joint[n] for n in nets], [base[n] for n in nets])
            print(f"  lambda {value:<5g} L = {np.mean([joint[n] for n in nets]):.3f}   "
                  f"vs control {mean:+.4f} +/- {sem:.4f}, p = {p:.4f}")
        if base:
            print(f"  control     L = {np.mean(list(base.values())):.3f}")


def representation(args):
    prefix = "repr_all_context" if args.all_trials else "repr_context"
    rows = runs.read(f"{args.results}/{prefix}_*_net*.csv")
    print(f"pool: {'every trial' if args.all_trials else 'correct trials only'}")
    conds = ("theta0", "control", "position", "joint", "reward")
    print(f"{'condition':<12}{'crossnobis':>12}{'XOR':>10}{'match':>10}{'context':>10}")
    table = {}
    for cond in conds:
        cond_rows = [r for r in rows if r["cond"] == cond]
        if not cond_rows:
            continue
        table[cond] = {k: by_net(cond_rows, lambda r, k=k: float(r[k]))
                       for k in ("crossnobis", "xor_acc", "match_acc", "context_acc")}
        means = {k: np.mean([np.mean(v) for v in table[cond][k].values()]) for k in table[cond]}
        print(f"{cond:<12}{means['crossnobis']:>12.4f}{means['xor_acc']:>10.4f}"
              f"{means['match_acc']:>10.4f}{means['context_acc']:>10.4f}")

    if "theta0" not in table:
        return
    print("\nagainst theta_0                 value     SEM       p")
    for cond in conds[1:]:
        if cond not in table:
            continue
        for key in ("crossnobis", "xor_acc"):
            nets = sorted(set(table[cond][key]) & set(table["theta0"][key]))
            mean, sem, p = stats.paired([np.mean(table[cond][key][n]) for n in nets],
                                        [np.mean(table["theta0"][key][n]) for n in nets])
            print(f"{cond + ' ' + key:<28}{mean:>+8.4f}{sem:>8.4f}{p:>8.4f}")

    if "theta0" in table:
        behaviour_vs_code(args, table)


def behaviour_vs_code(args, table):
    """Does the behavioural level follow the decodability of the conjunction?

    The levels come from the runs that produced the checkpoints the representation pass read, so the
    two quantities are measured on the same weights.
    """
    rows = [r for r in runs.read(f"{args.results}/nf_context_*_lam1_*_net*.csv") if published(r)]
    reps = [int(s) for s in args.reps.split(",")]
    conds = [c for c in table if c != "theta0"]
    cells = []
    for cond in conds:
        L = levels([r for r in rows if r["cond"] == cond], reps)
        for net, value in L.items():
            if net in table[cond]["xor_acc"]:
                cells.append((value, float(np.mean(table[cond]["xor_acc"][net]))))
    if len(cells) < 3:
        return

    level_values = np.array([c[0] for c in cells])
    xor_values = np.array([c[1] for c in cells])
    from scipy.stats import pearsonr
    r, p = pearsonr(level_values, xor_values)
    print(f"\nlevel against XOR decoding over {len(cells)} cells: r = {r:.3f}, p = {p:.2g}")

    if "control" not in table:
        return
    base_level = np.mean([levels([r for r in rows if r["cond"] == "control"], reps)[n]
                          for n in sorted(table["control"]["xor_acc"])])
    base_xor = np.mean([np.mean(v) for v in table["control"]["xor_acc"].values()])
    print("condition   measured   predicted from XOR    difference")
    for cond in conds:
        if cond == "control":
            continue
        L = levels([r for r in rows if r["cond"] == cond], reps)
        measured = np.mean([L[n] for n in sorted(table[cond]["xor_acc"]) if n in L]) - base_level
        predicted = stats.probit(np.mean([np.mean(v) for v in table[cond]["xor_acc"].values()])) \
            - stats.probit(base_xor)
        print(f"{cond:<12}{measured:>+9.3f}{predicted:>+21.3f}{measured - predicted:>+14.3f}")


def gradients(args):
    rows = runs.read(f"{args.results}/grad_context_*_net*.csv")
    print(f"{'condition':<12}{'cos':>10}{'SEM':>8}{'rho':>10}{'|g_task|':>10}{'|g_aux|':>10}")
    for cond in sorted({r["cond"] for r in rows}):
        cond_rows = [r for r in rows if r["cond"] == cond]
        cos = [float(r["cos"]) for r in cond_rows]
        print(f"{cond:<12}{np.mean(cos):>+10.3f}{stats.sem(cos):>8.3f}"
              f"{np.mean([float(r['rho']) for r in cond_rows]):>10.2f}"
              f"{np.mean([float(r['g_task']) for r in cond_rows]):>10.2f}"
              f"{np.mean([float(r['g_aux']) for r in cond_rows]):>10.2f}")


def projection(args):
    """Predicted against realised change in the task loss for a step of fixed length along each
    gradient. The criterion is only meaningful where the two agree for both directions."""
    rows = runs.read(f"{args.results}/grad_step_context_*_net*.csv")
    lengths = sorted({r["_file"].split("_L")[1].split("_net")[0] for r in rows}, key=float)
    print(f"{'length':<10}{'direction':<10}{'predicted':>11}{'realised':>11}{'r':>8}{'n':>5}")
    for length in lengths:
        at = [r for r in rows if f"_L{length}_" in r["_file"]]
        for name in ("task", "aux"):
            pred = [float(r[f"d_task_{name}_pred"]) for r in at]
            real = [float(r[f"d_task_{name}_real"]) for r in at]
            corr = np.corrcoef(pred, real)[0, 1] if len(pred) > 2 else float("nan")
            print(f"{length:<10}{name:<10}{np.mean(pred):>+11.4f}{np.mean(real):>+11.4f}"
                  f"{corr:>8.3f}{len(pred):>5d}")

    shortest = lengths[0]
    print()
    print(f"at length {shortest}, by condition: does the step along each gradient lower the "
          f"task loss?")
    print(f"{'condition':<10}{'cos':>8}{'rho':>8}{'direction':>12}{'realised':>11}{'SEM':>9}"
          f"{'p':>8}{'lower':>8}")
    for cond in sorted({r["cond"] for r in rows}):
        at = [r for r in rows if r["cond"] == cond and f"_L{shortest}_" in r["_file"]]
        for name in ("task", "aux"):
            real = [float(r[f"d_task_{name}_real"]) for r in at]
            mean, sem, p = stats.one_sample(real)
            print(f"{cond:<10}{np.mean([float(r['cos']) for r in at]):>+8.3f}"
                  f"{np.mean([float(r['rho']) for r in at]):>8.2f}{name:>12}"
                  f"{mean:>+11.5f}{sem:>9.5f}{p:>8.3f}"
                  f"{sum(v < 0 for v in real):>5d}/{len(real):<3d}")


def movement(args):
    """How far each c1 signal moved the recurrent weights, against what it did to the code and to
    behaviour: the expansion arms with their shuffled null, and reward against control."""
    def arm(pattern):
        out = {}
        for r in runs.read(f"{args.results}/{pattern}"):
            values = curve(r)[1]
            out[int(r["net"])] = dict(
                behaviour=float(np.mean(values[-6:])) - values[0], moved=float(r["dW_rec"]),
                code=float(r["crossnobis"]) - float(r["crossnobis_start"]) if r["crossnobis"] else None)
        return out

    arms = {"shuffled, lr 1e-4": arm("nf_c1_expand_*_k1_shuf_lr0.0001_net*.csv"),
            "shuffled, lr 1.3e-4": arm("nf_c1_expand_*_k1_shuf_lr0.00013_net*.csv"),
            "kappa 1, lr 1e-4": arm("nf_c1_expand_*_k1_lr0.0001_net*.csv"),
            "kappa 2, lr 7.5e-5": arm("nf_c1_expand_*_k2_lr7.5e-05_net*.csv"),
            "kappa 2, lr 1e-4": arm("nf_c1_expand_*_k2_lr0.0001_net*.csv"),
            "control, lr 1e-4": arm("nf_c1_control_move_net*.csv"),
            "reward, lr 1e-4": arm("nf_c1_reward_move_net*.csv")}
    arms = {k: v for k, v in arms.items() if v}
    print(f"{'arm':<22}{'n':>3}{'moved':>9}{'code':>10}{'behaviour':>11}")
    for name, a in arms.items():
        codes = [v["code"] for v in a.values() if v["code"] is not None]
        print(f"{name:<22}{len(a):>3}{np.mean([v['moved'] for v in a.values()]):>9.4f}"
              + (f"{np.mean(codes):>+10.4f}" if codes else f"{'':>10}")
              + f"{np.mean([v['behaviour'] for v in a.values()]):>+11.3f}")

    def contrast(label, x, y, key, per_move=False):
        if x not in arms or y not in arms:
            return
        nets = sorted(set(arms[x]) & set(arms[y]))
        value = (lambda a, n: a[n][key] / a[n]["moved"]) if per_move else (lambda a, n: a[n][key])
        mean, sem, p = stats.paired([value(arms[x], n) for n in nets], [value(arms[y], n) for n in nets])
        print(f"  {label:<48}{mean:>+9.3f} +- {sem:.3f}  p = {p:.4f}")

    print("\nat matched movement:")
    contrast("kappa 1 - shuffled (1.3e-4), behaviour", "kappa 1, lr 1e-4", "shuffled, lr 1.3e-4", "behaviour")
    contrast("kappa 2 (7.5e-5) - shuffled (1.3e-4), behaviour", "kappa 2, lr 7.5e-5",
             "shuffled, lr 1.3e-4", "behaviour")
    print("reward against control:")
    contrast("behaviour gained", "reward, lr 1e-4", "control, lr 1e-4", "behaviour")
    contrast("movement", "reward, lr 1e-4", "control, lr 1e-4", "moved")
    contrast("behaviour per unit of movement", "reward, lr 1e-4", "control, lr 1e-4", "behaviour",
             per_move=True)


def retention(args):
    """Reward against control on c1 at the same rate: how far each moved the weights and the
    policy, and what happened to the tasks that share the read-out."""
    rows = runs.read(f"{args.results}/nf_c1_*_ret_*net*.csv")
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        values = curve(r)[1]
        moved, kl = float(r["dW_rec"]), float(r["kl_task"])
        for key, value in (("level", float(np.mean(values[-6:]))), ("moved", moved), ("kl", kl),
                           ("kl per movement", kl / moved),
                           ("pretraining task", float(r["pretrain_end"]) - float(r["pretrain_start"])),
                           ("patch cell", float(r["patch_end"]) - float(r["patch_start"]))):
            per[(r["cond"], int(r["net"]))][key].append(value)
    if not any(c == "reward" for c, _ in per) or not any(c == "control" for c, _ in per):
        print("needs reward and control runs with --retention")
        return
    nets = sorted({n for c, n in per if c == "reward"} & {n for c, n in per if c == "control"})
    value = lambda cond, n, key: float(np.mean(per[(cond, n)][key]))
    print(f"{'quantity':<20}{'reward':>9}{'control':>9}{'difference':>12}{'SEM':>8}{'p':>8}")
    for key in ("level", "moved", "kl", "kl per movement", "pretraining task", "patch cell"):
        a = [value("reward", n, key) for n in nets]
        b = [value("control", n, key) for n in nets]
        mean, sem, p = stats.paired(a, b)
        print(f"{key:<20}{np.mean(a):>+9.4f}{np.mean(b):>+9.4f}{mean:>+12.4f}{sem:>8.4f}{p:>8.4f}")
    print("\nchange on the shared-read-out tasks against their starting level:")
    for cond in ("reward", "control"):
        for key in ("pretraining task", "patch cell"):
            mean, sem, p = stats.one_sample([value(cond, n, key) for n in nets])
            print(f"  {cond:<8}{key:<18}{mean:>+8.3f} +- {sem:.3f}  p = {p:.4f}")


def share(args):
    """What fraction of the measured headroom each c1 signal realises. The ceilings are trained
    with the read-out's bias, so they sit at their best threshold; the runs are read both at the
    frozen decoder's own threshold and at the best one, so numerator and denominator can be put
    on the same footing."""
    heads = runs.read(f"{args.results}/headroom_c1_*_net*.csv")
    arms = {a: by_net([r for r in heads if r["arm"] == a], lambda r: float(r["C_mean"]))
            for a in ("A", "B")}
    c_a = {n: st.median(v) for n, v in arms["A"].items()}
    c_b = {n: st.median(v) for n, v in arms["B"].items()}
    print(f"C_A {np.mean(list(c_a.values())):.3f}   C_B {np.mean(list(c_b.values())):.3f}")
    print(f"{'condition':<10}{'n':>3}{'own threshold':>15}{'best threshold':>16}{'best - C_B':>12}{'p':>8}")
    for cond, pattern in (("reward", "nf_c1_reward_lam1_n10000_crit_net*_r[01].csv"),
                          ("control", "nf_c1_control_lam1_n10000_crit_lr0.0001_net*_r[01].csv")):
        rows = runs.read(f"{args.results}/{pattern}")
        own = by_net(rows, lambda r: float(np.mean(curve(r)[1][-6:])))
        best = by_net(rows, lambda r: float(r["crit_d_end"]))
        nets = sorted(set(own) & set(c_a) & set(c_b))
        if not nets:
            continue
        frac = lambda d, n: (np.mean(d[n]) - c_a[n]) / (c_b[n] - c_a[n])
        mean, sem, p = stats.paired([np.mean(best[n]) for n in nets], [c_b[n] for n in nets])
        print(f"{cond:<10}{len(nets):>3}{np.mean([frac(own, n) for n in nets]):>+15.2f}"
              f"{np.mean([frac(best, n) for n in nets]):>+16.2f}{mean:>+12.3f}{p:>8.3f}")


def swept(args, task="c1", also=(), where=None):
    """Every run of a task that recorded the threshold sweep at both ends and changed nothing
    beyond what the threshold tables pool, plus the knobs in `also`, filtered by `where`."""
    allowed = SWEPT | set(also)
    return [r for r in runs.read(f"{args.results}/nf_{task}_*_net*.csv")
            if r.get("crit_d_end") and r.get("crit_d_start") and protocol(r) <= allowed
            and (where is None or where(r))]


def per_net(rows, value):
    out = defaultdict(list)
    for r in rows:
        out[int(r["net"])].append(value(r))
    return {n: float(np.mean(v)) for n, v in out.items()}


def margins(args):
    out = {}
    for task in ("c1", "context"):
        rows = runs.read(f"{args.results}/criterion_resolution_{task}_net*.csv")
        if rows:
            out[task] = dict(d=max(RES_BEHAVIOUR, stats.resolution([float(r["best_d"]) for r in rows])),
                             auc=stats.resolution([float(r["auc"]) for r in rows]))
    return out


def equivalence(args):
    """The registered equivalence tests, with margins from the sweep's own resolution."""
    m = margins(args)
    if "c1" not in m:
        print("needs block 26 (criterion_resolution) first")
        return
    print(f"margins: c1 best d' {m['c1']['d']:.3f}, c1 AUC {m['c1']['auc']:.4f}")
    rows = [r for r in swept(args) if int(r["net"]) < 5 and int(r["updates"]) == 1200
            and float(r["lr"]) == 1e-4
            and (r.get("rec_noise_scale") or "1") == "1" and float(r.get("fa_cost") or 1) == 1
            and r.get("optimizer", "adam") in ("adam", "")]
    reward = [r for r in rows if r["cond"] == "reward" and int(r["rep"]) in (0, 1, 2)]
    control = [r for r in rows if r["cond"] == "control" and int(r["rep"]) in (0, 1, 2)]
    for label, key, margin in (("E1 c1 reward vs control, best d'", "crit_d_end", m["c1"]["d"]),
                               ("E1 c1 reward vs control, AUC", "auc_end", m["c1"]["auc"])):
        a = per_net([r for r in reward if r.get(key)], lambda r: float(r[key]))
        b = per_net([r for r in control if r.get(key)], lambda r: float(r[key]))
        nets = sorted(set(a) & set(b))
        mean, se, p = stats.equivalence([a[n] for n in nets], [b[n] for n in nets], margin)
        print(f"  {label:<40}{mean:+.4f} +- {se:.4f}  margin {margin:.4f}  p_equiv {p:.4f}")
    noisy = [r for r in swept(args) if r.get("rec_noise_scale") == "2" and int(r["net"]) < 5]
    a = per_net([r for r in noisy if r["cond"] == "reward"], lambda r: float(r["crit_d_end"]))
    b = per_net([r for r in noisy if r["cond"] == "control"], lambda r: float(r["crit_d_end"]))
    nets = sorted(set(a) & set(b))
    if nets:
        mean, se, p = stats.equivalence([a[n] for n in nets], [b[n] for n in nets], m["c1"]["d"])
        print(f"  {'E2 noise x2 reward vs control, best d':<40}{mean:+.4f} +- {se:.4f}  p_equiv {p:.4f}")
    mag = [r for r in swept(args) if r["cond"] == "expand" and r.get("kappa") == "2"]
    end = per_net(mag, lambda r: float(r["crit_d_end"]))
    begin = per_net(mag, lambda r: float(r["crit_d_start"]))
    nets = sorted(end)
    if nets:
        mean, se, p = stats.equivalence([end[n] for n in nets], [begin[n] for n in nets], m["c1"]["d"])
        print(f"  {'E3 magnified target, best d end - start':<40}{mean:+.4f} +- {se:.4f}  p_equiv {p:.4f}")


def criterion(args):
    """Block 27: where each payoff and each penalty leaves the decoder's threshold."""
    rows = [r for r in swept(args) if int(r["rep"]) == 3 and r.get("roc_end")]
    print(f"{'condition':<18}{'n':>3}{'payoff-optimal':>16}{'in [-0.5, 0.5]':>16}{'d-optimal':>11}{'AUC':>8}")
    groups = [("reward", float(c), c) for c in ("1", "2", "4")] + \
             [("control_fa", 1.0, w) for w in ("0", "0.4", "1.6")]
    for cond, cost, dose in groups:
        key = "fa_cost" if cond == "reward" else "fa_penalty"
        sel = [r for r in rows if r["cond"] == cond and (r.get(key) or "1") == dose]
        if not sel:
            continue
        opt = per_net(sel, lambda r: behaviour.payoff_optimal(r["roc_end"], cost)[0])
        dopt = per_net(sel, lambda r: float(r["crit_offset_end"]))
        auc = per_net(sel, lambda r: float(r["auc_end"]))
        inside = sum(abs(v) <= 0.5 for v in opt.values())
        print(f"{cond + ' ' + key[:2] + ' ' + dose:<18}{len(opt):>3}{np.mean(list(opt.values())):>+16.2f}"
              f"{inside:>13d}/{len(opt):<2d}{np.mean(list(dopt.values())):>+11.2f}"
              f"{np.mean(list(auc.values())):>8.3f}")


def components(args):
    """Block 28 against what the full runs did."""
    rows = runs.read(f"{args.results}/components_c1_net*.csv")
    if not rows:
        print("needs block 28")
        return
    step = {s: (per_net([r for r in rows if r["signal"] == s], lambda r: float(r["criterion"])),
                per_net([r for r in rows if r["signal"] == s], lambda r: float(r["separation"])))
            for s in sorted({r["signal"] for r in rows})}
    full = [r for r in swept(args) if int(r["net"]) < 5]
    runs_of = {"control": lambda r: r["cond"] == "control" and float(r["lr"]) == 1e-4
               and int(r["rep"]) in (0, 1) and (r.get("rec_noise_scale") or "1") == "1",
               "control_fa": lambda r: r["cond"] == "control_fa" and int(r["rep"]) == 2,
               "reward": lambda r: r["cond"] == "reward" and int(r["rep"]) in (0, 1)
               and (r.get("rec_noise_scale") or "1") == "1",
               "joint": lambda r: r["cond"] == "joint",
               "position k1": lambda r: r["cond"] == "expand" and r.get("kappa") == "1"
               and r.get("shuffle") in ("0", ""),
               "position k2": lambda r: r["cond"] == "expand" and r.get("kappa") == "2",
               "shuffled": lambda r: r["cond"] == "expand" and r.get("shuffle") == "1"}
    print(f"{'signal':<14}{'step crit':>11}{'step sep':>10}{'run d-offset':>14}{'run best gain':>15}")
    xs, ys, zs, ws = [], [], [], []
    for name, pick in runs_of.items():
        sel = [r for r in full if pick(r)]
        if name not in step or not sel:
            continue
        shift = np.mean(list(per_net(sel, lambda r: float(r["crit_offset_end"])
                                     - float(r["crit_offset_start"])).values()))
        gain = np.mean(list(per_net(sel, lambda r: float(r["crit_d_end"])
                                    - float(r["crit_d_start"])).values()))
        c, sp = np.mean(list(step[name][0].values())), np.mean(list(step[name][1].values()))
        xs.append(c), ys.append(shift), zs.append(sp), ws.append(gain)
        print(f"{name:<14}{c:>+11.4f}{sp:>+10.4f}{shift:>+14.2f}{gain:>+15.3f}")
    from scipy.stats import spearmanr
    if len(xs) > 2:
        print(f"C1 step criterion vs run offset change: rho = {spearmanr(xs, ys)[0]:+.2f}")
        print(f"C2 step separation vs run best-threshold gain: rho = {spearmanr(zs, ws)[0]:+.2f}")


def axis(args):
    """Block 29: the axis target's cosine as a dose."""
    rows = [r for r in swept(args) if r["cond"] == "axis"]
    if not rows:
        print("needs block 29")
        return
    for label, pick in (("classifier target (block 29)", lambda r: not r.get("axis_increment")),
                        ("equal increment (block 32)", lambda r: bool(r.get("axis_increment")))):
        chosen = [r for r in rows if pick(r)]
        if chosen:
            print(label)
            axis_dose(chosen)


def axis_dose(rows):
    cells = [(int(r["net"]), float(r["cosine"]), float(r["crit_d_end"]) - float(r["crit_d_start"]),
              float(np.mean(curve(r)[1][-6:])) - curve(r)[1][0], float(r["axis_auc_start"]),
              float(r["axis_auc_end"]),
              float(r["axis_m_end"]) - float(r["axis_m_start"]) if r.get("axis_m_end") else np.nan)
             for r in rows]
    print(f"{'cosine':>7}{'n':>3}{'best gain':>11}{'own change':>12}{'sep start':>11}{'sep end':>9}"
          f"{'rise (SD)':>11}")
    for cos in sorted({c[1] for c in cells}):
        at = [c for c in cells if c[1] == cos]
        print(f"{cos:>+7.1f}{len(at):>3}" + "".join(f"{np.mean([c[i] for c in at]):>+11.3f}"
                                                  for i in (2, 3)) +
              f"{np.mean([c[4] for c in at]):>11.3f}{np.mean([c[5] for c in at]):>9.3f}"
              f"{np.nanmean([c[6] for c in at]):>+11.3f}")
    nets = sorted({c[0] for c in cells})
    x = np.array([c[1] for c in cells])
    y = np.array([c[2] - np.mean([d[2] for d in cells if d[0] == c[0]]) for c in cells])
    x = x - np.array([np.mean([d[1] for d in cells if d[0] == c[0]]) for c in cells])
    slope = float((x * y).sum() / (x * x).sum())
    resid = y - slope * x
    dof = len(cells) - len(nets) - 1
    se = float(np.sqrt((resid ** 2).sum() / dof / (x * x).sum()))
    from scipy import stats as sps
    p = 2 * (1 - sps.t.cdf(abs(slope / se), dof))
    print(f"D1 best-threshold gain on cosine, within network: slope {slope:+.3f} +- {se:.3f}, p = {p:.4f}")


def batch2(args):
    """Block 30: the two registered primaries on networks 10-24."""
    nets = set(range(10, 25))
    for label, task, key, cond_a, cond_b in (("B1 context AUC, reward - control", "context", "auc_end",
                                              "reward", "control"),
                                             ("B2 c1 best offset, reward - control", "c1",
                                              "crit_offset_end", "reward", "control")):
        rows = [r for r in swept(args, task) if int(r["net"]) in nets and int(r["rep"]) in (0, 1)]
        a = per_net([r for r in rows if r["cond"] == cond_a], lambda r: float(r[key]))
        b = per_net([r for r in rows if r["cond"] == cond_b], lambda r: float(r[key]))
        both = sorted(set(a) & set(b))
        if len(both) < 2:
            print(f"{label}: not enough networks yet ({len(both)})")
            continue
        mean, se, p = stats.paired([a[n] for n in both], [b[n] for n in both])
        print(f"{label:<38}{mean:+.4f} +- {se:.4f}  p = {p:.4f}  n = {len(both)}  "
              f"{'passes' if p < 0.025 else 'does not pass'} at 0.025")


def readout(args):
    """Block 31: the payoff-optimal threshold of each learner under the greedy read-out it is scored
    with and under the sampled one the reward signal trains with."""
    rows = [r for r in swept(args) if r.get("roc_end_sampled")]
    print(f"{'learner':<22}{'n':>3}{'greedy':>9}{'in +-0.5':>10}{'sampled':>9}{'in +-0.5':>10}{'<= -0.5':>9}")
    for cond, key, dose in (("reward", "fa_cost", "1"), ("reward", "fa_cost", "4"),
                            ("control_fa", "fa_penalty", "0")):
        sel = [r for r in rows if r["cond"] == cond and (r.get(key) or "1") == dose]
        if not sel:
            continue
        cost = float(dose) if cond == "reward" else 1.0
        greedy = per_net(sel, lambda r: behaviour.payoff_optimal(r["roc_end"], cost)[0])
        sampled = per_net(sel, lambda r: behaviour.payoff_optimal(r["roc_end_sampled"], cost)[0])
        n = len(greedy)
        print(f"{cond + ' ' + dose:<22}{n:>3}{np.mean(list(greedy.values())):>+9.2f}"
              f"{sum(abs(v) <= 0.5 for v in greedy.values()):>8d}/{n}"
              f"{np.mean(list(sampled.values())):>+9.2f}{sum(abs(v) <= 0.5 for v in sampled.values()):>8d}/{n}"
              f"{sum(v <= -0.5 for v in sampled.values()):>7d}/{n}")


ARMS = ("reward", "control", "control_fa", "joint", "k2", "shuffled")


def arm(run):
    return next(v for k, v in (("control_fa", "control_fa"), ("_shuf", "shuffled"), ("_k2_", "k2"),
                               ("joint", "joint"), ("reward", "reward"), ("control", "control"))
                if k in run)


def stepwise(args):
    """Blocks 33-34 on the saved c1 states: discriminability before saturation (K1), the
    payoff-optimal threshold with the noise left on (K2), the shape and place of the change (K5, K6),
    and what moves each run's best threshold."""
    rows = runs.read(f"{args.results}/stepwise_c1_net*.csv")
    sweeps = [r for r in rows if r["kind"] == "sweep"]
    print("K1  update  reward  control    diff  range                p   AUC diff")
    for upd in sorted({int(r["upd"]) for r in sweeps}):
        at = [r for r in sweeps if int(r["upd"]) == upd]
        d, auc = ({a: per_net([r for r in at if arm(r["run"]) == a], lambda r: float(r[key]))
                   for a in ("reward", "control")} for key in ("best_d", "auc"))
        nets = sorted(set(d["reward"]) & set(d["control"]))
        diff = [d["reward"][n] - d["control"][n] for n in nets]
        print(f"    {upd:>6}  {np.mean([d['reward'][n] for n in nets]):.3f}   "
              f"{np.mean([d['control'][n] for n in nets]):.3f}  {np.mean(diff):+.3f}  "
              f"[{min(diff):+.3f}, {max(diff):+.3f}]  {stats.one_sample(diff)[2]:.3f}  "
              f"{np.mean([auc['reward'][n] - auc['control'][n] for n in nets]):+.4f}")
    print("\nK2  payoff-optimal offset at the end, recurrent noise on")
    for a in ("reward", "control"):
        g, s_ = (per_net([r for r in rows if r["kind"] == k and arm(r["run"]) == a],
                         lambda r: float(r["payoff_offset"]))
                 for k in ("noise_greedy", "noise_sampled"))
        print(f"    {a:<9} greedy {np.mean(list(g.values())):+.2f}  sampled "
              f"{np.mean(list(s_.values())):+.2f}  sampled in +-0.5 "
              f"{sum(abs(v) <= 0.5 for v in s_.values())}/{len(s_)}")
    act = [r for r in rows if r["kind"] == "activity"]
    start = per_net([r for r in sweeps if int(r["upd"]) == 0], lambda r: float(r["offset"]))
    end = {r["run"]: float(r["offset"]) for r in sweeps if int(r["upd"]) == 1200}
    for r in act:
        r["d_offset"] = end[r["run"]] - start[int(r["net"])]
    print("\nK5, K6 and the threshold, network means\n    " + " " * 22 +
          "".join(f"{a:>11}" for a in ARMS))
    for c in ("dw_rank", "top_pc_share", "potent_over_random", "criterion_along_axis",
              "separation_along_axis", "norm_ratio", "logit_shift", "d_hold_level", "d_hold_sd",
              "d_resp_max", "passage_offset", "d_offset"):
        print(f"    {c:<22}" + "".join(
            f"{np.mean([float(r[c]) for r in act if arm(r['run']) == a]):>11.3f}" for a in ARMS))
    print(f"\n    best-threshold shift against its predictors, {len(act)} runs (slopes, R2):")
    for xs in (("logit_shift",), ("passage_offset",), ("d_hold_level", "d_hold_sd")):
        for label, sel in (("all", act), ("without reward", [r for r in act
                                                             if arm(r["run"]) != "reward"])):
            X = np.column_stack([np.ones(len(sel))] + [[float(r[c]) for r in sel] for c in xs])
            t = np.array([r["d_offset"] for r in sel])
            b = np.linalg.lstsq(X, t, rcond=None)[0]
            r2 = 1 - ((t - X @ b) ** 2).sum() / ((t - t.mean()) ** 2).sum()
            print(f"    {' + '.join(xs):<26}{label:<16}" + " ".join(f"{v:+.2f}" for v in b[1:]) +
                  f"   R2 {r2:.2f}")


def refit(args):
    """Block 35: reward against supervision through a decoder refitted on the frozen state (K4)."""
    rows = swept(args, also={"refit_decoder"}, where=lambda r: "refit_decoder" in protocol(r))
    print("measure              reward  control    diff   >0       p")
    for name, value in (("own-threshold d'", lambda r: curve(r)[1][-1]),
                        ("best-threshold d'", lambda r: float(r["crit_d_end"])),
                        ("AUC", lambda r: float(r["auc_end"])),
                        ("best offset", lambda r: float(r["crit_offset_end"])),
                        ("payoff-optimal", lambda r: behaviour.payoff_optimal(r["roc_end"])[0])):
        a = per_net([r for r in rows if r["cond"] == "reward"], value)
        b = per_net([r for r in rows if r["cond"] == "control"], value)
        diff = [a[n] - b[n] for n in sorted(set(a) & set(b))]
        print(f"{name:<20}{np.mean(list(a.values())):>7.3f}{np.mean(list(b.values())):>9.3f}"
              f"{np.mean(diff):>+8.3f}{sum(d > 0 for d in diff):>3d}/{len(diff)}"
              f"{stats.one_sample(diff)[2]:>8.4f}")


def entropy(args):
    """Block 36: the reward learner's operating point at entropy bonus 0 and 0.01 (K3); 0.05 is
    block 31."""
    print("bonus   n   own d'  best d'    AUC  greedy payoff  sampled payoff  sampled in +-1")
    for ent in (0.0, 0.01, 0.05):
        sel = [r for r in swept(args, also={"entropy"},
                                where=lambda r: float(r.get("entropy") or 0.05) == ent)
               if r["cond"] == "reward"
               and r.get("roc_end_sampled") and (r.get("fa_cost") or "1") == "1"]
        if not sel:
            continue
        g = per_net(sel, lambda r: behaviour.payoff_optimal(r["roc_end"])[0])
        s_ = per_net(sel, lambda r: behaviour.payoff_optimal(r["roc_end_sampled"])[0])
        mean = lambda f: np.mean(list(per_net(sel, f).values()))
        print(f"{ent:<6g}{len(g):>3}{mean(lambda r: curve(r)[1][-1]):>8.3f}"
              f"{mean(lambda r: float(r['crit_d_end'])):>9.3f}"
              f"{mean(lambda r: float(r['auc_end'])):>7.3f}"
              f"{np.mean(list(g.values())):>+15.2f}{np.mean(list(s_.values())):>+16.2f}"
              f"{sum(abs(v) <= 1 for v in s_.values()):>14d}/{len(s_)}")


def clamp_precheck(args):
    """How much of each signal's step on the context task moves the criterion, at the start and at
    the saved end points, and whether two independent halves of the gradient agree."""
    rows = runs.read(f"{args.results}/criterion_share_context_net*.csv")
    where = lambda r: "start" if r["state"] == "start" else r["state"].split("_net")[0] + " end"
    print("state          signal   criterion separation      share halves cos  sign agreement")
    for w in ("start", "control end", "reward end"):
        for sig in ("reward", "control"):
            sel = [r for r in rows if where(r) == w and r["signal"] == sig]
            agree = np.mean([h.split("/")[0][0] == h.split("/")[1][0]
                             for h in (r["halves_criterion"] for r in sel)])
            print(f"{w:<15}{sig:<9}" + "".join(
                f"{np.mean([float(r[c]) for r in sel]):>+11.3f}"
                for c in ("criterion", "separation", "criterion_share", "halves_cos")) +
                f"{agree:>12.1f}")


def calibration(args):
    """Block 39: the reward learners at false-alarm cost 1, 2 and 4, read with the noise on, under
    each cost's payoff (D1-D3), and the payoff lost at the learner's own threshold."""
    rows = runs.read(f"{args.results}/calibration_c1*.csv")

    def payoffs(r, cost):
        out = {}
        for point in r["roc"].split("|"):
            o, _, _, correct, fa, miss, abort = (float(v) for v in point.split(":"))
            out[o] = correct - cost * fa - miss - abort
        return out

    for label, nets in (("networks 0-4", range(0, 5)), ("networks 10-24", range(10, 25))):
        calibration_set([r for r in rows if int(r["net"]) in nets], label, payoffs)


def calibration_set(rows, label, payoffs):
    if not rows:
        return
    print(f"\n{label}")
    for readout in ("sampled", "greedy"):
        print(f"{readout} read-out  trained  optimum under c1   c2    c4   own in +-1  "
              f"FA at own   hit at own  payoff lost   best d'   AUC")
        for cost in (1.0, 2.0, 4.0):
            sel = [r for r in rows if r["readout"] == readout and float(r["trained_cost"]) == cost]
            own = [float(r[f"payoff_c{cost:g}"]) for r in sel]
            lost = [max(payoffs(r, cost).values()) - payoffs(r, cost)[0.0] for r in sel]
            print(f"{'':<19}{cost:>5g}   " + "".join(
                f"{np.mean([float(r[f'payoff_c{c:g}']) for r in sel]):>+8.2f}" for c in (1, 2, 4))
                + f"{sum(abs(v) <= 1 for v in own):>9d}/{len(own)}"
                + f"{np.mean([float(r['fa_at_zero']) for r in sel]):>11.3f}"
                + f"{np.mean([float(r['hit_at_zero']) for r in sel]):>12.3f}"
                + f"{np.mean(lost):>13.4f}{np.mean([float(r['best_d']) for r in sel]):>10.3f}"
                + f"{np.mean([float(r['auc']) for r in sel]):>7.3f}")


def passage(args):
    """Block 40: the first-passage account on networks 10-24 with the coefficients fixed on
    networks 0-4 (F1), against the mean logit (F2), and the route of each signal (F3)."""
    rows = runs.read(f"{args.results}/stepwise_c1_net*.csv")
    sweeps = [r for r in rows if r["kind"] == "sweep"]
    start = per_net([r for r in sweeps if int(r["upd"]) == 0], lambda r: float(r["offset"]))
    end = {r["run"]: float(r["offset"]) for r in sweeps if int(r["upd"]) == 1200}
    act = [r for r in rows if r["kind"] == "activity"]
    for r in act:
        r["d_offset"] = end[r["run"]] - start[int(r["net"])]
    fit = [r for r in act if int(r["net"]) < 5]
    new = [r for r in act if int(r["net"]) >= 10]
    x = np.array([float(r["passage_offset"]) for r in fit])
    slope, icept = np.polyfit(x, [r["d_offset"] for r in fit], 1)
    y = np.array([r["d_offset"] for r in new])
    pred = icept + slope * np.array([float(r["passage_offset"]) for r in new])
    r2 = lambda t, f: 1 - ((t - f) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    print(f"F1  fitted on nets 0-4: {icept:+.2f} + {slope:.2f} x;  out of sample on "
          f"{len(new)} runs: R2 {r2(y, pred):.2f}")
    for c in ("passage_offset", "logit_shift"):
        xs = np.array([float(r[c]) for r in new])
        b = np.polyfit(xs, y, 1)
        print(f"F2  refitted on nets 10-24, {c:<15} slope {b[0]:+.2f} intercept {b[1]:+.2f} "
              f"R2 {r2(y, np.polyval(b, xs)):.2f}")
    print("F3  networks 10-24, count of networks")
    for a, tests in (("reward", (("hold level falls", lambda r: float(r["d_hold_level"]) < 0),
                                 ("|d SD| < |d level|", lambda r: abs(float(r["d_hold_sd"]))
                                  < abs(float(r["d_hold_level"]))))),
                     ("k2", (("hold SD rises", lambda r: float(r["d_hold_sd"]) > 0),
                             ("best offset falls", lambda r: r["d_offset"] < 0))),
                     ("shuffled", (("norm ratio < 1", lambda r: float(r["norm_ratio"]) < 1),
                                   ("response peak falls", lambda r: float(r["d_resp_max"]) < 0),
                                   ("best offset rises", lambda r: r["d_offset"] > 0)))):
        sel = [r for r in new if arm(r["run"]) == a]
        print(f"    {a:<9}" + "; ".join(f"{name} {sum(t(r) for r in sel)}/{len(sel)}"
                                        for name, t in tests)
              + f"; all {sum(all(t(r) for _, t in tests) for r in sel)}/{len(sel)}")
    print("    network means, nets 0-4 | nets 10-24:")
    for a in ("reward", "control", "k2", "shuffled"):
        print(f"    {a:<9}" + " | ".join(
            " ".join(f"{c} {np.mean([float(r[c]) for r in grp if arm(r['run']) == a]):+.2f}"
                     for c in ("d_offset", "passage_offset", "d_hold_level", "d_hold_sd",
                               "d_resp_max", "norm_ratio"))
            for grp in (fit, new)))


TRACKED = SWEPT | {"save_at", "track", "track_rate", "yoke_from", "baseline", "batch",
                   "track_every", "track_step"}


def adaptive(args):
    """Blocks 37 and 38 on the context task: the reward learner with its threshold tracked (c = 0,
    a hit rate of 0.7, or yoked), at a fixed threshold, with the trial-type baseline and with four
    times the batch, against supervision; the tracker's manipulation checks."""
    rows = [r for r in runs.read(f"{args.results}/nf_context_*_net*_r9.csv")
            if protocol(r) <= TRACKED and r.get("crit_d_end")]

    def label(r):
        if r["cond"] == "control":
            return "control"
        return ((r.get("track") or "") + ("typebase" if r.get("baseline") == "type" else "")
                + (f"batch{r['batch']}" if int(r.get("batch") or 48) != 48 else "")
                + (f"_u{r['updates']}" if int(r["updates"]) != BUDGET["context"] else "")
                or "fixed")

    def late(r, key):
        v = runs.split(r.get(key) or "")
        return float(np.mean(np.abs(v[-len(v) // 4:]))) if v else float("nan")

    arms = ("fixed", "balance", "rate", "yoke", "typebase", "batch192", "batch192_u2000",
            "control")
    auc = {a: per_net([r for r in rows if label(r) == a], lambda r: float(r["auc_end"]))
           for a in arms}
    best = {a: per_net([r for r in rows if label(r) == a], lambda r: float(r["crit_d_end"]))
            for a in arms}
    print("arm        n    AUC  best d'  late |imbalance|  late aborts  update aborts > 0.5")
    for a in arms:
        sel = [r for r in rows if label(r) == a]
        if not sel:
            continue
        cross = [next((25 * (i + 1) for i, v in enumerate(runs.split(r["abort_traj"]))
                       if v > 0.5), None) for r in sel if r.get("abort_traj")]
        print(f"{a:<9}{len(sel):>3}{np.mean(list(auc[a].values())):>7.3f}"
              f"{np.mean(list(best[a].values())):>8.3f}"
              f"{np.mean([late(r, 'imbalance_traj') for r in sel]):>17.3f}"
              f"{np.mean([late(r, 'abort_traj') for r in sel]):>13.3f}   "
              + (",".join(str(c) for c in cross) if cross else ""))
    print("\ncontrast               AUC diff      >0      p   best d' diff")
    for a, b in (("balance", "fixed"), ("yoke", "fixed"), ("rate", "fixed"),
                 ("typebase", "fixed"), ("batch192", "fixed"), ("batch192_u2000", "fixed"),
                 ("batch192_u2000", "batch192"), ("fixed", "control"),
                 ("batch192", "control"), ("typebase", "control")):
        nets = sorted(set(auc[a]) & set(auc[b]))
        if not nets:
            continue
        d = [auc[a][n] - auc[b][n] for n in nets]
        e = [best[a][n] - best[b][n] for n in nets]
        print(f"{a + ' - ' + b:<20}{np.mean(d):>+10.4f}{sum(v > 0 for v in d):>6d}/{len(d)}"
              f"{stats.one_sample(d)[2]:>8.3f}{np.mean(e):>+12.3f}")


def adaptive_split(args):
    """Block 37e: the step split of the reward gradient on the states each block-37 arm kept."""
    rows = [r for r in runs.read(f"{args.results}/criterion_share_b37_net*.csv")
            if r["signal"] == "reward" and r["state"] != "start"]
    name = lambda s: ("balance" if "_balance" in s else "rate" if "_track" in s else
                      "yoke" if "_yoke" in s else "control" if "_control_" in s else "fixed")
    print("arm      update  criterion  separation  share  halves cos  margin go  margin no-go")
    for a in ("fixed", "balance", "rate", "yoke", "control"):
        for upd in (2000, 4000, 8000):
            sel = [r for r in rows if name(r["state"]) == a and r["state"].endswith(f"_u{upd}")]
            if sel:
                print(f"{a:<9}{upd:>6}" + "".join(
                    f"{np.mean([float(r[c]) for r in sel]):>+11.3f}"
                    for c in ("criterion", "separation", "criterion_share", "halves_cos",
                              "margin_signal", "margin_noise")))


def context_optimum(args):
    """Where the reward learner's payoff is best on the context task against where the tracker's
    c = 0 sits, on saved states read with the noise on and sampled releases."""
    rows = runs.read(f"{args.results}/context_optimum_net*.csv")
    kind = lambda r: r["state"].split(" (")[0]
    print("state          n   c=0 exists   optimum conservative   c=0 - optimum   payoff lost "
          "at own   at c=0   aborts at c=0")
    for k in ("start", "reward u2000", "reward u4000", "reward u8000", "reward end", "control end"):
        sel = [r for r in rows if kind(r) == k]
        if not sel:
            continue
        found = [r for r in sel if r["offset_c0"] not in ("", "nan")]
        gap = [float(r["offset_c0"]) - float(r["offset_payoff"]) for r in found]
        lost = lambda key: np.mean([float(r["payoff_best"]) - float(r[key]) for r in sel])
        print(f"{k:<14}{len(sel):>3}{len(found):>8}/{len(sel)}{sum(g > 0 for g in gap):>17}/"
              f"{len(found)}{np.mean(gap) if gap else float('nan'):>+16.2f}"
              f"{lost('payoff_at_own'):>18.3f}{lost('payoff_at_c0'):>9.3f}"
              f"{np.mean([float(r['abort_c0']) for r in found]) if found else float('nan'):>14.3f}")


def location(args):
    """Block 46: best-threshold d' on c1 with the distractors moved to the cue, to the opposite edge,
    or no cue, against the task, per state; and the share of discriminability location carries."""
    rows = runs.read(f"{args.results}/location_c1_*.csv")
    for state in ("theta0", "control"):
        at = {v: per_net([r for r in rows if r["state"] == state and r["variant"] == v],
                         lambda r: float(r["best_d"])) for v in ("task", "at_cue", "opposite", "no_cue")}
        nets = sorted(at["task"])
        if not nets:
            continue
        print(f"\n{state}: best-threshold d' on the task {np.mean([at['task'][n] for n in nets]):.3f} "
              f"({len(nets)} networks)")
        for v in ("at_cue", "opposite", "no_cue"):
            d = [at[v][n] - at["task"][n] for n in nets if n in at[v]]
            mean, sem, p = stats.one_sample(d)
            print(f"    {v:<9} {np.mean([at[v][n] for n in nets]):.3f}   minus task {mean:+.3f} "
                  f"+- {sem:.3f}  p = {p:.4f}  ({sum(x < 0 for x in d)}/{len(d)} lower)")
        share = [(at["task"][n] - at["at_cue"][n]) / at["task"][n] for n in nets]
        mean, sem, _ = stats.one_sample(share)
        print(f"    share of the task's best d' lost at_cue: {mean:.3f} "
              f"(95 % CI {mean - 2.09 * sem:.3f} to {mean + 2.09 * sem:.3f})")


def correlate(x, y):
    from scipy.stats import pearsonr
    r, p = pearsonr(x, y)
    return f"r = {r:+.2f} (p = {p:.3f})"


def partial(x, y, z):
    """Correlation of x and y with z regressed out of both."""
    x, y, z = (np.asarray(v, dtype=float) for v in (x, y, z))
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return correlate(rx, ry)


def induction(args):
    """Block 47: across networks, how well each representational target was induced against what
    behaviour did: at the decoder's own threshold, through its best threshold, and in between."""
    rows = runs.read(f"{args.results}/induction_c1_net*.csv")
    for label, sel in (("magnified (k 2)", [r for r in rows if not int(r["shuffle"])]),
                       ("shuffled", [r for r in rows if int(r["shuffle"])])):
        if len(sel) < 4:
            continue
        v = lambda k: [float(r[k]) for r in sel]
        print(f"\n{label}, {len(sel)} networks; induction {np.mean(v('induction')):.3f} "
              f"(SD {np.std(v('induction'), ddof=1):.3f})")
        for measure in ("induction", "d_gain", "d_crossnobis"):
            print(f"    {measure:<13} vs own-threshold d' {correlate(v(measure), v('d_own'))}; "
                  f"vs best offset {correlate(v(measure), v('d_offset'))}; "
                  f"vs best-threshold d' {correlate(v(measure), v('d_best'))}; "
                  f"vs AUC {correlate(v(measure), v('d_auc'))}")
        print(f"    induction vs own-threshold d', best offset held fixed: "
              f"{partial(v('induction'), v('d_own'), v('d_offset'))}")


def dose(args):
    """Block 45e: the magnified target at kappa 0.5, 1, 1.5 and 2 on networks 10-24."""
    rows = [r for r in runs.read(f"{args.results}/induction_dose_c1_net*.csv")
            if not int(r["shuffle"])]
    kappas = sorted({float(r["kappa"]) for r in rows})
    cols = ("induction", "d_gain", "d_offset", "d_best", "d_own", "d_auc", "d_hold_level",
            "d_hold_sd", "d_resp_max", "norm_ratio")
    print("kappa  n " + "".join(f"{c:>13}" for c in cols))
    for k in kappas:
        sel = [r for r in rows if float(r["kappa"]) == k]
        print(f"{k:<5}{len(sel):>3} " + "".join(f"{np.mean([float(r[c]) for r in sel]):>+13.3f}"
                                            for c in cols))
    nets = sorted({int(r["net"]) for r in rows})

    def slopes(y, x="kappa"):
        out = []
        for n in nets:
            sel = [r for r in rows if int(r["net"]) == n]
            if len(sel) >= 3:
                out.append(np.polyfit([float(r[x]) for r in sel], [float(r[y]) for r in sel], 1)[0])
        return out

    for y, sign in (("d_offset", -1), ("d_hold_sd", 1), ("norm_ratio", 1)):
        sl = slopes(y)
        print(f"slope of {y} on kappa: mean {np.mean(sl):+.3f}, "
              f"{sum(np.sign(v) == sign for v in sl)}/{len(sl)} in the predicted direction")
    half = [float(r["d_offset"]) for r in rows if float(r["kappa"]) == 0.5]
    if half:
        print(f"kappa 0.5 best-offset shift: mean {np.mean(half):+.3f}, "
              f"{sum(v > 0 for v in half)}/{len(half)} conservative")
    for y in ("d_offset", "d_best"):
        sl = slopes(y, x="d_gain")
        print(f"within networks, slope of {y} on the head-gain change: mean {np.mean(sl):+.3f}, "
              f"{sum(v < 0 for v in sl)}/{len(sl)} negative")


def rt(args):
    """Block 48: each start network's criterion line against every trained state read at its own
    threshold: the change a pure criterion shift would have produced at the same c, and the rest."""
    rows = runs.read(f"{args.results}/rt_signatures_c1_*.csv")
    lines = defaultdict(list)
    for r in rows:
        if r["kind"] == "line":
            lines[int(r["net"])].append(r)
    rising, residual = [], defaultdict(lambda: defaultdict(list))
    for net, line in lines.items():
        line = sorted(line, key=lambda r: float(r["offset"]))
        peak = int(np.argmax([float(r["hit"]) for r in line]))
        branch = [r for r in line[:peak + 1] if r["rt"] not in ("", "nan")]
        c = np.array([float(r["c"]) for r in branch])
        order = np.argsort(c)
        from scipy.stats import spearmanr
        rising.append(spearmanr(c, [float(r["rt"]) for r in branch])[0] > 0)
        base = next(r for r in line if float(r["offset"]) == 0.0)
        for r in rows:
            if r["kind"] != "point" or int(r["net"]) != net or r["rt"] in ("", "nan"):
                continue
            cp = float(r["c"])
            if not c.min() <= cp <= c.max():
                continue
            # the kappa-dose runs (block 45) share the folder; they are labelled by their kappa
            k = next((t for t in r["run"].split("_") if t.startswith("k") and t[1:2].isdigit()), "")
            a = arm(r["run"] + "_") if k in ("", "k2", "k1") and "n2000" not in r["run"]                 else f"expand {k}"
            for key in ("rt", "rt_sd", "d"):
                pred = np.interp(cp, c[order], [float(branch[i][key]) for i in order])
                residual[a][key].append(float(r[key]) - pred)
            residual[a]["dc"].append(cp - float(base["c"]))
    print(f"R1: along the criterion line hit RT rises with c on {sum(rising)}/{len(rising)} networks")
    print("arm          n    change of c   RT beyond the line (ms)   RT SD beyond (ms)   d' beyond")
    for a in ("reward", "control", "control_fa", "joint", "k2", "shuffled", "expand k0.5",
              "expand k1", "expand k1.5"):
        res = residual.get(a)
        if not res:
            continue
        cell = lambda k: "{:+.2f} +- {:.2f} (p {:.3f})".format(*stats.one_sample(res[k]))
        print(f"{a:<11}{len(res['rt']):>3}   {np.mean(res['dc']):+.3f}        {cell('rt'):<26}"
              f"{cell('rt_sd'):<24}{cell('d')}")


def repertoire(args):
    """Block 49: re-association or new patterns on the context task, by state."""
    rows = runs.read(f"{args.results}/repertoire_context_net*.csv")
    print("state             n  outside subspace  beyond repertoire  potent x random    AUC")
    for state in sorted({r["state"] for r in rows}, key=lambda s: (s.split(" u")[0], len(s), s)):
        sel = [r for r in rows if r["state"] == state]
        auc = [float(r["auc"]) for r in sel if r.get("auc")]
        print(f"{state:<17}{len(sel):>2}{np.mean([float(r['outside_share']) for r in sel]):>13.3f}"
              f"{np.mean([float(r['new_patterns']) for r in sel]):>19.3f}"
              f"{np.mean([float(r['potent_over_random']) for r in sel]):>16.1f}"
              f"{np.mean(auc) if auc else float('nan'):>11.3f}")
    nets = sorted({int(r["net"]) for r in rows})
    for upd in (2000, 4000, 8000):
        pot = {a: per_net([r for r in rows if r["state"] == f"{a} u{upd}"],
                          lambda r: float(r["potent_over_random"])) for a in ("reward", "control")}
        both = sorted(set(pot["reward"]) & set(pot["control"]))
        if both:
            print(f"P1 update {upd}: reward more potent than supervision on "
                  f"{sum(pot['reward'][n] > pot['control'][n] for n in both)}/{len(both)}")
    diffs = []
    for n in nets:
        rew = next((r for r in rows if int(r["net"]) == n and r["state"] == "reward b192 u8000"), None)
        sup = [r for r in rows if int(r["net"]) == n and r["state"].startswith("control")
               and r.get("auc")]
        if rew and rew.get("auc") and sup:
            match = min(sup, key=lambda r: abs(float(r["auc"]) - float(rew["auc"])))
            diffs.append((float(match["outside_share"]) - float(rew["outside_share"]),
                          float(match["auc"]) - float(rew["auc"]), match["state"]))
    if diffs:
        mean, sem, p = stats.one_sample([d for d, _, _ in diffs])
        print(f"P2 at matched AUC (AUC gap {np.mean([g for _, g, _ in diffs]):+.3f}): supervision "
              f"minus reward in outside share {mean:+.4f} +- {sem:.4f}, p = {p:.3f}, "
              f"{sum(d > 0 for d, _, _ in diffs)}/{len(diffs)} larger; matched states "
              f"{[m for _, _, m in diffs]}")


def psychometric(args):
    """Block 50: the c1 comparison at five target strengths, without retraining."""
    every = runs.read(f"{args.results}/psychometric_c1_net*.csv")
    for label, nets in (("networks 0-4", range(0, 5)), ("networks 10-24", range(10, 25))):
        psychometric_set([r for r in every if int(r["net"]) in nets], label)


def psychometric_set(rows, label):
    if not rows:
        return
    print(f"\n{label}")
    print("strength   theta0 best   reward - control: best d'       own d'          AUC")
    for strength in sorted({float(r["strength"]) for r in rows}):
        at = lambda state, key: per_net([r for r in rows if r["state"] == state
                                         and float(r["strength"]) == strength],
                                        lambda r: float(r[key]))
        cells = []
        for key in ("best_d", "own_d", "auc"):
            a, b = at("reward", key), at("control", key)
            d = [a[n] - b[n] for n in sorted(set(a) & set(b))]
            m, se, p = stats.one_sample(d)
            cells.append(f"{m:+.3f} (p {p:.2f})")
        start = list(at("theta0", "best_d").values())
        print(f"{strength:<9}{np.mean(start) if start else float('nan'):>10.3f}     "
              + "   ".join(f"{c:<16}" for c in cells))


def focus(args):
    """Block 51e: the magnified target applied in one sector, each network read inside the sector
    and opposite it; the change inside minus the change opposite is the within-network effect."""
    rows = runs.read(f"{args.results}/focus_c1_*.csv")
    nets = sorted({int(r["net"]) for r in rows})
    at = {(int(r["net"]), int(r["upd"])): r for r in rows}
    nets = [n for n in nets if (n, 0) in at and (n, 1200) in at]
    print(f"{len(nets)} networks; change from the start to the end")
    print("measure          inside   opposite   inside - opposite        p    inside larger")
    for key, label in (("gain", "head gain"), ("best", "best-threshold d'"), ("auc", "AUC"),
                       ("own", "own-threshold d'"), ("offset", "best offset")):
        change = {side: [float(at[(n, 1200)][f"{side}_{key}"]) - float(at[(n, 0)][f"{side}_{key}"])
                         for n in nets] for side in ("inside", "opposite")}
        diff = [a - b for a, b in zip(change["inside"], change["opposite"])]
        mean, sem, pv = stats.one_sample(diff)
        print(f"{label:<17}{np.mean(change['inside']):>+7.3f}{np.mean(change['opposite']):>+10.3f}"
              f"{mean:>+12.3f} +- {sem:.3f}{pv:>10.4f}{sum(d > 0 for d in diff):>10d}/{len(diff)}")


def closed_loop(args):
    """Block 52: supervised training under closed-loop difficulty against the yoked control."""
    rows = runs.read(f"{args.results}/closed_loop_*_net*_r0.csv")
    arm_rows = {k: {int(r["net"]): r for r in rows if r["kind"] == k} for k in ("closed", "yoked")}
    nets = sorted(set(arm_rows["closed"]) & set(arm_rows["yoked"]))
    print(f"{len(nets)} networks with both arms")
    measures = (("decoding error, change", lambda r: float(r["err_end"]) - float(r["err_start"])),
                ("late lapse rate", lambda r: float(r["lapse_late"])),
                ("late target strength", lambda r: float(r["strength_late"])),
                ("hit rate, end", lambda r: float(r["hit_end"])),
                ("false-alarm rate, end", lambda r: float(r["fa_end"])),
                ("own-threshold d', change", lambda r: float(r["own_d_end"]) - float(r["own_d_start"])),
                ("best-threshold d', change", lambda r: float(r["best_d_end"]) - float(r["best_d_start"])),
                ("AUC, change", lambda r: float(r["auc_end"]) - float(r["auc_start"])),
                ("best offset, change", lambda r: float(r["offset_end"]) - float(r["offset_start"])))
    print("measure                      closed     yoked   closed - yoked        p   closed lower")
    for label, value in measures:
        c = [value(arm_rows["closed"][n]) for n in nets]
        y = [value(arm_rows["yoked"][n]) for n in nets]
        diff = [a - b for a, b in zip(c, y)]
        mean, sem, pv = stats.one_sample(diff)
        print(f"{label:<27}{np.mean(c):>+8.3f}{np.mean(y):>+10.3f}{mean:>+10.3f} +- {sem:.3f}"
              f"{pv:>9.4f}{sum(d < 0 for d in diff):>8d}/{len(diff)}")


def relocated(args):
    """Block 54: after closed-loop or yoked training, what the frozen head and a refitted decoder
    read, how separable the delay code is, and how much behaviour still relies on location."""
    rows = runs.read(f"{args.results}/relocated_c1_net*.csv")
    at = {s: {int(r["net"]): r for r in rows if r["state"] == s} for s in ("theta0", "closed", "yoked")}
    nets = sorted(set(at["theta0"]) & set(at["closed"]) & set(at["yoked"]))
    print(f"{len(nets)} networks\nmeasure                    theta0    closed     yoked   yoked - closed        p")
    for key, label in (("frozen_acc", "frozen head, accuracy"), ("refit_acc", "refitted decoder"),
                       ("crossnobis", "crossnobis"), ("inside_frozen", "refit inside frozen")):
        v = {s: [float(at[s][n][key]) for n in nets] for s in at}
        mean, sem, pv = stats.paired(v["yoked"], v["closed"])
        print(f"{label:<25}{np.mean(v['theta0']):>8.3f}{np.mean(v['closed']):>10.3f}"
              f"{np.mean(v['yoked']):>10.3f}{mean:>+11.3f} +- {sem:.3f}{pv:>9.4f}")
    for key in ("refit_acc", "crossnobis"):
        mean, sem, pv = stats.paired([float(at["yoked"][n][key]) for n in nets],
                                     [float(at["theta0"][n][key]) for n in nets])
        print(f"yoked - theta0, {key}: {mean:+.4f} +- {sem:.4f}  p = {pv:.4f}")
    drop = {s: [float(at[s][n]["best_at_cue"]) - float(at[s][n]["best_task"]) for n in nets]
            for s in ("closed", "yoked")}
    for s_ in drop:
        mean, sem, pv = stats.one_sample(drop[s_])
        print(f"{s_}: best d' with distractors at the cue minus task {mean:+.3f} +- {sem:.3f} "
              f"({sum(d < 0 for d in drop[s_])}/{len(nets)} lower)")
    mean, sem, pv = stats.paired(drop["closed"], drop["yoked"])
    print(f"closed - yoked in that drop: {mean:+.3f} +- {sem:.3f}  p = {pv:.3f}")


def content_or_drift(args):
    """Post hoc: the magnified target against the shuffled one on the same networks, raw and per
    unit of weight movement."""
    rows = runs.read(f"{args.results}/induction_c1_net*.csv")
    moved = {r["_file"][:-4]: float(r["dW_rec"])
             for r in runs.read(f"{args.results}/nf_c1_expand_lam1_n10000_k*_net*_r6.csv")}
    print("networks    k2 best d'   shuffled   k2 - shuffled        p   per unit movement k2 / shuffled")
    for label, nets in (("10-24", range(10, 25)), ("0-4", range(0, 5)), ("all", range(0, 25))):
        k2 = {int(r["net"]): r for r in rows if not int(r["shuffle"]) and int(r["net"]) in nets}
        sh = {int(r["net"]): r for r in rows if int(r["shuffle"]) and int(r["net"]) in nets}
        both = sorted(set(k2) & set(sh))
        a = [float(k2[n]["d_best"]) for n in both]
        b = [float(sh[n]["d_best"]) for n in both]
        mean, sem, pv = stats.paired(a, b)
        per = [np.mean([float(x[n]["d_best"]) / moved[x[n]["run"]] for n in both]) for x in (k2, sh)]
        print(f"{label:<10}{np.mean(a):>+10.3f}{np.mean(b):>+11.3f}{mean:>+10.3f} +- {sem:.3f}"
              f"{pv:>9.4f}{per[0]:>12.1f} / {per[1]:.1f}")


TABLES = {"headroom": headroom, "control": control, "resolutions": resolutions,
          "degradation": degradation, "patch": patch, "competence": competence,
          "nf-c1": nf_c1, "nf-context": nf_context, "lambda": lam, "labelfree": labelfree,
          "manifold": manifold,
          "representation": representation, "gradients": gradients,
          "projection": projection, "movement": movement, "retention": retention,
          "share": share, "equivalence": equivalence, "criterion": criterion,
          "components": components, "axis": axis, "batch2": batch2, "readout": readout,
          "stepwise": stepwise, "refit": refit, "entropy": entropy, "clamp-precheck": clamp_precheck,
          "calibration": calibration, "passage": passage, "adaptive": adaptive,
          "adaptive-split": adaptive_split, "context-optimum": context_optimum,
          "location": location, "induction": induction, "dose": dose, "rt": rt,
          "repertoire": repertoire, "psychometric": psychometric, "focus": focus,
          "closed-loop": closed_loop, "relocated": relocated, "content-or-drift": content_or_drift}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("table", choices=list(TABLES))
    p.add_argument("--results", default="results")
    p.add_argument("--reps", default="0,1")
    p.add_argument("--eval-n", type=int, default=1000, help="which evaluation size nf-c1 reads")
    p.add_argument("--all-trials", action="store_true",
                   help="representation: read the pass that pooled every trial")
    args = p.parse_args()
    TABLES[args.table](args)


if __name__ == "__main__":
    main()
