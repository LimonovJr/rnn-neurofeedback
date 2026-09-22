"""Turn the per-run CSVs into the tables the report quotes. Aggregation is deliberately separate
from the runs, so every table can be rebuilt without retraining anything.

    python -m experiments.aggregate headroom
    python -m experiments.aggregate nf-context --reps 0,1
"""

import argparse
import statistics as st
from collections import defaultdict

import numpy as np

from src import runs, stats
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


def published(r, target="target"):
    """The registered protocol: the stimulus target, no anchor, no decoder co-adaptation,
    back-propagation, the condition's own learning rate, the full budget. Later runs share the file pattern and vary one of these;
    the repeats run for other questions (weight movement, retention) are left out too, so the
    published tables stay what they were."""
    return (int(r["updates"]) == BUDGET[r["task"]] and float(r["lr"]) == LR[r["cond"]]
            and r.get("target") in (target, "", None)
            and float(r.get("anchor") or 0) == 0 and not int(r.get("adapt_decoder") or 0)
            and r.get("rule") in ("bptt", "", None) and "_move_" not in r["_file"]
            and not r.get("kl_task"))


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


TABLES = {"headroom": headroom, "control": control, "resolutions": resolutions,
          "degradation": degradation, "patch": patch, "competence": competence,
          "nf-c1": nf_c1, "nf-context": nf_context, "lambda": lam, "labelfree": labelfree,
          "manifold": manifold,
          "representation": representation, "gradients": gradients,
          "projection": projection, "movement": movement, "retention": retention}


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
