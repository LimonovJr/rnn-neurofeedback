"""The figures of the paper, drawn from the same per-run CSVs as the tables in `aggregate`, so a
figure and the table it illustrates cannot disagree. Six main figures and one supplementary.

    python -m experiments.figures                  # every figure into results/figures
    python -m experiments.figures calibration      # one of them

Each figure is written as PDF and PNG. A figure whose runs are not there yet is skipped with a
note rather than drawn from part of its data.
"""

import argparse
import os
import statistics as st
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.aggregate import (BUDGET, SETTINGS, TRACKED, arm, by_net, per_net,  # noqa: E402
                                   protocol, swept)
from src import runs  # noqa: E402

COLOURS = {"reward": "#d95f02", "control": "#1f78b4", "k2": "#1b9e77", "shuffled": "#7570b3",
           "control_fa": "#a6cee3", "joint": "#999999"}
NAMES = {"reward": "reward", "control": "supervision", "k2": "magnified target (κ 2)",
         "shuffled": "shuffled target"}
RES_BEHAVIOUR = 0.068

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "axes.titleweight": "bold", "legend.frameon": False,
                     "figure.dpi": 100})


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[saved] {os.path.join(out, name)}.pdf/.png", flush=True)


def roc_points(roc):
    """(offset, hit, false alarm, payoff parts) for every point of a recorded sweep."""
    out = []
    for point in roc.split("|"):
        o, hit, fa, correct, false_alarm, miss, abort = (float(v) for v in point.split(":"))
        out.append((o, hit, fa, correct, false_alarm, miss, abort))
    return out


def dots(ax, x, values, colour, jitter=0.08, mean_dx=0.22, label=None, hollow=False):
    """One dot per network, and the mean with its standard error beside them. `hollow` draws open
    squares, for the discovery networks next to the replication ones."""
    values = np.asarray(values, dtype=float)
    xs = x + np.random.uniform(-jitter, jitter, len(values))
    if hollow:
        ax.scatter(xs, values, s=11, marker="s", facecolors="none", edgecolors=colour, lw=0.7,
                   alpha=0.8, zorder=3, label=label)
    else:
        ax.scatter(xs, values, s=10, color=colour, alpha=0.6, lw=0, zorder=3, label=label)
    sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    ax.errorbar(x + mean_dx, values.mean(), yerr=sem, fmt="s" if hollow else "o", color="k",
                mfc="white" if hollow else "k", ms=3.5, capsize=2, zorder=4)


def headroom(args):
    """What better dynamics could still buy through the frozen decoder: H = C_B - C_A per network,
    on the cell tasks and on the context task."""
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    labels = {"c1": "c1,\nintact", "c1r1": "c1, mild\ninput noise",
              "c1r3": "c1, stronger\ninput noise", "patch": "patch cell",
              "context": "context\n(positive control)"}
    for i, setting in enumerate(labels):
        rows = runs.read(f"{args.results}/headroom_{setting}_*_net*.csv")
        c = {a: by_net([r for r in rows if r["arm"] == a], lambda r: float(r["C_mean"]))
             for a in ("A", "B")}
        nets = sorted(set(c["A"]) & set(c["B"]))
        dots(ax, i, [st.median(c["B"][n]) - st.median(c["A"][n]) for n in nets],
             COLOURS["reward"] if setting == "context" else "#555555")
    labels = list(labels.values())
    ax.axhline(0, color="k", lw=0.5)
    ax.axhspan(-RES_BEHAVIOUR, RES_BEHAVIOUR, color="0.9", zorder=0, lw=0)
    ax.set_xticks(range(len(labels)), labels, fontsize=6.5)
    ax.set_ylabel("headroom (d′):\nall weights − read-out only")
    ax.set_title("Decoder-limited tasks, dynamics-limited context")
    save(fig, args.out, "fig1_headroom")


def hull(points):
    """Upper convex hull of (false alarm, hit) points with the two corners, as the AUC is taken:
    at the liberal end releases turn into aborts and the raw points fold back."""
    out = []
    for pt in sorted([(0.0, 0.0), (1.0, 1.0)] + points):
        while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (pt[1] - out[-2][1])
                                 - (out[-1][1] - out[-2][1]) * (pt[0] - out[-2][0])) >= 0:
            out.pop()
        out.append(pt)
    return np.array(out)


def mean_roc(rows, grid=np.linspace(0, 1, 101), key="roc_end"):
    """Two views of the same sweeps, averaged over networks. The swept curve is the (false alarm,
    hit) point at each offset in offset order, so it passes through the point at the decoder's own
    threshold; at the liberal end releases turn into aborts and it folds back. The hull is the
    frontier the AUC is taken under, averaged vertically over false-alarm rates."""
    curves, trace = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for r in rows:
        pts = roc_points(r[key])
        h = hull([(fa, hit) for _, hit, fa, *_ in pts])
        curves[int(r["net"])].append(np.interp(grid, h[:, 0], h[:, 1]))
        for o, hit, fa, *_ in pts:
            trace[o][int(r["net"])].append((fa, hit))
    offsets = sorted(trace)
    swept_curve = np.array([np.mean([np.mean(v, 0) for v in trace[o].values()], 0) for o in offsets])
    frontier = np.mean([np.mean(v, 0) for v in curves.values()], 0)
    return grid, frontier, swept_curve, swept_curve[offsets.index(0.0)]


def strength_panel(ax, args):
    """Reward minus supervision against target strength: equal at the best threshold, reversing at
    the decoder's own."""
    rows = [r for r in runs.read(f"{args.results}/psychometric_c1_net*.csv")
            if r["state"] in ("reward", "control")]
    strengths = sorted({float(r["strength"]) for r in rows})
    for (lo, hi), marker, fill, label in (((10, 25), "o", True, "networks 10–24"),
                                          ((0, 5), "s", False, "networks 0–4")):
        for key, colour, name in (("own_d", COLOURS["reward"], "decoder's own threshold"),
                                  ("best_d", "0.25", "best threshold")):
            mean, sem = [], []
            for strength in strengths:
                v = {a: per_net([r for r in rows if r["state"] == a and lo <= int(r["net"]) < hi
                                 and float(r["strength"]) == strength], lambda r: float(r[key]))
                     for a in ("reward", "control")}
                d = [v["reward"][n] - v["control"][n] for n in sorted(set(v["reward"]) & set(v["control"]))]
                mean.append(np.mean(d))
                sem.append(np.std(d, ddof=1) / np.sqrt(len(d)))
            ax.errorbar(strengths, mean, yerr=sem, fmt=marker + ("-" if fill else ":"), color=colour,
                        ms=4, lw=1.1 if fill else 0.8, capsize=1.5,
                        mfc=colour if fill else "white", label=f"{name}, {label}")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0.9, color="0.8", lw=0.8, zorder=0)
    ax.set_xticks(strengths, [f"{v:g}" for v in strengths])
    ax.set_xlabel("target strength (trained at 0.9)")
    ax.set_ylabel("reward − supervision (d′)")
    ax.set_title("The winner depends on the test")
    ax.legend(fontsize=5.5, loc="upper left")


def one_roc(args):
    """Reward and supervision on the same ROC on c1, not on context; and across target strengths
    the same discriminability with an own-threshold difference that reverses."""
    nets = set(range(10, 25))
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    strength_panel(axes[2], args)
    # on context the reward learner's curve peaks beyond the standard grid, so that panel is drawn
    # from the end states swept to +12
    wide = runs.read(f"{args.results}/wide_sweep_context_b60_net*.csv")
    panels = (("c1: one ROC (networks 10–24)", "roc_end",
               [r for r in swept(args, "c1") if int(r["net"]) in nets and int(r["rep"]) in (0, 1)]),
              ("context: two ROCs (networks 10–24)", "roc", wide))
    for ax, (title, key, rows) in zip(axes, panels):
        if not rows:
            ax.set_title("no runs")
            continue
        for cond in ("control", "reward"):
            grid, frontier, curve, own = mean_roc([r for r in rows if r["cond"] == cond], key=key)
            ax.plot(grid, frontier, "--", color=COLOURS[cond], lw=0.8, alpha=0.75)
            ax.plot(curve[:, 0], curve[:, 1], "-", color=COLOURS[cond], lw=1.3, label=NAMES[cond])
            ax.plot(*own, "o", color=COLOURS[cond], ms=6, mec="k", mew=0.6, zorder=5)
        ax.plot([0, 1], [0, 1], ":", color="0.6", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("false-alarm rate")
        ax.set_ylabel("hit rate")
        ax.set_title(title, fontsize=8)
    axes[0].legend(loc="lower right", fontsize=7)
    axes[1].text(0.98, 0.25, "solid: swept curve\ndashed: upper hull\ncircle: own threshold",
                 ha="right", va="bottom", fontsize=6.5, transform=axes[1].transAxes)
    fig.tight_layout()
    save(fig, args.out, "fig2_one_roc")


def calibration(args):
    """Read the way it learns, the reward learner sits at the payoff optimum of its own
    false-alarm cost; read greedily with the noise off, it gives most of that payoff away."""
    rows = runs.read(f"{args.results}/calibration_c1*.csv")
    if not rows:
        print("calibration: no runs yet")
        return
    costs = sorted({float(r["trained_cost"]) for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.6))
    shades = dict(zip(costs, ("#fdae6b", "#e6550d", "#a63603")))
    new = lambda r: int(r["net"]) >= 10
    for cost in costs:
        for readout, style in (("sampled", "-"), ("greedy", "--")):
            sel = [r for r in rows if float(r["trained_cost"]) == cost and r["readout"] == readout
                   and new(r)]
            curves = defaultdict(list)
            for r in sel:
                pts = roc_points(r["roc"])
                best = max(c - cost * f - m - a for _, _, _, c, f, m, a in pts)
                for o, _, _, c, f, m, a in pts:
                    curves[o].append(c - cost * f - m - a - best)
            offsets = sorted(curves)
            axes[0].plot(offsets, [np.mean(curves[o]) for o in offsets], style,
                         color=shades[cost], lw=1.2,
                         label=f"cost {cost:g}" if readout == "sampled" else None)
    axes[0].axvline(0, color="k", lw=0.6)
    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-0.6, 0.02)
    axes[0].set_xlabel("threshold offset")
    axes[0].set_ylabel("payoff − best payoff")
    axes[0].set_title("Payoff around the own threshold")
    axes[0].legend(fontsize=6.5, loc="upper left")
    axes[0].text(0.98, 0.04, "networks 10–24\nsolid: sampled, noise on\ndashed: greedy, noise off",
                 ha="right", fontsize=6, transform=axes[0].transAxes)

    def lost(r, cost):
        pts = {o: c - cost * f - m - a for o, _, _, c, f, m, a in roc_points(r["roc"])}
        return max(pts.values()) - pts[0.0]

    # filled circles: networks 10-24; open squares: networks 0-4
    for i, cost in enumerate(costs):
        for readout, dx, colour in (("sampled", -0.24, shades[cost]), ("greedy", 0.14, "0.55")):
            for fresh, shift in ((True, 0.0), (False, 0.13)):
                sel = [r for r in rows if float(r["trained_cost"]) == cost
                       and r["readout"] == readout and new(r) == fresh]
                first = cost == costs[-1] and fresh
                dots(axes[1], i + dx + shift, [lost(r, cost) for r in sel], colour, jitter=0.03,
                     mean_dx=0.0, hollow=not fresh,
                     label=({"sampled": "sampled, noise on", "greedy": "greedy, noise off"}[readout]
                            if first else None))
    axes[1].set_xticks(range(len(costs)), [f"{c:g}" for c in costs])
    axes[1].set_xlabel("false-alarm cost trained with")
    axes[1].set_ylabel("payoff lost at own threshold")
    axes[1].set_title("Payoff given away")
    axes[1].legend(fontsize=6.5, loc="upper left")
    axes[1].text(0.98, 0.9, "● networks 10–24\n□ networks 0–4", ha="right", va="top", fontsize=6,
                 transform=axes[1].transAxes)

    for i, cost in enumerate(costs):
        for fresh, dx in ((True, -0.12), (False, 0.16)):
            sel = [r for r in rows if float(r["trained_cost"]) == cost and r["readout"] == "sampled"
                   and new(r) == fresh]
            dots(axes[2], i + dx, [float(r["fa_at_zero"]) for r in sel], shades[cost], jitter=0.04,
                 mean_dx=0.0, hollow=not fresh)
    axes[2].set_xticks(range(len(costs)), [f"{c:g}" for c in costs])
    axes[2].set_xlabel("false-alarm cost trained with")
    axes[2].set_ylabel("false alarms at own threshold")
    axes[2].set_title("The point follows the cost")
    fig.tight_layout()
    save(fig, args.out, "fig3_calibration")


def routes(axes, args):
    """Three signals move the best threshold three ways: reward by the level of the release margin
    during the hold, the magnified target by its spread, the shuffled target by quieting the
    population and shrinking the response peak; the mean logit does not say which."""
    rows = runs.read(f"{args.results}/stepwise_c1_net*.csv")
    sweeps = [r for r in rows if r["kind"] == "sweep"]
    start = per_net([r for r in sweeps if int(r["upd"]) == 0], lambda r: float(r["offset"]))
    end = {r["run"]: float(r["offset"]) for r in sweeps if int(r["upd"]) == 1200}
    act = [r for r in rows if r["kind"] == "activity" and r["run"] in end]
    for r in act:
        r["d_offset"] = end[r["run"]] - start[int(r["net"])]
    arms = ("reward", "control", "k2", "shuffled")
    measures = (("d_hold_level", "margin level, hold"), ("d_hold_sd", "margin spread, hold"),
                ("d_resp_max", "margin peak, response"), ("d_offset", "best threshold"))
    for ax, (key, label) in zip(axes, measures):
        for i, a in enumerate(arms):
            for net_set, marker, dx in (((0, 5), "o", -0.13), ((10, 25), "s", 0.13)):
                v = [float(r[key]) for r in act
                     if arm(r["run"]) == a and net_set[0] <= int(r["net"]) < net_set[1]]
                if v:
                    ax.errorbar(i + dx, np.mean(v), yerr=np.std(v, ddof=1) / np.sqrt(len(v)),
                                fmt=marker, color=COLOURS[a], ms=4, capsize=1.5)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(len(arms)), ["rew", "sup", "κ2", "shuf"])
        ax.set_title(f"Δ {label}")
    axes[0].set_ylabel("change (logit units)")
    ax = axes[4]
    for a in arms:
        sel = [r for r in act if arm(r["run"]) == a]
        ax.scatter([float(r["passage_offset"]) for r in sel], [r["d_offset"] for r in sel], s=9,
                   color=COLOURS[a], alpha=0.7, lw=0, label=NAMES[a])
    lim = [-9, 9]
    ax.plot(lim, lim, ":", color="0.5", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("first-passage prediction")
    ax.set_ylabel("best-threshold shift")
    ax.set_title("Qualitative, not quantitative")
    axes[3].text(0.02, 0.02, "○ nets 0–4   □ nets 10–24", fontsize=6, transform=axes[3].transAxes)


def tracked_runs(args):
    """Context reward runs with the threshold tracked at c = 0, labelled by the tracker's pace and
    baseline."""
    out = defaultdict(list)
    for r in runs.read(f"{args.results}/nf_context_reward_*_net*_r*.csv"):
        if not (protocol(r) <= TRACKED | {"baseline_decay"}) or r.get("track") != "balance":
            continue
        every = int(r.get("track_every") or 25)
        label = f"every {every}" + (", lagging baseline" if r.get("baseline") == "running" else "")
        out[label].append(r)
    return out


def adaptive(args):
    """An adaptive threshold at c = 0 against the reward learner: a fast tracker chases it into an
    all-abort state, a slow one stops tracking; none holds the balance and keeps the learner."""
    tracked = tracked_runs(args)
    fixed = [r for r in runs.read(f"{args.results}/nf_context_*_net*_r9.csv")
             if r["cond"] == "reward" and protocol(r) <= TRACKED and not r.get("track")
             and r.get("crit_d_end") and int(r.get("batch") or 48) == 48
             and (r.get("baseline") or "batch") == "batch"]
    order = [k for k in ("every 25", "every 100", "every 400", "every 1600",
                         "every 25, lagging baseline") if k in tracked]
    shades = dict(zip(order, ("#67000d", "#cb181d", "#fb6a4a", "#fcbba1", "#6a51a3")))
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.4))
    for label in order:
        for r in tracked[label]:
            every = int(r.get("track_every") or 25)
            off = runs.split(r["offset_traj"])
            ab = runs.split(r["abort_traj"])
            x = every * np.arange(1, len(off) + 1)
            axes[0].plot(x, off, color=shades[label], lw=0.8, alpha=0.8)
            axes[1].plot(x, ab, color=shades[label], lw=0.8, alpha=0.8)
        axes[0].plot([], [], color=shades[label], label=label)
    axes[0].set_xlabel("update")
    axes[0].set_ylabel("tracker offset")
    axes[0].set_title("The threshold chases the learner")
    axes[0].legend(fontsize=6)
    axes[1].set_xlabel("update")
    axes[1].set_ylabel("share of trials aborted")
    axes[1].set_title("…into an absorbing all-abort state")
    labels = ["fixed"] + order
    for i, label in enumerate(labels):
        sel = fixed if label == "fixed" else tracked[label]
        dots(axes[2], i, list(per_net(sel, lambda r: float(r["auc_end"])).values()),
             COLOURS["reward"] if label == "fixed" else shades[label])
    axes[2].axhline(0.5, color="k", lw=0.5, ls=":")
    axes[2].set_xticks(range(len(labels)),
                       [lbl.replace("every ", "").replace(", lagging baseline", "\nlag")
                        for lbl in labels])
    axes[2].set_xlabel("tracker interval (updates)")
    axes[2].set_ylabel("area under the ROC at the end")
    axes[2].set_title("No pace helps")
    fig.tight_layout()
    save(fig, args.out, "figS1_adaptive")


def context_deficit(args):
    """How much of reward's deficit on the context task a better gradient estimate recovers."""
    rows = [r for r in runs.read(f"{args.results}/nf_context_*_net*_r9.csv")
            if protocol(r) <= TRACKED and not r.get("track") and r.get("crit_d_end")]

    def label(r):
        if r["cond"] == "control":
            return "supervision"
        if r.get("baseline") == "type":
            return "reward,\ntrial-type\nbaseline"
        if int(r.get("batch") or 48) != 48:
            return "reward,\nbatch 192" + ("\n2000 updates\n(equal trials)"
                                             if int(r["updates"]) != BUDGET["context"]
                                             else "\n8000 updates\n(4× trials)")
        return "reward,\nbatch 48\n8000 updates"

    labels = ["reward,\nbatch 48\n8000 updates", "reward,\ntrial-type\nbaseline",
              "reward,\nbatch 192\n8000 updates\n(4× trials)",
              "reward,\nbatch 192\n2000 updates\n(equal trials)", "supervision"]
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    for i, lbl in enumerate(labels):
        v = list(per_net([r for r in rows if label(r) == lbl],
                         lambda r: float(r["auc_end"])).values())
        if v:
            dots(ax, i, v, COLOURS["control"] if lbl == "supervision" else COLOURS["reward"])
    ax.set_xticks(range(len(labels)), labels, fontsize=6)
    ax.set_ylabel("area under the ROC at the end")
    ax.set_title("Context task: more trials help, less variance does not")
    save(fig, args.out, "fig4_context_deficit")


def location_and_dose(axes, args):
    """Location carries part of c1's discriminability, and a magnified positional code moves the
    criterion in proportion to its gain without buying any."""
    loc = runs.read(f"{args.results}/location_c1_*.csv")
    loc += runs.read(f"{args.results}/location_mirror_c1_net*.csv")
    dose_rows = [r for r in runs.read(f"{args.results}/induction_dose_c1_net*.csv")
                 if not int(r["shuffle"])]
    ax = axes[0]
    x = 0
    ticks = []
    for state, name in (("theta0", "start"), ("control", "trained")):
        base = per_net([r for r in loc if r["state"] == state and r["variant"] == "task"],
                       lambda r: float(r["best_d"]))
        for variant, colour in (("at_cue", COLOURS["reward"]), ("mirror", COLOURS["control"])):
            v = per_net([r for r in loc if r["state"] == state and r["variant"] == variant],
                        lambda r: float(r["best_d"]))
            dots(ax, x, [v[n] - base[n] for n in sorted(base) if n in v], colour, jitter=0.07,
                 label={"at_cue": "at the cue", "mirror": "mirrored"}[variant]
                 if state == "theta0" else None)
            ticks.append((x, name))
            x += 1
        x += 0.5
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks([0.5, 3.0], ["start", "trained"])
    ax.set_ylabel("best-threshold d′ − task")
    ax.set_title("Distractors moved")
    ax.set_ylim(-1.1, 0.85)
    ax.legend(fontsize=6, loc="upper left", ncol=2, columnspacing=0.8, handletextpad=0.2)

    kappas = sorted({float(r["kappa"]) for r in dose_rows})
    nets = sorted({int(r["net"]) for r in dose_rows})
    for ax, key, label, title in (
            (axes[1], "d_offset", "best-threshold shift", "Criterion follows the gain"),
            (axes[2], "d_hold_sd", "Δ margin spread (SD), hold", "…through the margin's spread"),
            (axes[3], "d_best", "Δ best-threshold d′", "Discriminability falls")):
        for n in nets:
            sel = sorted((r for r in dose_rows if int(r["net"]) == n), key=lambda r: float(r["kappa"]))
            ax.plot([float(r["kappa"]) for r in sel], [float(r[key]) for r in sel], "-",
                    color=COLOURS["k2"], lw=0.5, alpha=0.35)
        mean = [np.mean([float(r[key]) for r in dose_rows if float(r["kappa"]) == k]) for k in kappas]
        ax.plot(kappas, mean, "o-", color="k", ms=3.5, lw=1.2)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(kappas, [f"{k:g}" for k in kappas])
        ax.set_xlabel("κ asked of the position head")
        ax.set_ylabel(label)
        ax.set_title(title)


def representational(args):
    """Targets aimed at the representation: the code behaviour uses, the dose, and the routes by
    which signals reach the criterion."""
    # two rows with different numbers of panels: each row is laid out on its own
    fig = plt.figure(figsize=(10.4, 5.4), layout="constrained")
    top, bottom = fig.subfigures(2, 1, hspace=0.06)
    location_and_dose(top.subplots(1, 4), args)
    routes(bottom.subplots(1, 5, gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25]}), args)
    save(fig, args.out, "fig5_representational")


def loss_anatomy(args):
    """What the loss after a representational target is made of: how much of it a target that is
    not about position costs too, and how much a read-out fitted to the end state returns."""
    swept_ = {(r["state"], int(r["net"])): r for r in runs.read(f"{args.results}/access_c1_net*.csv")}
    other = {(r["state"], int(r["net"])): r for r in runs.read(f"{args.results}/neutral_c1_net*.csv")}
    if not swept_ or not other:
        print("loss-anatomy: runs missing")
        return
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.8), gridspec_kw={"width_ratios": [1.1, 1.4, 0.8]})
    ax = axes[0]
    nets = sorted({n for _, n in other})
    targets = (("k2", "positional\nκ 2", COLOURS["k2"]), ("neutral k2", "neutral\nκ 2", "0.4"),
               ("neutral k3", "neutral\nκ 3", "0.4"), ("scrambled change", "scrambled\nchange", "0.65"))
    for i, (state, _, colour) in enumerate(targets):
        end = lambda n: swept_["k2", n] if state == "k2" else other[state, n]
        dots(ax, i, [float(end(n)["frozen_best_d"]) - float(swept_["theta0", n]["frozen_best_d"])
                     for n in nets], colour)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(targets)), [label for _, label, _ in targets], fontsize=6.5)
    ax.set_ylabel("Δ best-threshold d′")
    ax.set_title("Most of the loss is the target's")

    ax = axes[1]
    states = (("theta0", "start"), ("control", "supervised"), ("k2", "κ 2"), ("shuffled", "shuffled"))
    nets = sorted(n for state, n in swept_ if state == "theta0")
    for i, (state, _) in enumerate(states):
        for key, dx, colour, label in (("frozen_best_d", -0.19, "0.4", "frozen read-out"),
                                       ("refit_best_d", 0.19, "#e6ab02", "read-out refitted")):
            dots(ax, i + dx, [float(swept_[state, n][key]) for n in nets if (state, n) in swept_],
                 colour, jitter=0.05, mean_dx=0.0, label=label if i == 0 else None)
    ax.set_xticks(range(len(states)), [label for _, label in states])
    ax.set_ylabel("best-threshold d′")
    ax.set_title("A refitted read-out returns part of it")
    ax.legend(fontsize=6, loc="lower left", handletextpad=0.2)

    # the fall at the decoder's own threshold, by what gives it back
    ax = axes[2]
    fall = lambda state, key: np.mean([float(swept_[state, n][key]) - float(swept_["theta0", n][key])
                                       for n in nets])
    parts = (("returned by the threshold", "0.8"), ("by a refitted read-out", "#e6ab02"),
             ("remaining", "0.25"))
    for i, state in enumerate(("k2", "shuffled")):
        own, best, refit = (fall(state, k) for k in ("frozen_own_d", "frozen_best_d", "refit_best_d"))
        bottom = 0.0
        for (label, colour), size in zip(parts, (own - best, best - refit, refit)):
            ax.bar(i, size, bottom=bottom, width=0.6, color=colour, label=label if i == 0 else None)
            bottom += size
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(2), ["κ 2", "shuffled"])
    ax.set_ylabel("Δ d′ at the own threshold")
    ax.set_title("Three parts of the fall")
    # room above zero for the legend, which would otherwise sit on the bars
    ax.set_ylim(top=0.75)
    ax.legend(fontsize=5.5, loc="upper center", handlelength=1.0)
    fig.tight_layout()
    save(fig, args.out, "fig6_loss_anatomy")


def wrong_sign(args):
    """After task training the code is more separable and discriminability is higher, while the
    decoder fitted at the start reads it worse: block 52's two arms read as in block 54."""
    rows = runs.read(f"{args.results}/relocated_c1_net*.csv")
    loops = runs.read(f"{args.results}/closed_loop_*_net*_r0.csv")
    if not rows or not loops:
        print("wrong-sign: runs missing")
        return
    states = (("theta0", "start"), ("closed", "closed\nloop"), ("yoked", "yoked"))
    value = lambda state, key: per_net([r for r in rows if r["state"] == state],
                                       lambda r: float(r[key]))
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.5))
    ax = axes[0]
    for key, colour, dx, label in (("frozen_acc", "0.35", -0.17, "decoder fitted at the start"),
                                   ("refit_acc", COLOURS["k2"], 0.17, "decoder refitted")):
        for i, (state, _) in enumerate(states):
            dots(ax, i + dx, list(value(state, key).values()), colour, jitter=0.05, mean_dx=0.0,
                 label=label if i == 0 else None)
    ax.axhline(0.125, color="k", lw=0.5, ls=":")
    ax.set_xticks(range(3), [name for _, name in states])
    ax.set_ylim(0.08, 0.95)
    ax.set_ylabel("accuracy, 8 position bins")
    ax.set_title("What the fixed decoder reports")
    ax.legend(fontsize=6, loc="upper left", handletextpad=0.2)

    ax = axes[1]
    for i, (state, _) in enumerate(states):
        dots(ax, i, list(value(state, "crossnobis").values()), COLOURS["k2"], jitter=0.06)
    ax.set_xticks(range(3), [name for _, name in states])
    ax.set_ylabel("crossnobis of the delay code")
    ax.set_title("…what happened to the code")

    ax = axes[2]
    for i, kind in enumerate(("closed", "yoked")):
        sel = [r for r in loops if r["kind"] == kind]
        dots(ax, i, [float(r["best_d_end"]) - float(r["best_d_start"]) for r in sel],
             COLOURS["control"], jitter=0.06)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(2), ["closed\nloop", "yoked"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("Δ best-threshold d′")
    ax.set_title("…and to discriminability")

    ax = axes[3]
    mirror = runs.read(f"{args.results}/location_mirror_c1_net*.csv")
    at_cue = {"theta0": per_net([r for r in runs.read(f"{args.results}/location_c1_*.csv")
                                 if r["state"] == "theta0" and r["variant"] == "at_cue"],
                                lambda r: float(r["best_d"]))}
    for state in ("closed", "yoked"):
        at_cue[state] = {int(r["net"]): float(r["best_at_cue"]) for r in rows if r["state"] == state}
    for i, (state, _) in enumerate(states):
        control = per_net([r for r in mirror if r["state"] == state and int(r["net"]) >= 10],
                          lambda r: float(r["best_d"]))
        dots(ax, i, [at_cue[state][n] - control[n] for n in sorted(control) if n in at_cue[state]],
             "0.35", jitter=0.06)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(3), [name for _, name in states])
    ax.set_ylabel("best d′, distractors\nat the cue − mirrored")
    ax.set_title("…and to its use of location")
    fig.tight_layout()
    save(fig, args.out, "fig7_wrong_sign")


FIGURES = {"headroom": headroom, "one-roc": one_roc, "calibration": calibration,
           "context-deficit": context_deficit, "representational": representational,
           "loss-anatomy": loss_anatomy, "wrong-sign": wrong_sign, "adaptive": adaptive}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("figure", nargs="?", choices=list(FIGURES))
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/figures")
    args = p.parse_args()
    # the jitter of the dots only; fixed so the figures do not change between builds
    np.random.seed(0)
    for name, draw in FIGURES.items():
        if args.figure in (None, name):
            draw(args)


if __name__ == "__main__":
    main()
