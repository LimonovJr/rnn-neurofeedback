"""The figures of the report, drawn from the same per-run CSVs as the tables in `aggregate`, so a
figure and the table it illustrates cannot disagree.

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


def dots(ax, x, values, colour, jitter=0.08, mean_dx=0.22, label=None):
    """One dot per network, and the mean with its standard error beside them."""
    values = np.asarray(values, dtype=float)
    ax.scatter(x + np.random.uniform(-jitter, jitter, len(values)), values, s=10, color=colour,
               alpha=0.6, lw=0, zorder=3, label=label)
    sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    ax.errorbar(x + mean_dx, values.mean(), yerr=sem, fmt="o", color="k", ms=3.5, capsize=2,
                zorder=4)


def headroom(args):
    """What better dynamics could still buy through the frozen decoder: H = C_B - C_A per network,
    on the cell tasks and on the context task."""
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    labels = []
    for i, setting in enumerate(SETTINGS):
        rows = runs.read(f"{args.results}/headroom_{setting}_*_net*.csv")
        c = {a: by_net([r for r in rows if r["arm"] == a], lambda r: float(r["C_mean"]))
             for a in ("A", "B")}
        nets = sorted(set(c["A"]) & set(c["B"]))
        dots(ax, i, [st.median(c["B"][n]) - st.median(c["A"][n]) for n in nets], "#555555")
        labels.append(setting)
    rows = runs.read(f"{args.results}/control_net*.csv")
    c = {a: by_net([r for r in rows if r["arm"] == a], lambda r: float(r["reeval"]))
         for a in ("A", "B")}
    nets = sorted(set(c["A"]) & set(c["B"]))
    dots(ax, len(labels), [st.median(c["B"][n]) - st.median(c["A"][n]) for n in nets],
         COLOURS["reward"])
    labels.append("context")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhspan(-RES_BEHAVIOUR, RES_BEHAVIOUR, color="0.9", zorder=0, lw=0)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("headroom H = C_B − C_A (d′)")
    ax.set_title("Decoder-limited cells, dynamics-limited context")
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


def mean_roc(rows, grid=np.linspace(0, 1, 101)):
    """Each network's hull, averaged vertically over false-alarm rates, and the mean point at the
    decoder's own threshold."""
    curves, own = defaultdict(list), defaultdict(list)
    for r in rows:
        pts = roc_points(r["roc_end"])
        h = hull([(fa, hit) for _, hit, fa, *_ in pts])
        curves[int(r["net"])].append(np.interp(grid, h[:, 0], h[:, 1]))
        own[int(r["net"])].append(next((fa, hit) for o, hit, fa, *_ in pts if o == 0.0))
    mean = np.mean([np.mean(v, 0) for v in curves.values()], 0)
    return grid, mean, np.mean([np.mean(v, 0) for v in own.values()], 0)


def one_roc(args):
    """Reward and supervision on the same ROC on c1, not on context (networks 10-24)."""
    nets = set(range(10, 25))
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.8))
    for ax, task, title in zip(axes, ("c1", "context"), ("c1: one ROC", "context: two ROCs")):
        rows = [r for r in swept(args, task) if int(r["net"]) in nets and int(r["rep"]) in (0, 1)]
        if not rows:
            ax.set_title(f"{task}: no runs")
            continue
        for cond in ("control", "reward"):
            grid, curve, own = mean_roc([r for r in rows if r["cond"] == cond])
            ax.plot(grid, curve, "-", color=COLOURS[cond], lw=1.2, label=NAMES[cond])
            ax.plot(*own, "o", color=COLOURS[cond], ms=6, mec="k", mew=0.6)
        ax.plot([0, 1], [0, 1], ":", color="0.6", lw=0.8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("false-alarm rate")
        ax.set_ylabel("hit rate")
        ax.set_title(title)
    axes[0].legend(loc="lower right", fontsize=7)
    axes[1].text(0.98, 0.04, "circles: the decoder's own threshold", ha="right", fontsize=6.5,
                 transform=axes[1].transAxes)
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
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.5))
    shades = dict(zip(costs, ("#fdae6b", "#e6550d", "#a63603")))
    for cost in costs:
        for readout, style in (("sampled", "-"), ("greedy", "--")):
            sel = [r for r in rows if float(r["trained_cost"]) == cost and r["readout"] == readout]
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
    axes[0].text(0.98, 0.04, "solid: sampled, noise on\ndashed: greedy, noise off", ha="right",
                 fontsize=6, transform=axes[0].transAxes)

    for i, cost in enumerate(costs):
        for readout, dx, colour in (("sampled", -0.2, shades[cost]), ("greedy", 0.2, "0.6")):
            sel = [r for r in rows if float(r["trained_cost"]) == cost and r["readout"] == readout]
            lost = []
            for r in sel:
                pts = {o: c - cost * f - m - a for o, _, _, c, f, m, a in roc_points(r["roc"])}
                lost.append(max(pts.values()) - pts[0.0])
            dots(axes[1], i + dx, lost, colour, jitter=0.05, mean_dx=0.0,
                 label=("sampled, noise on" if cost == costs[-1] else None)
                 if readout == "sampled" else ("greedy, noise off" if cost == costs[-1] else None))
    axes[1].set_xticks(range(len(costs)), [f"{c:g}" for c in costs])
    axes[1].set_xlabel("false-alarm cost trained with")
    axes[1].set_ylabel("payoff lost at own threshold")
    axes[1].set_title("Payoff given away")
    axes[1].legend(fontsize=6.5, loc="upper left")

    for i, cost in enumerate(costs):
        sel = [r for r in rows if float(r["trained_cost"]) == cost and r["readout"] == "sampled"]
        dots(axes[2], i, [float(r["fa_at_zero"]) for r in sel], shades[cost], jitter=0.05)
    axes[2].set_xticks(range(len(costs)), [f"{c:g}" for c in costs])
    axes[2].set_xlabel("false-alarm cost trained with")
    axes[2].set_ylabel("false alarms at own threshold")
    axes[2].set_title("The point follows the cost")
    fig.tight_layout()
    save(fig, args.out, "fig3_calibration")


def routes(args):
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
    measures = (("d_hold_level", "hold level"), ("d_hold_sd", "hold spread (SD)"),
                ("d_resp_max", "response peak"), ("d_offset", "best threshold"))
    fig, axes = plt.subplots(1, 5, figsize=(9.6, 2.3),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.25]})
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
    fig.tight_layout()
    save(fig, args.out, "fig4_routes")


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
    axes[2].set_ylabel("end AUC")
    axes[2].set_title("No pace helps")
    fig.tight_layout()
    save(fig, args.out, "fig5_adaptive")


def context_deficit(args):
    """How much of reward's deficit on the context task a better gradient estimate recovers."""
    rows = [r for r in runs.read(f"{args.results}/nf_context_*_net*_r9.csv")
            if protocol(r) <= TRACKED and not r.get("track") and r.get("crit_d_end")]

    def label(r):
        if r["cond"] == "control":
            return "supervision"
        if r.get("baseline") == "type":
            return "trial-type\nbaseline"
        if int(r.get("batch") or 48) != 48:
            return f"batch {r['batch']}" + ("\nequal trials" if int(r["updates"])
                                             != BUDGET["context"] else "")
        return "reward"

    labels = ["reward", "trial-type\nbaseline", "batch 192", "batch 192\nequal trials",
              "supervision"]
    fig, ax = plt.subplots(figsize=(4.0, 2.5))
    for i, lbl in enumerate(labels):
        v = list(per_net([r for r in rows if label(r) == lbl],
                         lambda r: float(r["auc_end"])).values())
        if v:
            dots(ax, i, v, COLOURS["control"] if lbl == "supervision" else COLOURS["reward"])
    ax.set_xticks(range(len(labels)), labels, fontsize=6.5)
    ax.set_ylabel("end AUC")
    ax.set_title("Context: reward's deficit and its estimate")
    save(fig, args.out, "fig6_context_deficit")


def location_and_dose(args):
    """Location carries part of c1's discriminability, and a magnified positional code moves the
    criterion in proportion to its gain without buying any."""
    loc = runs.read(f"{args.results}/location_c1_*.csv")
    dose_rows = [r for r in runs.read(f"{args.results}/induction_dose_c1_net*.csv")
                 if not int(r["shuffle"])]
    if not loc or not dose_rows:
        print("location-dose: runs missing")
        return
    fig, axes = plt.subplots(1, 4, figsize=(9.4, 2.5))
    ax = axes[0]
    x = 0
    ticks = []
    for state, name in (("theta0", "start"), ("control", "trained")):
        base = per_net([r for r in loc if r["state"] == state and r["variant"] == "task"],
                       lambda r: float(r["best_d"]))
        for variant, colour in (("at_cue", COLOURS["reward"]), ("opposite", COLOURS["control"])):
            v = per_net([r for r in loc if r["state"] == state and r["variant"] == variant],
                        lambda r: float(r["best_d"]))
            dots(ax, x, [v[n] - base[n] for n in sorted(base) if n in v], colour, jitter=0.07,
                 label={"at_cue": "at the cue", "opposite": "opposite the cue"}[variant]
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
            (axes[2], "d_hold_sd", "Δ hold spread (SD)", "…through the margin's spread"),
            (axes[3], "d_best", "Δ best-threshold d′", "No discriminability")):
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
    fig.tight_layout()
    save(fig, args.out, "fig7_location_dose")


FIGURES = {"headroom": headroom, "one-roc": one_roc, "calibration": calibration,
           "routes": routes, "adaptive": adaptive, "context-deficit": context_deficit,
           "location-dose": location_and_dose}


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
