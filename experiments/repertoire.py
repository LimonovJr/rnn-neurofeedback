"""Whether a learner on the context task reuses the starting network's activity patterns or makes
new ones.

Brain-computer-interface learning is fast when it re-associates patterns the network already
produces and slow when it needs new ones (Golub et al., 2018; Oby et al., 2019). If the reward
learner on the context task mostly re-associates, while the conjunction the task needs calls for
new patterns, its deficit against supervision would have a measured cause rather than the untested
reading in terms of information per trial.

The repertoire is the starting network's activity on a fixed batch of context trials, read with the
noise off: its top-k principal subspace and, inside it, the distribution of its states. For a
trained state, on the same batch:
- the share of its activity variance outside the starting subspace;
- the share of its time steps whose Mahalanobis distance inside that subspace exceeds the 99th
  percentile of the starting network's own states;
- the share of its change in mean activity along the frozen decision axis relative to a random
  direction.
Supervision improves discriminability more than reward, so any difference is compared at matched
performance: the supervised states at 2000 and 4000 updates are swept here for their area under
the ROC.

    python -m experiments.repertoire --nets 0,1
"""

import argparse
import glob
import os

import numpy as np
import torch

from src import behaviour, runs, tasks, trials
from src.model import CKPT, load

ENV = tasks.context()
K = 10


def restore(net, path=None):
    model = load(net, head="context")
    if path:
        model.load_state_dict(torch.load(path, weights_only=True)["state_dict"], strict=False)
    model.eval()
    return model


def activity(model, batch):
    x, _, _, _, _, _, lengths = batch
    with torch.no_grad():
        h = model(x, return_hidden=True)[2]
    valid = torch.arange(x.shape[1])[None, :] < lengths[:, None]
    return h[valid].numpy()


def states(net):
    yield "theta0", None
    for arm, tag in (("control", "control_lam1_n10000_crit"), ("reward", "reward_lam1_n10000_crit"),
                     ("reward b192", "reward_lam1_n10000_crit_b192")):
        for upd in (2000, 4000, 8000):
            path = CKPT / "nf_context_b37" / f"nf_context_{tag}_net{net}_r9_u{upd}.pt"
            if path.exists():
                yield f"{arm} u{upd}", path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nets", required=True)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--sweep-n", default="2000,10000")
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or f"results/repertoire_context_{a.nets.replace(',', '_')}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    n_select, n_report = (int(v) for v in a.sweep_n.split(","))
    ends = {(int(r["net"]), r["_file"]): r for r in runs.read("results/nf_context_*_r9.csv")}
    rows = []
    for net in (int(n) for n in a.nets.split(",")):
        torch.manual_seed(31 + net)
        np.random.seed(31 + net)
        batch = trials.batch(ENV, a.n, 20)
        h0 = activity(restore(net), batch)
        mean0 = h0.mean(0)
        _, sv, vt = np.linalg.svd(h0 - mean0, full_matrices=False)
        basis = vt[:K]
        z0 = (h0 - mean0) @ basis.T
        cov_inv = np.linalg.inv(np.cov(z0.T))
        dist = lambda z: np.sqrt(np.einsum("ij,jk,ik->i", z, cov_inv, z))
        limit = np.percentile(dist(z0), 99)
        w = None
        for state, path in states(net):
            model = restore(net, path)
            if w is None:
                w = (model.W_out_eff()[1] - model.W_out_eff()[0]).detach().numpy()
                w = w / np.linalg.norm(w)
            h = activity(model, batch)
            centred = h - h.mean(0)
            inside = centred @ basis.T
            outside = 1 - (inside ** 2).sum() / (centred ** 2).sum()
            shift = h.mean(0) - mean0
            potent = (shift @ w) ** 2 / max((shift ** 2).sum(), 1e-12) * len(w)
            row = dict(net=net, state=state, outside_share=round(float(outside), 4),
                       outside_start=round(float(1 - (sv[:K] ** 2).sum() / (sv ** 2).sum()), 4),
                       new_patterns=round(float((dist((h - mean0) @ basis.T) > limit).mean()), 4),
                       mean_shift=round(float(np.linalg.norm(shift)), 4),
                       potent_over_random=round(float(potent), 3))
            stem = os.path.basename(str(path))[:-len("_u8000.pt")] if path else ""
            if path and str(path).endswith("_u8000.pt") and (net, stem + ".csv") in ends:
                row["auc"] = ends[(net, stem + ".csv")]["auc_end"]
            elif path is None or state.startswith("control"):
                row["auc"] = behaviour.threshold_sweep(model, ENV, behaviour.go_nogo, n_select,
                                                       n_report)["auc"]
            rows.append(row)
            print(f"  [net {net} {state:<16}] outside {row['outside_share']:.3f} "
                  f"new {row['new_patterns']:.3f} potent x{row['potent_over_random']:.1f} "
                  f"auc {row.get('auc', '')}", flush=True)
    runs.write(a.out, rows)


if __name__ == "__main__":
    main()
