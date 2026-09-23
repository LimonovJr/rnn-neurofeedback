"""Leaky E/I network on a 2-D sheet with geometric receptive fields (Chen & Gong 2022)."""

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT = Path(os.environ.get("NF_CHECKPOINTS", Path(__file__).resolve().parents[1] / "checkpoints"))

HIDDEN = 180
SHEET = 12
NOISE_SD = 0.10


class BioLeakyRNNTopo(nn.Module):
    """144 E neurons on a 12x12 sheet plus 36 I neurons at random positions; cue and stimulus
    reach the sheet through fixed Gaussian receptive fields, recurrent connections through a
    distance-dependent mask."""

    def __init__(self, hidden_size=HIDDEN, output_size=2, dt=20.0, tau=100.0, noise_sd=NOISE_SD,
                 mask_seed: Optional[int] = None, sheet_side=SHEET, tau_ee=0.25, tau_ie=0.32,
                 tau_ei=0.64, tau_ii=0.64, rf_sigma=0.3, tau_e_range=(50, 150), tau_i_range=(10, 50)):
        super().__init__()
        self.input_size = 7
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dt = float(dt)
        self.tau = float(tau)
        self.rf_sigma = float(rf_sigma)
        self.sheet_side = int(sheet_side)
        self.phi = F.softplus

        n_exc = sheet_side * sheet_side
        n_inh = hidden_size - n_exc
        if n_inh < 1:
            raise ValueError(f"hidden_size={hidden_size} leaves no room for I units on a {sheet_side} sheet")
        self.n_exc, self.n_inh = n_exc, n_inh

        ei_sign = torch.ones(hidden_size)
        ei_sign[n_exc:] = -1.0
        self.register_buffer("ei_sign", ei_sign)

        gen_tau = torch.Generator()
        gen_tau.manual_seed((mask_seed or 0) + 7)
        tau_per = torch.full((hidden_size,), float(tau))
        tau_per[:n_exc] = _log_uniform(tau_e_range, n_exc, gen_tau)
        tau_per[n_exc:] = _log_uniform(tau_i_range, n_inh, gen_tau)
        alpha = 1.0 - torch.exp(-float(dt) / tau_per)  # exp-Euler, so taus below dt stay stable
        self.register_buffer("alpha", alpha)
        self.register_buffer("tau_per_neuron", tau_per)
        self.register_buffer("sigma_eff", (torch.sqrt(2.0 / alpha) * noise_sd).float())

        coords = sheet_coords(n_exc, n_inh, sheet_side, mask_seed)
        self.register_buffer("coords", coords)
        self.register_buffer("rec_mask", sheet_mask(coords, n_exc, n_inh, tau_ee, tau_ie, tau_ei,
                                                    tau_ii, mask_seed))
        self.register_buffer("out_mask", torch.ones(output_size, hidden_size))

        self.W_in_fix = nn.Parameter(torch.empty(hidden_size))     # fixation is the only learned input
        self.W_rec = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.W_out = nn.Parameter(torch.empty(output_size, hidden_size))
        self.b_out = nn.Parameter(torch.zeros(output_size))
        self.W_aux = nn.Parameter(torch.empty(2, hidden_size))     # position head, outside the Dale pathway
        self.b_aux = nn.Parameter(torch.zeros(2))
        self.register_buffer("h0", torch.zeros(hidden_size))

        self.rec_noise = False
        self.sigma_in = 0.0
        self.noise_at_eval = False
        self.readout_head = None
        self.noise_log = None            # a list makes step() keep each draw, for node perturbation
        self.noise_basis = None          # an orthonormal [hidden, k] basis confines the noise to it
        self.preact_log = None           # a list keeps the noise-free drive, for reinforcement on the state
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.W_in_fix.normal_(0.0, 1.0 / math.sqrt(self.input_size))
            self.W_rec.zero_()
            self.W_rec.add_(torch.diag(0.8 * torch.ones(self.hidden_size)))
            self.W_rec.add_(0.01 * torch.randn_like(self.W_rec))
            self.W_out.normal_(0.0, 1.0 / math.sqrt(self.hidden_size))
            self.W_aux.normal_(0.0, 1.0 / math.sqrt(self.hidden_size))
            self.W_rec.abs_()
            self.W_rec.mul_(self.rec_mask)
            self.W_out.mul_(self.out_mask)

    def init_hidden(self, batch_size, device):
        return self.h0.unsqueeze(0).expand(batch_size, -1).clone().to(device)

    def rf_drive(self, xy_s):
        """(x, y, strength) -> per-neuron drive, with distance wrapped on the torus."""
        xy, s = xy_s[..., :2], xy_s[..., 2:3]
        diff = self.coords.view(*([1] * (xy.dim() - 1)), self.hidden_size, 2) - xy.unsqueeze(-2)
        diff = diff - 2.0 * torch.round(diff / 2.0)
        return s * torch.exp(-diff.pow(2).sum(dim=-1) / (2.0 * self.rf_sigma ** 2))

    def W_rec_eff(self):
        return (self.W_rec * self.rec_mask).abs() * self.ei_sign.view(1, -1)

    def W_out_eff(self):
        return self.W_out * self.out_mask

    def decode_xy(self, h):
        return F.linear(h, self.W_aux, self.b_aux)

    def step(self, x_t, h_t):
        drive_fix = x_t[..., 0:1] * self.W_in_fix.view(1, -1)
        drive_cue = self.rf_drive(x_t[..., 1:4])
        drive_stim = self.rf_drive(x_t[..., 4:7])

        noisy = self.training or self.noise_at_eval
        if noisy and self.sigma_in > 0.0:
            # input noise degrades perception, unlike rec_noise which corrupts internal maintenance
            sensory = drive_cue + drive_stim
            drive_cue, drive_stim = sensory + self.sigma_in * torch.randn_like(sensory), 0.0

        preact = drive_fix + drive_cue + drive_stim + F.linear(h_t, self.W_rec_eff()) + self.b_h
        if self.preact_log is not None:
            self.preact_log.append(preact)
        if noisy and self.rec_noise:
            if self.noise_basis is None:
                xi = self.sigma_eff * torch.randn_like(preact)
            else:
                z = torch.randn(preact.shape[:-1] + (self.noise_basis.shape[1],),
                                device=preact.device)
                xi = self.sigma_eff * (z @ self.noise_basis.T)
            if self.noise_log is not None:
                self.noise_log.append(xi)
            preact = preact + xi

        h_next = (1.0 - self.alpha) * h_t + self.alpha * self.phi(preact)
        return h_next, F.linear(h_next, self.W_out_eff(), self.b_out)

    def forward(self, x, h0=None, return_hidden=False):
        B, T, D = x.shape
        if D != self.input_size:
            raise ValueError(f"expected {self.input_size} input channels, got {D}")

        h = self.init_hidden(B, x.device) if h0 is None else h0
        logits, hidden = [], []
        keep_hidden = return_hidden or self.readout_head is not None
        for t in range(T):
            h, y_t = self.step(x[:, t, :], h)
            logits.append(y_t)
            if keep_hidden:
                hidden.append(h)

        if self.readout_head is not None:
            out = self.readout_head(torch.stack(hidden, dim=1))
        else:
            out = torch.stack(logits, dim=1)
        if return_hidden:
            return out, h, torch.stack(hidden, dim=1)
        return out, h


def _log_uniform(rng_range, n, generator):
    lo, hi = float(rng_range[0]), float(rng_range[1])
    u = torch.rand(n, generator=generator)
    return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


def sheet_coords(n_exc, n_inh, sheet_side, seed):
    """E units on the integer grid rescaled to [-1, 1]^2, I units uniformly random, E first."""
    idx = torch.arange(n_exc)
    denom = max(1.0, float(sheet_side - 1))
    e = torch.stack([(idx % sheet_side).float() / denom * 2.0 - 1.0,
                     (idx // sheet_side).float() / denom * 2.0 - 1.0], dim=1)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.cat([e, torch.rand(n_inh, 2, generator=gen) * 2.0 - 1.0], dim=0)


def sheet_mask(coords, n_exc, n_inh, tau_ee, tau_ie, tau_ei, tau_ii, seed):
    """p_ij = exp(-D_ij / tau_class) on a torus of period 2, so edge units are not special.
    mask[i, j] is post-i <- pre-j."""
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    diff = diff - 2.0 * torch.round(diff / 2.0)
    d = diff.norm(dim=-1)

    n = n_exc + n_inh
    taus = torch.zeros(n, n)
    taus[:n_exc, :n_exc] = tau_ee
    taus[n_exc:, :n_exc] = tau_ie
    taus[:n_exc, n_exc:] = tau_ei
    taus[n_exc:, n_exc:] = tau_ii

    gen = torch.Generator()
    gen.manual_seed(seed + 1)
    return (torch.rand(n, n, generator=gen) < torch.exp(-d / taus)).float()


def build(net):
    """A network instance: the seed fixes sheet layout, masks and time constants."""
    return BioLeakyRNNTopo(mask_seed=int(net))


def load(net, stage=2, head=None):
    """Network `net` at the end of pretraining stage `stage`, or with a refitted position head."""
    name = f"head_{head}.pt" if head else f"stage{stage}.pt"
    m = build(net)
    state = torch.load(CKPT / f"net{net}" / name, weights_only=True)["state_dict"]
    m.load_state_dict(state, strict=False)
    m.rec_noise = True
    m.eval()
    return m


def save(model, path, **meta):
    """Takes a network or a state dict already captured from one."""
    state = model.state_dict() if isinstance(model, nn.Module) else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, **meta}, path)


def mlp_readout(hidden, seed):
    """Nonlinear read-out for Arm A+; seeded per restart so the restarts are real head draws."""
    def factory():
        torch.manual_seed(seed)
        np.random.seed(seed)
        return nn.Sequential(nn.Linear(HIDDEN, hidden), nn.Tanh(), nn.Linear(hidden, 2))
    return factory
