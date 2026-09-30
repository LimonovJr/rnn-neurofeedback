"""Cued detection tasks. The observation is 7 channels sampled at 20 ms:

    [0] fixation      (also carries the context level in ContextReversal)
    [1:4] cue x, y, strength
    [4:7] stimulus x, y, strength      (target or distractor)

Position enters the network through fixed Gaussian receptive fields, so the raw coordinates are
written into the channels and presence is encoded by the strength channel being non-zero.
"""

import contextlib
import math

import numpy as np

if not hasattr(np, "_no_nep50_warning"):
    # neurogym still calls this numpy-1.x internal
    @contextlib.contextmanager
    def _no_nep50_warning():
        yield
    np._no_nep50_warning = _no_nep50_warning

from neurogym import spaces
from neurogym.core import TrialEnv

C1 = (0.46, 0.90, 1.12)                                  # cue, target, distractor strength
PATCH_CELL = dict(cue=0.5, tgt=1.0, dstr=1.0, r_patch=0.35, jitter=0.30, T_avg=20.0)


class CuedDetection(TrialEnv):
    """Hold fixation through a randomised cue-to-target delay, ignore distractor flashes, release
    when the target appears at the cued location."""

    def __init__(self, dt=20, rt_window=(150, 750), cue_strength=1.0, target_strength=1.0,
                 distractor_strength=1.0, distractor_duration=100, p_distractor_trial=0.6,
                 max_distractors=4, min_gap=80, fixation_jitter=(700, 1200), cue_duration=350,
                 post_target_duration=900, ctoa_range=(1000, 3300), ctoa_beta=(2.2, 1.6),
                 distractor_hazard_early=0.10, distractor_hazard_late=0.01,
                 cue_radius=0.4, target_radius=0.7, distractor_min_sep_rad=math.pi / 3):
        super().__init__(dt=dt)
        self.rt_window = tuple(rt_window)
        self.cue_strength = float(cue_strength)
        self.target_strength = float(target_strength)
        self.distractor_strength = float(distractor_strength)
        self.distractor_duration = int(distractor_duration)
        self.p_distractor_trial = float(p_distractor_trial)
        self.max_distractors = int(max_distractors)
        self.min_gap = int(min_gap)
        self.ctoa_range = tuple(ctoa_range)
        self.ctoa_beta = tuple(ctoa_beta)
        self.distractor_hazard_early = float(distractor_hazard_early)
        self.distractor_hazard_late = float(distractor_hazard_late)
        self.cue_radius = float(cue_radius)
        self.target_radius = float(target_radius)
        self.distractor_min_sep_rad = float(distractor_min_sep_rad)

        self.timing = {"fixation": tuple(fixation_jitter), "cue": int(cue_duration),
                       "delay": 1500, "target": 100, "post_target": int(post_target_duration)}
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def ctoa_ms(self):
        lo, hi = self.ctoa_range
        a, b = self.ctoa_beta
        return int(lo + self.rng.beta(a, b) * (hi - lo))

    def ctoa_bin(self, ctoa_ms):
        edges = np.linspace(self.ctoa_range[0], self.ctoa_range[1], 11)
        return int(np.digitize([ctoa_ms], edges[1:-1])[0])

    def quadrant(self, theta):
        x, y = math.cos(theta), math.sin(theta)
        if x < 0 and y >= 0:
            return 1
        if x >= 0 and y >= 0:
            return 2
        if x < 0 and y < 0:
            return 3
        return 4

    def positions(self):
        """Cue and target on the ring at a shared random angle."""
        theta = float(self.rng.uniform(0.0, 2.0 * math.pi))
        cue = (self.cue_radius * math.cos(theta), self.cue_radius * math.sin(theta))
        target = (self.target_radius * math.cos(theta), self.target_radius * math.sin(theta))
        return cue, target, theta, self.quadrant(theta)

    def distractor_theta(self, cue_theta, max_tries=64):
        two_pi = 2.0 * math.pi
        for _ in range(max_tries):
            t = float(self.rng.uniform(0.0, two_pi))
            if abs((t - cue_theta + math.pi) % two_pi - math.pi) >= self.distractor_min_sep_rad:
                return t
        return (cue_theta + math.pi) % two_pi

    def distractor_position(self, cue_theta):
        theta = self.distractor_theta(cue_theta)
        return (self.target_radius * math.cos(theta), self.target_radius * math.sin(theta)), theta

    def distractor_onsets(self, delay_start, t_target0, dur_steps):
        """Onsets drawn from a hazard that falls through the delay, with a refractory gap."""
        onsets, blocked_until = [], -1
        gap_steps = max(1, int(round(self.min_gap / self.dt)))
        latest = max(delay_start, t_target0 - dur_steps - 1)
        if latest <= delay_start:
            return onsets

        denom = max(1, latest - delay_start)
        for t in range(delay_start, latest + 1):
            if len(onsets) >= self.max_distractors:
                break
            if t <= blocked_until:
                continue
            frac = (t - delay_start) / denom
            p_on = self.distractor_hazard_early * (1 - frac) + self.distractor_hazard_late * frac
            if self.rng.rand() < p_on:
                onsets.append(int(t))
                blocked_until = t + dur_steps + gap_steps
        return onsets

    def write_cue(self, cue_xy):
        if self.cue_strength == 0.0:
            return
        t0, t1 = self.start_ind["cue"], self.end_ind["cue"]
        self.ob[t0:t1, 1], self.ob[t0:t1, 2] = cue_xy
        self.ob[t0:t1, 3] = self.cue_strength

    def write_target(self, target_xy):
        if self.target_strength == 0.0:
            return
        t0, t1 = self.start_ind["target"], self.end_ind["target"]
        self.ob[t0:t1, 4], self.ob[t0:t1, 5] = target_xy
        self.ob[t0:t1, 6] = self.target_strength

    def write_distractor(self, t0, t1, d_xy):
        if self.distractor_strength == 0.0:
            return
        self.ob[t0:t1, 4], self.ob[t0:t1, 5] = d_xy
        self.ob[t0:t1, 6] = self.distractor_strength

    def _new_trial(self, **kwargs):
        cue_pos, target_pos, cue_theta, cue_loc = self.positions()
        ctoa = self.ctoa_ms()
        self.timing["delay"] = ctoa

        trial = {"cue_loc": cue_loc, "target_loc": cue_loc, "cue_pos": cue_pos,
                 "target_pos": target_pos, "cue_theta": cue_theta, "target_theta": cue_theta,
                 "ctoa_ms": ctoa, "ctoa_bin": self.ctoa_bin(ctoa), "has_distractors": False,
                 "n_distractors": 0, "distractor_positions": [], "distractor_thetas": []}
        trial.update(kwargs)

        self.add_period(["fixation", "cue", "delay", "target", "post_target"])
        T = self.end_ind["post_target"]
        self.ob = np.zeros((T, 7), dtype=np.float32)
        self.gt = np.zeros(T, dtype=np.int64)
        self.ob[:, 0] = 1.0
        self.write_cue(cue_pos)
        self.write_target(target_pos)

        t_target0 = self.start_ind["target"]
        rt0 = int(round(self.rt_window[0] / self.dt))
        rt1 = int(round(self.rt_window[1] / self.dt))
        dur_steps = max(1, int(round(self.distractor_duration / self.dt)))
        win_start = int(np.clip(t_target0 + rt0, 0, T))
        win_end = int(np.clip(t_target0 + rt1, 0, T))

        self._resp_window = np.zeros(T, dtype=bool)
        if win_end > win_start:
            self._resp_window[win_start:win_end] = True
            self.gt[win_start:win_end] = 1
        self._t_target0 = int(t_target0)

        self._fa_window = np.zeros(T, dtype=bool)
        self._distractor_events = []
        on = (self.p_distractor_trial > 0.0 and self.distractor_strength != 0.0
              and self.max_distractors > 0 and self.rng.rand() < self.p_distractor_trial)
        onsets = self.distractor_onsets(self.start_ind["delay"], t_target0, dur_steps) if on else []

        for d0 in onsets:
            d1 = min(d0 + dur_steps, T)
            d_pos, d_theta = self.distractor_position(cue_theta)
            trial["distractor_positions"].append((float(d_pos[0]), float(d_pos[1])))
            trial["distractor_thetas"].append(float(d_theta))
            self.write_distractor(d0, d1, d_pos)

            fa0, fa1 = int(np.clip(d0 + rt0, 0, T)), int(np.clip(d0 + rt1, 0, T))
            if fa1 > fa0:
                self._fa_window[fa0:fa1] = True
            self._distractor_events.append({"onset_step": int(d0), "offset_step": int(d1),
                                            "fa_start": int(fa0), "fa_end": int(fa1)})

        trial["has_distractors"] = bool(onsets)
        trial["n_distractors"] = len(onsets)
        return trial

    def _step(self, action):
        # neurogym takes one step inside reset(); training and analysis read ob/gt directly
        t = self.t_ind
        released = action == 1
        terminated = released or t == self.ob.shape[0] - 1
        return self.ob_now, 0.0, terminated, False, {"gt": int(self.gt_now)}


class SameLocation(CuedDetection):
    """Cue and target share one position drawn uniformly over the disk, and distractors are drawn
    from the same disk, so location is a continuous variable the network has to hold."""

    def disk_point(self):
        r = self.target_radius * math.sqrt(self.rng.rand())
        theta = 2.0 * math.pi * self.rng.rand()
        return (r * math.cos(theta), r * math.sin(theta)), theta

    def positions(self):
        pos, theta = self.disk_point()
        return pos, pos, theta, self.quadrant(theta)

    def distractor_position(self, cue_theta):
        return self.disk_point()


class Sector(SameLocation):
    """The cue (and the target) confined to an angular sector of the disk; distractors anywhere,
    as on the task. Reading a network on two sectors compares positions inside one network."""

    def __init__(self, *args, centre=0.0, half_width=math.pi / 4, **kwargs):
        self.centre, self.half_width = centre, half_width
        super().__init__(*args, **kwargs)

    def positions(self):
        r = self.target_radius * math.sqrt(self.rng.rand())
        theta = self.centre + self.half_width * (2.0 * self.rng.rand() - 1.0)
        pos = (r * math.cos(theta), r * math.sin(theta))
        return pos, pos, theta, self.quadrant(theta)


class PatchStimulus(SameLocation):
    """Stimuli are blurred patches: the written position is re-drawn around the true centre every
    T_avg ms, so a single frame is uninformative and the location has to be averaged over the
    stimulus window."""

    def __init__(self, *args, r_patch=0.25, patch_jitter=0.20, T_avg=20.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.r_patch = float(r_patch)
        self.patch_jitter = float(patch_jitter)
        self.T_avg = float(T_avg)

    def patch_offset(self):
        dx = self.rng.normal(0.0, self.patch_jitter)
        dy = self.rng.normal(0.0, self.patch_jitter)
        r = math.hypot(dx, dy)
        if r > self.r_patch and r > 0.0:                 # scale down rather than reject
            dx *= self.r_patch / r
            dy *= self.r_patch / r
        return dx, dy

    def write_patch(self, t0, t1, sx, sy, strength):
        if strength == 0.0:
            return
        hold = max(1, int(round(self.T_avg / self.dt)))
        t = t0
        while t < t1:
            te = min(t + hold, t1)
            dx, dy = self.patch_offset()
            self.ob[t:te, 4] = sx + dx
            self.ob[t:te, 5] = sy + dy
            self.ob[t:te, 6] = strength
            t = te

    def write_target(self, target_xy):
        self.write_patch(self.start_ind["target"], self.end_ind["target"],
                         target_xy[0], target_xy[1], self.target_strength)

    def write_distractor(self, t0, t1, d_xy):
        self.write_patch(t0, t1, d_xy[0], d_xy[1], self.distractor_strength)


class ContextReversal(SameLocation):
    """Go if the probe is at the cued location exclusive-or the context is B, with context delivered
    as a pulse on the fixation channel during the cue epoch only.

    A linear read-out of "probe at cued location" and "context" cannot compute their exclusive-or,
    so a network that has only the single-context solution is at chance by construction. The pulse
    is transient because a tonic level lets the read-out linearise the conjunction at decision time.
    """

    def __init__(self, *args, context_levels=(0.5, 2.0), **kwargs):
        kwargs.setdefault("p_distractor_trial", 0.0)
        super().__init__(*args, **kwargs)
        self.context_levels = tuple(context_levels)

    def _new_trial(self, **kwargs):
        cue_pos, _, cue_theta, cue_loc = self.positions()
        context = int(self.rng.rand() < 0.5)
        is_match = bool(self.rng.rand() < 0.5)
        if is_match:
            probe_pos, probe_theta = cue_pos, cue_theta
        else:
            probe_theta = self.distractor_theta(cue_theta)
            r = self.target_radius * math.sqrt(self.rng.rand())
            probe_pos = (r * math.cos(probe_theta), r * math.sin(probe_theta))
        is_go = bool(is_match ^ (context == 1))

        ctoa = self.ctoa_ms()
        self.timing["delay"] = ctoa
        trial = {"cue_loc": cue_loc, "target_loc": cue_loc, "cue_pos": cue_pos,
                 "target_pos": probe_pos, "cue_theta": cue_theta, "target_theta": probe_theta,
                 "ctoa_ms": ctoa, "ctoa_bin": self.ctoa_bin(ctoa), "has_distractors": False,
                 "n_distractors": 0, "distractor_positions": [], "distractor_thetas": [],
                 "context": context, "is_match": is_match, "is_go": is_go}
        trial.update(kwargs)

        self.add_period(["fixation", "cue", "delay", "target", "post_target"])
        T = self.end_ind["post_target"]
        self.ob = np.zeros((T, 7), dtype=np.float32)
        self.gt = np.zeros(T, dtype=np.int64)
        self.ob[:, 0] = 1.0
        c0, c1 = self.start_ind["cue"], self.end_ind["cue"]
        self.ob[c0:c1, 0] = self.context_levels[context]
        self.write_cue(cue_pos)
        self.write_target(probe_pos)

        t_target0 = self.start_ind["target"]
        rt0 = int(round(self.rt_window[0] / self.dt))
        rt1 = int(round(self.rt_window[1] / self.dt))
        win_start = int(np.clip(t_target0 + rt0, 0, T))
        win_end = int(np.clip(t_target0 + rt1, 0, T))

        self._resp_window = np.zeros(T, dtype=bool)
        self._fa_window = np.zeros(T, dtype=bool)
        self._distractor_events = []
        if is_go and win_end > win_start:
            self._resp_window[win_start:win_end] = True
            self.gt[win_start:win_end] = 1
        elif win_end > win_start:                        # releasing to a no-go probe is a false alarm
            self._fa_window[win_start:win_end] = True
            self._distractor_events.append({"onset_step": int(t_target0),
                                            "offset_step": int(self.end_ind["target"]),
                                            "fa_start": int(win_start), "fa_end": int(win_end)})
        self._t_target0 = int(t_target0)
        return trial


def stage(k):
    """Pretraining curriculum: target only, then add the cue, then add distractors."""
    cue = 0.0 if k == 0 else 1.0
    p_distractor = 0.6 if k == 2 else 0.0
    dstr = 1.0 if k == 2 else 0.0
    return lambda: CuedDetection(dt=20, cue_strength=cue, p_distractor_trial=p_distractor,
                                 distractor_strength=dstr)


def sameloc(cue, tgt, dstr):
    """A cell of the difficulty grid: cue, target and distractor strengths."""
    return lambda: SameLocation(dt=20, cue_strength=cue, target_strength=tgt,
                                p_distractor_trial=0.6, distractor_strength=dstr)


def sector(centre, half_width, cell=C1):
    cue, tgt, dstr = cell
    return lambda: Sector(dt=20, cue_strength=cue, target_strength=tgt, p_distractor_trial=0.6,
                          distractor_strength=dstr, centre=centre, half_width=half_width)


def patch(cell=PATCH_CELL):
    return lambda: PatchStimulus(dt=20, cue_strength=cell["cue"], target_strength=cell["tgt"],
                                 p_distractor_trial=0.6, distractor_strength=cell["dstr"],
                                 r_patch=cell["r_patch"], patch_jitter=cell["jitter"],
                                 T_avg=cell["T_avg"])


def context():
    return lambda: ContextReversal(dt=20, cue_strength=1.0, target_strength=1.0)
