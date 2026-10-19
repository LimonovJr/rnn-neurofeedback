# Neurofeedback through a fixed decoder in a recurrent network

Code for a modelling study of what feedback signals change when they are delivered through a decoder
that stays fixed while the network learns. The network is a leaky rate network of 180 units with
Dale's law and topographic connectivity, pretrained on a cued detection task (c1). Behaviour goes
through a frozen decision read-out. The positional feedback signals are computed by a second frozen
read-out, of the cued position. Feedback trains the recurrent weights only, and each signal is
compared with ordinary supervised training through the same decoder at the same budget.

Two tasks sit at the two ends of a headroom axis. Headroom is what training all weights reaches
from a frozen state minus what training the read-out alone reaches from the same state (C_B − C_A,
in d′). It is 0.27 on c1, where the decoder is the bottleneck, and 1.92 on a context-reversal task
whose go rule is an exclusive-or that the pretrained dynamics do not compute.

## Findings

Every number below is rebuilt from the per-run CSVs by `experiments.aggregate`. Networks 0–4 were
used to find the effects and networks 10–24 to test them, with the predictions written before the
runs.

- Where the decoder limits (c1), reward on the trial outcome and supervision end on one ROC at
  different operating points. Reward is ahead by 0.35 d′ at the decoder's own threshold and level
  with supervision at the best threshold and in the area under the ROC. Each learner ends near the
  payoff optimum of the conditions it learned under.
- Where the dynamics limit (context), the difference survives the threshold: supervision ends 0.14
  higher in the area under the ROC. A read-out with one hidden layer on the frozen dynamics closes
  about a third of the context headroom.
- Targets aimed at the representation (a magnified, shuffled or sector-confined positional code, a
  separation axis, a go/no-go template) change the decoded code as asked and lower
  discriminability, at every dose from one update to the full budget. About two thirds of the loss
  is specific to the target, and a read-out refitted on the end state, linear or not, returns less
  than half of it.
- A decoder fitted before training can report the wrong sign. After task training the positional
  code was more separable and discriminability higher, while the accuracy of the decoder fitted at
  the start fell from 0.56 to 0.21.
- With the decision read-out rotated out of the intrinsic manifold, a schedule that advances the
  rotation only while behaviour holds left 1 of 45 reward-learner runs at the floor, against 30 of
  45 for a direct jump.

## Layout

```
src/            the network, the tasks, trial collection, training, the threshold sweep, the
                headroom instrument, the feedback signals, representation measures, statistics
experiments/    one entry point per measurement, the aggregation pass and the figures
scripts/        launchers that spread runs across CPU processes
checkpoints/    trained weights (not versioned; produced by experiments.pretrain)
results/        one CSV per run (not versioned; tables are rebuilt from them)
```

## Installation

```bash
conda create -n nf-env python=3.10 -y && conda activate nf-env
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "neurogym @ git+https://github.com/neurogym/neurogym.git"
pip install -r requirements.txt
```

Training is CPU-bound. One update on a 250-step unroll takes about 0.7 s and a GPU is no faster at
this size, so the launchers run many processes with two threads each.

## Quickstart

```bash
python -m experiments.pretrain --net 0            # three-stage curriculum, saves each stage
python -m experiments.fit_head --task c1 --net 0  # position head, fitted on frozen dynamics
python -m experiments.headroom --setting c1 --arm A --init incumbent --net 0
python -m experiments.aggregate headroom          # rebuild the table from whatever has been run
python -m experiments.figures                     # figures, into results/figures
```

`scripts/smoke.sh` runs the early entry points on budgets small enough to finish in minutes.

## Entry points

Each entry point writes one CSV per run, with the trajectory and the individual re-evaluations
behind any average, and `python -m experiments.aggregate <table>` rebuilds a table from them.

The instrument:

| Entry point | What it measures |
|---|---|
| `pretrain` | the three-stage curriculum for one network |
| `fit_head` | the position read-out, fitted on frozen dynamics |
| `headroom` | one cell of C_A, C_B or C_A+ (setting, arm, network, restart) |
| `positive_control` | the context task, where a linear read-out of the frozen dynamics cannot succeed |
| `resolutions`, `criterion_resolution` | what each measure can tell apart on repeated draws |
| `degradation`, `patch_integration` | an input-noise ladder; what averaging over the stimulus is worth |
| `competence` | whether the code forms before behaviour leaves the floor |

Training runs:

| Entry point | What it trains |
|---|---|
| `nf` | one feedback run (task, condition, network, repeat), with knobs for the target, a weight anchor, decoder adaptation, a local learning rule, payoff and threshold tracking |
| `dose` | every signal from the start, read along its dose on seeded trials shared by the states of a network; `--save` keeps the end state |
| `closed_loop` | supervised training under difficulty set by the decoded state, or yoked to another run |
| `manifold` | the decision read-out rotated within or outside the intrinsic manifold and relearned by reward, directly or by a ladder |
| `practice` | the decision read-out alone retrained on the end state of a feedback run |

Analyses of saved states, without training:

| Entry point | What it reads |
|---|---|
| `stepwise`, `calibration`, `temperature`, `context_optimum` | operating points, payoff optima and the release margin |
| `access`, `neutral`, `neutral_routes` | whether a loss after a representational target is criterion, access or computation |
| `location`, `relocated`, `focus` | how much behaviour relies on location, and where the code went |
| `induction`, `rt_signatures`, `psychometric`, `cue_strength` | induction against behaviour, reaction times, other target and cue strengths |
| `representation`, `repertoire`, `template_transfer` | the code on the context task, its reuse of old patterns, templates across networks |
| `response` | whether a signal reaches the decision through the decoder's coupling to it |
| `gradients`, `components`, `criterion_share`, `snr` | gradient geometry and the criterion and separation parts of one step |
| `wide_sweep` | the context comparison on a wider grid of thresholds |

`scripts/pretrain.sh`, `headroom.sh`, `instrument.sh`, `nf_c1.sh` and `nf_context.sh` run the first
campaign, a few hundred process-hours. `scripts/paper.sh` holds every later block as a function
that states its question and the prediction it was written against:

```bash
PY=.venv/bin/python JOBS=10 bash scripts/paper.sh 63 66
```

## Reproducibility

The trial stream of a training run is not seeded, so two runs of the same command see different
trials and the design replicates runs rather than re-deriving them. Network initialisation, read-out
restarts and the nonlinear head are seeded. The analyses of saved states read every state of a
network on the same seeded trials (`trials.seeded`).

The base network and the detection task come from
[leaky-rnn-dale](https://github.com/LimonovJr/leaky-rnn-dale), where the same architecture is
analysed without feedback.

## License

MIT, see [LICENSE](LICENSE).
