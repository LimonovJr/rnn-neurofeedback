#!/usr/bin/env bash
# The runs that turn the campaign into a paper, in the order they should be decided on. Each block
# names what it tests and what would count as the answer. Nothing here has been run yet.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/pool.sh

NETS="0 1 2 3 4"

# 1. The most promising positive lead, measured properly: reward against control on c1 at the
#    10k evaluation, two repeats. The published +0.19 sat at p = 0.24 on 1k evaluations.
for net in $NETS; do for rep in 0 1; do
    run "nf_c1_reward_10k_net${net}_r$rep" -m experiments.nf --task c1 --cond reward \
        --net $net --rep $rep --eval-n 10000
done; done
wait

# 2. The representation table without the correct-trial selection: every trial in the pool.
#    Repeats the context representation pass on the saved checkpoints. The answer changes if the
#    crossnobis differences between conditions shrink toward zero.
for cond in control joint reward position; do for net in $NETS; do for rep in 0 1; do
    run "repr_all_${cond}_net${net}_r$rep" -m experiments.representation --cond $cond \
        --net $net --rep $rep --out results/repr_all_context_${cond}_net${net}_r$rep.csv
done; done; done
wait

# 3. The positional signals on the context task aimed at the cue rather than the probe.
#    Prediction written before the runs: the joint deficit shrinks but does not reverse.
for cond in position joint; do for net in $NETS; do for rep in 0 1; do
    run "nf_context_${cond}_cue_net${net}_r$rep" -m experiments.nf --task context --cond $cond \
        --net $net --rep $rep --target cue
done; done; done
wait

# 4. The label-free regime: reward alone against reward plus the positional term. This is the
#    first place neurofeedback is expected to win. Prediction: reward_aux reaches the reward
#    level earlier; the level itself may not differ.
for task in context c1; do for net in $NETS; do for rep in 0 1; do
    run "nf_${task}_reward_aux_net${net}_r$rep" -m experiments.nf --task $task --cond reward_aux \
        --net $net --rep $rep --target cue --lam 0.5 --eval-n 10000
done; done; done
wait

# 5. Lazy against rich: the same context runs with the recurrent weights anchored. Prediction
#    from the BCI models: outside-manifold learning (the control on this task) slows or breaks
#    while within-manifold relearning survives.
for anchor in 0.01 0.1; do for cond in control reward; do for net in 0 1 2; do
    run "nf_context_${cond}_anchor${anchor}_net$net" -m experiments.nf --task context \
        --cond $cond --net $net --anchor $anchor --updates 4000
done; done; done
wait

# 6a. Calibrate the perturbation angle. On net 0 the immediate d' is 1.15 / 0.63 / 0.06 for a
#     within-manifold rotation of 15 / 30 / 45 degrees (intact 1.24), and 1.21 / 1.27 / 1.36 for
#     an outside-manifold rotation of the same angles: the complement carries no variance, the
#     decoder still reads cos(angle) of the in-manifold signal, and the recalibrated threshold
#     absorbs the scaling. An outside perturbation only bites at 90 degrees, when nothing of the
#     old axis is left. The two kinds are therefore matched at the floor -- within 45, outside 90
#     -- and the ladder for outside climbs the angle. Re-run this block on any new network.
for angle in 30 45 60 90; do for kind in within outside; do
    run "manifold_calib_${kind}_a$angle" -m experiments.manifold --net 0 --kind $kind         --angle $angle --updates 0 --out results/manifold_calib_${kind}_a${angle}_net0.csv
done; done
wait

# 6b. Within against outside manifold from a matched floor, relearned by reward alone, then the
#     outside case as an incremental ladder. Prediction: within recovers, outside does not at
#     this budget, the ladder does. The supervised reference says whether a failure to recover is
#     the geometry or the weakness of the reward learner.
WITHIN=45
OUTSIDE=90
for net in $NETS; do for c in 0 1 2; do
    run "manifold_within_net${net}_c$c" -m experiments.manifold --net $net --kind within         --angle $WITHIN --candidate $c
    run "manifold_outside_net${net}_c$c" -m experiments.manifold --net $net --kind outside         --angle $OUTSIDE --candidate $c
    run "manifold_ladder_net${net}_c$c" -m experiments.manifold --net $net --kind outside         --angle $OUTSIDE --candidate $c --steps 8
done; done
wait
for kind_angle in "within $WITHIN" "outside $OUTSIDE"; do set -- $kind_angle; for net in 0 1 2; do
    run "manifold_${1}_control_net$net" -m experiments.manifold --net $net --kind $1         --angle $2 --candidate 0 --cond control
done; done
wait

# 7. The within-manifold target itself: radial expansion of the delay manifold with no task
#    signal. Prediction: crossnobis rises and, unlike the positional signal, coupling rises too.
#    Calibrate on one network first: at the default learning rate the stretch costs behaviour
#    within a few updates, so start at 1e-4 and kappa 1.2 and read the curve.
run "nf_c1_expand_calib" -m experiments.nf --task c1 --cond expand --net 0 --lr 1e-4 \
    --kappa 1.2 --lam 1.0 --eval-n 2000 --eval-every 50 --updates 600 \
    --out results/nf_c1_expand_calib_net0.csv
wait
for net in $NETS; do
    run "nf_c1_expand_net$net" -m experiments.nf --task c1 --cond expand --net $net \
        --lr 1e-4 --kappa 1.2 --lam 1.0 --eval-n 10000 --panel --all-trials
done
wait

# 8. Co-adaptation: the decision read-out takes twenty supervised steps every 300 updates, with
#    and without the reward signal in between. Prediction: neurofeedback plus a slowly adapting
#    decoder beats the adapting decoder alone in updates-to-level.
for cond in control reward; do for net in $NETS; do
    run "nf_c1_${cond}_coadapt_net$net" -m experiments.nf --task c1 --cond $cond --net $net \
        --adapt-decoder 300 --eval-n 10000 --out results/nf_c1_${cond}_coadapt_net${net}_r0.csv
done; done
wait

# 9. The projection criterion, tested one step at a time on the saved context checkpoints.
for cond in theta0 control joint; do for net in $NETS; do
    run "grad_step_${cond}_net$net" -m experiments.gradients --cond $cond --net $net --rep 0 \
        --step 1e-3 --out results/grad_step_context_${cond}_net${net}_r0.csv
done; done
wait

# 10. The resolutions of the instrument on the networks the experiments actually run on.
for rep in 0 1 2 3 4 5 6 7 8 9; do
    run "resolution_c1_r$rep" -m experiments.resolutions --task c1 --rep $rep
done
wait
