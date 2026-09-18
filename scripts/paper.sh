#!/usr/bin/env bash
# The runs that turn the campaign into a paper, one function per block. Each block names what it
# tests and what would count as the answer. Pass the block numbers to run, in order:
#
#     scripts/paper.sh 1 10 2 3 4 6 5 9 7 8
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/pool.sh

NETS="0 1 2 3 4"

# 1. The most promising positive lead, measured properly: reward against control on c1 at the
#    10k evaluation, two repeats. The published +0.19 sat at p = 0.24 on 1k evaluations.
block1() {
    for net in $NETS; do for rep in 0 1; do
        run "nf_c1_reward_10k_net${net}_r$rep" -m experiments.nf --task c1 --cond reward \
            --net $net --rep $rep --eval-n 10000
    done; done
    wait
}

# 2. The representation table without the correct-trial selection: every trial in the pool.
#    Repeats the context representation pass on the saved checkpoints. The answer changes if the
#    crossnobis differences between conditions shrink toward zero.
block2() {
    for cond in control joint reward position; do for net in $NETS; do for rep in 0 1; do
        run "repr_all_${cond}_net${net}_r$rep" -m experiments.representation --cond $cond \
            --net $net --rep $rep --out results/repr_all_context_${cond}_net${net}_r$rep.csv
    done; done; done
    for net in $NETS; do
        run "repr_all_theta0_net$net" -m experiments.representation --cond theta0 --net $net \
            --out results/repr_all_context_theta0_net${net}_r0.csv
    done
    wait
}

# 3. The positional signals on the context task aimed at the cue rather than the probe.
#    Prediction written before the runs: the joint deficit shrinks but does not reverse.
block3() {
    for cond in position joint; do for net in $NETS; do for rep in 0 1; do
        run "nf_context_${cond}_cue_net${net}_r$rep" -m experiments.nf --task context \
            --cond $cond --net $net --rep $rep --target cue
    done; done; done
    wait
}

# 4. The label-free regime: reward alone against reward plus the positional term. This is the
#    first place neurofeedback is expected to win. Prediction: reward_aux reaches the reward
#    level earlier; the level itself may not differ.
block4() {
    for task in context c1; do for net in $NETS; do for rep in 0 1; do
        run "nf_${task}_reward_aux_net${net}_r$rep" -m experiments.nf --task $task \
            --cond reward_aux --net $net --rep $rep --target cue --lam 0.5 --eval-n 10000
    done; done; done
    wait
}

# 5. Lazy against rich: the same context runs with the recurrent weights anchored. Prediction
#    from the BCI models: outside-manifold learning (the control on this task) slows or breaks
#    while within-manifold relearning survives.
block5() {
    for anchor in 0.01 0.1; do for cond in control reward; do for net in 0 1 2; do
        run "nf_context_${cond}_anchor${anchor}_net$net" -m experiments.nf --task context \
            --cond $cond --net $net --anchor $anchor --updates 4000
    done; done; done
    wait
}

# 6a. Calibrate the perturbation angle. On net 0 the immediate d' is 1.15 / 0.63 / 0.06 for a
#     within-manifold rotation of 15 / 30 / 45 degrees (intact 1.24), and 1.21 / 1.27 / 1.36 for
#     an outside-manifold rotation of the same angles: the complement carries no variance, the
#     decoder still reads cos(angle) of the in-manifold signal, and the recalibrated threshold
#     absorbs the scaling. An outside perturbation only bites at 90 degrees, when nothing of the
#     old axis is left. Re-run on any new network.
block6a() {
    for angle in 30 45 60 90; do for kind in within outside; do
        run "manifold_calib_${kind}_a$angle" -m experiments.manifold --net 0 --kind $kind \
            --angle $angle --updates 0 --out results/manifold_calib_${kind}_a${angle}_net0.csv
    done; done
    wait
}

# 6b. Within against outside manifold from a matched floor (within 45, outside 90), relearned by
#     reward alone, then the outside case as an incremental ladder. Prediction: within recovers,
#     outside does not at this budget, the ladder does. The supervised reference says whether a
#     failure to recover is the geometry or the weakness of the reward learner.
block6() {
    WITHIN=45
    OUTSIDE=90
    for net in $NETS; do for c in 0 1 2; do
        run "manifold_within_net${net}_c$c" -m experiments.manifold --net $net --kind within \
            --angle $WITHIN --candidate $c
        run "manifold_outside_net${net}_c$c" -m experiments.manifold --net $net --kind outside \
            --angle $OUTSIDE --candidate $c
        run "manifold_ladder_net${net}_c$c" -m experiments.manifold --net $net --kind outside \
            --angle $OUTSIDE --candidate $c --steps 8
    done; done
    wait
    for net in 0 1 2; do
        run "manifold_within_control_net$net" -m experiments.manifold --net $net --kind within \
            --angle $WITHIN --candidate 0 --cond control
        run "manifold_outside_control_net$net" -m experiments.manifold --net $net --kind outside \
            --angle $OUTSIDE --candidate 0 --cond control
    done
    wait
}

# 7. The within-manifold target itself: radial expansion of the delay manifold with no task
#    signal. Prediction: crossnobis rises and, unlike the positional signal, coupling rises too.
#    Calibrate on one network first: at the default learning rate the stretch costs behaviour
#    within a few updates, so start at 1e-4 and kappa 1.2 and read the curve.
block7() {
    run "nf_c1_expand_calib" -m experiments.nf --task c1 --cond expand --net 0 --lr 1e-4 \
        --kappa 1.2 --lam 1.0 --eval-n 2000 --eval-every 50 --updates 600 \
        --out results/nf_c1_expand_calib_net0.csv
    wait
    for net in $NETS; do
        run "nf_c1_expand_net$net" -m experiments.nf --task c1 --cond expand --net $net \
            --lr 1e-4 --kappa 1.2 --lam 1.0 --eval-n 10000 --panel --all-trials
    done
    wait
}

# 8. Co-adaptation: the decision read-out takes twenty supervised steps every 300 updates, with
#    and without the reward signal in between. Prediction: neurofeedback plus a slowly adapting
#    decoder beats the adapting decoder alone in updates-to-level.
block8() {
    for cond in control reward; do for net in $NETS; do
        run "nf_c1_${cond}_coadapt_net$net" -m experiments.nf --task c1 --cond $cond --net $net \
            --adapt-decoder 300 --eval-n 10000 \
            --out results/nf_c1_${cond}_coadapt_net${net}_r0.csv
    done; done
    wait
}

# 9. The projection criterion, tested one step at a time on the saved context checkpoints. Two
#    step sizes: at 1e-3 the realised change sits inside the batch noise of the held-out loss.
block9() {
    for eta in 1e-3 1e-2; do for cond in theta0 control joint; do for net in $NETS; do
        run "grad_step_${cond}_e${eta}_net$net" -m experiments.gradients --cond $cond \
            --net $net --rep 0 --step $eta \
            --out results/grad_step_context_${cond}_e${eta}_net${net}_r0.csv
    done; done; done
    wait
}

# 10. The resolutions of the instrument on the networks the experiments actually run on.
block10() {
    for rep in 0 1 2 3 4 5 6 7 8 9; do
        run "resolution_c1_r$rep" -m experiments.resolutions --task c1 --rep $rep
    done
    wait
}

# 11. The learning rule as a variable: reward alone against reward plus the positional term,
#     learned by reward-modulated node perturbation instead of back-propagation. Prediction: under
#     the local rule the positional term helps more than under BPTT, because it reduces the
#     variance of a noise-based gradient estimate rather than diverting a budget. The learning
#     rate and the exploration scale are calibrated on one network first.
block11() {
    for lr in 3e-5 1e-4 3e-4; do for explore in 1 2; do
        run "nf_c1_reward_local_lr${lr}_x${explore}" -m experiments.nf --task c1 --cond reward \
            --rule local --lr $lr --explore $explore --net 0 --eval-n 2000 --updates 1200 \
            --out results/nf_c1_reward_local_calib_lr${lr}_x${explore}_net0.csv
    done; done
    wait
    LR=1e-4       # replace with the calibrated pair
    EXPLORE=1
    for task in c1 context; do for cond in reward reward_aux; do for net in $NETS; do for rep in 0 1; do
        run "nf_${task}_${cond}_local_net${net}_r$rep" -m experiments.nf --task $task --cond $cond \
            --rule local --lr $LR --explore $EXPLORE --net $net --rep $rep --target cue \
            --lam 0.5 --eval-n 10000
    done; done; done; done
    wait
}

# 12. The closed-loop ladder: the outside-manifold read-out advances a rung only while behaviour
#     on the next rung holds. Prediction: it reaches the target in fewer updates than the fixed
#     eight-rung ladder, or reaches it where the fixed ladder does not.
block12() {
    for net in $NETS; do for c in 0 1 2; do
        run "manifold_auto_net${net}_c$c" -m experiments.manifold --net $net --kind outside \
            --angle 90 --candidate $c --auto --updates 6000
    done; done
    wait
}

for b in "$@"; do
    echo "=== block $b started $(date)"
    "block$b"
    echo "=== block $b finished $(date)"
done
