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

# 4b. Block 4 again at a weight where the positional term actually acts. At lam 0.5 the
#     positional error did not move (0.17 -> 0.17) because the policy-gradient term swamps it, so
#     block 4 compared reward with reward. Calibration on context net 0, 1500 updates: lam 5 moves
#     it slowly (0.21 -> 0.09), lam 20 clearly (0.18 -> 0.04) with behaviour intact, lam 50 moves it
#     and floors behaviour.
block4b() {
    for task in context c1; do for net in $NETS; do for rep in 0 1; do
        run "nf_${task}_reward_aux20_net${net}_r$rep" -m experiments.nf --task $task \
            --cond reward_aux --net $net --rep $rep --target cue --lam 20 --eval-n 10000
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

# 5b. The same contrast on ten networks, at the anchor that separated the learners. anchor 0.01
#     sat between the two and is left at its three networks; the anchor 0 level is read off the
#     published context curves at 4000 updates, so only the anchored arm needs running.
block5b() {
    for cond in control reward; do for net in 0 1 2 3 4 5 6 7 8 9; do
        run "nf_context_${cond}_anchor0.1_net$net" -m experiments.nf --task context             --cond $cond --net $net --anchor 0.1 --updates 4000
    done; done
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

# 7. The within-manifold target itself: the frozen position head is asked to read the same
#    positions further apart, with no task signal. kappa 1 is the same target at its present size
#    and is the control the gain is read against. Prediction: kappa 2 raises crossnobis where
#    kappa 1 does not, and if any representational target can pay for itself behaviourally it is
#    this one, because it asks for no direction the code does not already carry.
block7() {
    for kappa in 1 1.5 2; do
        run "nf_c1_expand_calib_k$kappa" -m experiments.nf --task c1 --cond expand --net 0             --lr 1e-4 --kappa $kappa --eval-n 2000 --eval-every 50 --updates 600             --out results/nf_c1_expand_calib_k${kappa}_net0.csv
    done
    wait
    for kappa in 1 2; do for net in $NETS; do
        run "nf_c1_expand_k${kappa}_net$net" -m experiments.nf --task c1 --cond expand             --net $net --lr 1e-4 --kappa $kappa --eval-n 10000 --panel --all-trials
    done; done
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

# 9. The projection criterion, one step at a time on the saved context checkpoints. Steps of a
#    fixed length in parameter space, over a ladder of lengths: scaled by a learning rate instead,
#    the task step is far longer than the positional one and overshoots, which is what made the
#    first attempt unreadable. The criterion holds if the realised change tracks the predicted one
#    at the short end and for both directions.
block9() {
    for len in 0.003 0.01 0.03 0.1; do for cond in theta0 control joint; do for net in $NETS; do
        run "grad_step_${cond}_L${len}_net$net" -m experiments.gradients --cond $cond             --net $net --rep 0 --step $len             --out results/grad_step_context_${cond}_L${len}_net${net}_r0.csv
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
#     the local rule the positional term helps more than under back-propagation, because it reduces
#     the variance of a noise-based gradient estimate rather than diverting a budget. The learning
#     rate and the exploration scale are calibrated on one network first; 11 runs the calibration,
#     11c the c1 contrast and 11x the context one.
block11() {
    for lr in 3e-5 1e-4 3e-4; do for explore in 1 2; do
        run "nf_c1_reward_local_lr${lr}_x${explore}" -m experiments.nf --task c1 --cond reward             --rule local --lr $lr --explore $explore --net 0 --eval-n 2000 --updates 1200             --out results/nf_c1_reward_local_calib_lr${lr}_x${explore}_net0.csv
    done; done
    wait
}

LOCAL_LR=${LOCAL_LR:-1e-4}
LOCAL_EXPLORE=${LOCAL_EXPLORE:-1}

# 11a. The weight of the positional term under the local rule, with its manipulation check. The
#      term enters as a standardised reward component, so its weight is already on the scale of the
#      outcome reward and 0.5 should act -- but the rule is weak, and block 4 is the reminder that a
#      term which does not move its own quantity tests nothing. Read the positional error in the
#      logs: it has to fall against the reward-only run before the contrast means anything.
block11a() {
    for lam in 0.5 2; do
        run "nf_c1_reward_aux_local_lam$lam" -m experiments.nf --task c1 --cond reward_aux \
            --rule local --lr $LOCAL_LR --explore $LOCAL_EXPLORE --lam $lam --target cue \
            --net 0 --eval-n 2000 --out results/nf_c1_reward_aux_local_calib_lam${lam}_net0.csv
    done
    run "nf_c1_reward_local_lam0" -m experiments.nf --task c1 --cond reward --rule local \
        --lr $LOCAL_LR --explore $LOCAL_EXPLORE --target cue --net 0 --eval-n 2000 \
        --out results/nf_c1_reward_local_calib_lam0_net0.csv
    wait
}

block11c() {
    for cond in reward reward_aux; do for net in $NETS; do for rep in 0 1; do
        run "nf_c1_${cond}_local_net${net}_r$rep" -m experiments.nf --task c1 --cond $cond             --rule local --lr $LOCAL_LR --explore $LOCAL_EXPLORE --net $net --rep $rep             --target cue --lam 0.5 --eval-n 10000
    done; done; done
    wait
}

# one repeat on the context task: eight thousand updates a run is the budget that sets how much of
# this fits in a day, and the c1 contrast carries the repeats
block11x() {
    for cond in reward reward_aux; do for net in $NETS; do
        run "nf_context_${cond}_local_net${net}_r0" -m experiments.nf --task context --cond $cond             --rule local --lr $LOCAL_LR --explore $LOCAL_EXPLORE --net $net --rep 0             --target cue --lam 0.5 --eval-n 10000
    done; done
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

# 13. The one caveat on block 1: reward runs at lr 1e-4 and the control at 3e-4. The same control
#     at the reward's learning rate, so the contrast differs only in the signal.
block13() {
    for net in $NETS; do for rep in 0 1; do
        run "nf_c1_control_lr1e-4_net${net}_r$rep" -m experiments.nf --task c1 --cond control \
            --net $net --rep $rep --lr 1e-4 --eval-n 10000
    done; done
    wait
}

for b in "$@"; do
    echo "=== block $b started $(date)"
    "block$b"
    echo "=== block $b finished $(date)"
done
