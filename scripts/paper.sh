#!/usr/bin/env bash
# The runs that turn the campaign into a paper, one function per block. Each block names what it
# tests and what would count as the answer. Pass the block numbers to run, in order:
#
#     scripts/paper.sh 1 10 2 3 4 6 5 9 7 8
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/pool.sh

NETS="0 1 2 3 4"
NEW_NETS="5 6 7 8 9"          # trained after the campaign, for the confirmatory repeat

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

# 7b. The matched null for block 7: the same loss, at the same gain, asking each trial for another
#     trial's position. It reaches the weights the same way and the code it asks for carries
#     nothing, so it says whether the behavioural cost belongs to improving the representation or
#     to training against a representational loss at all.
block7b() {
    for net in $NETS; do
        run "nf_c1_expand_shuf_net$net" -m experiments.nf --task c1 --cond expand --net $net \
            --lr 1e-4 --kappa 1 --shuffle --eval-n 10000 --panel --all-trials
    done
    wait
}

# 7c. The arms of block 7 differ in how far they drag the recurrent weights (0.027 shuffled,
#     0.030 at kappa 1, 0.035 at kappa 2), and across those fifteen runs the behavioural loss
#     follows the weight movement better than it follows the change in the code -- with the
#     movement held fixed, the code explains nothing. This matches the movement instead of the
#     learning rate: the magnified target at a rate that should move the weights as far as the
#     shuffled one does, and the shuffled target at a rate that should move them as far as the
#     magnified one does. Prediction, if the movement is what matters: behaviour falls by the same
#     amount inside each matched pair, although the code moves in opposite directions.
block7c() {
    for net in $NETS; do
        run "nf_c1_expand_k2_slow_net$net" -m experiments.nf --task c1 --cond expand --net $net \
            --lr 7.5e-5 --kappa 2 --eval-n 10000 --panel --all-trials
        run "nf_c1_expand_shuf_fast_net$net" -m experiments.nf --task c1 --cond expand --net $net \
            --lr 1.3e-4 --kappa 1 --shuffle --eval-n 10000 --panel --all-trials
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

# 13b. Block 13 again, one repeat, now that runs record how far they move the recurrent weights.
#      Block 7 showed that a signal can cost behaviour in proportion to that movement alone, so the
#      reward advantage has to be read against it: does reward win by moving the weights less, or
#      by moving them as far in directions the frozen decoder credits?
block13b() {
    for cond in reward control; do for net in $NETS; do
        run "nf_c1_${cond}_move_net$net" -m experiments.nf --task c1 --cond $cond --net $net \
            --rep 0 --lr 1e-4 --eval-n 10000 --out results/nf_c1_${cond}_move_net${net}_r0.csv
    done; done
    wait
}

# 14. The positional signal with a gate: joint, with the positional gradient admitted only on the
#     updates where it points the same way as the task gradient (Du et al. 2018). Block 9 showed
#     the positional gradient helps at the start and stops helping once trained on; a gate that
#     follows the cosine should take what help there is and refuse the rest. Prediction: gated is
#     never below the control, where joint is -0.38 on context, and the gate closes more often as
#     training goes on. Compared against the published control and joint runs on the same seeds.
block14() {
    for net in $NETS; do for rep in 0 1; do
        run "nf_c1_gated_net${net}_r$rep" -m experiments.nf --task c1 --cond gated --net $net \
            --rep $rep --eval-n 10000
    done; done
    for net in $NETS; do for rep in 0 1; do
        run "nf_context_gated_cue_net${net}_r$rep" -m experiments.nf --task context --cond gated \
            --net $net --rep $rep --target cue
    done; done
    wait
}

# 15. Why reward wins: on-policy reinforcement is expected to reach the solution closest to where
#     it started, and so to disturb least what the network could already do (Shenfeld et al. 2025).
#     Reward and control at the same rate, with the policy shift on c1 and the pretraining task and
#     the patch cell scored before and after through the same read-out. Prediction: reward shifts
#     the c1 policy less per unit of behaviour gained and leaves the other two at least as good.
block15() {
    for cond in reward control; do for net in $NETS; do for rep in 0 1; do
        run "nf_c1_${cond}_retention_net${net}_r$rep" -m experiments.nf --task c1 --cond $cond \
            --net $net --rep $rep --lr 1e-4 --eval-n 10000 --retention
    done; done; done
    wait
}

# 16. The local rule with its exploration confined to the activity manifold (Kang, Richards and
#     Sabatini 2026): ten directions carry nearly all of the activity variance, so an isotropic
#     perturbation of 180 units spends almost all of its variance where the dynamics never go.
#     Calibration on net 0 first, with the positional term in half the runs as the manipulation
#     check: the positional error has to fall before the contrast of block 11 can be run again.
block16() {
    for lr in 1e-4 1e-3; do for explore in 1 3; do for cond in reward reward_aux; do
        run "nf_c1_${cond}_npk10_lr${lr}_x$explore" -m experiments.nf --task c1 --cond $cond \
            --rule local --explore-k 10 --lr $lr --explore $explore --lam 2 --target cue \
            --net 0 --eval-n 2000 \
            --out results/nf_c1_${cond}_npk10_calib_lr${lr}_x${explore}_net0.csv
    done; done; done
    wait
}

# 17. How much of the reward advantage is the release threshold. Six reward updates already lift
#     the pretraining task from 1.24 to 1.98, which looks like a shift of the shared criterion, and
#     re-fitting the read-out removes most of the advantage (block 8). At the start and at the end
#     of each run the threshold the frozen decoder would do best with is found on one draw and
#     scored on another; nothing is trained. If reward only moves the criterion, the two conditions
#     meet once each has its best threshold; if it builds something the threshold cannot, reward
#     stays ahead.
block17() {
    for cond in reward control; do for net in $NETS; do for rep in 0 1; do
        run "nf_c1_${cond}_criterion_net${net}_r$rep" -m experiments.nf --task c1 --cond $cond \
            --net $net --rep $rep --lr 1e-4 --eval-n 10000 --criterion
    done; done; done
    wait
}

# 18. Speed or level. Pooled over every repeat, reward is flat from about 300 updates (+0.02 per
#     thousand over the second half) while the control is still climbing at 1200 (+0.11 per
#     thousand), so the 1200-update contrast cannot say whether reward reaches a higher level or
#     only reaches it sooner. The same contrast at 4000 updates, with the control also at the rate
#     it does best with (3e-4). Prediction registered before the runs: the gap narrows; whether it
#     closes is the question.
block18() {
    for net in $NETS; do
        run "nf_c1_reward_u4000_net$net" -m experiments.nf --task c1 --cond reward --net $net             --lr 1e-4 --updates 4000 --eval-every 250 --eval-n 10000
        run "nf_c1_control_u4000_lr1e-4_net$net" -m experiments.nf --task c1 --cond control             --net $net --lr 1e-4 --updates 4000 --eval-every 250 --eval-n 10000
        run "nf_c1_control_u4000_lr3e-4_net$net" -m experiments.nf --task c1 --cond control             --net $net --lr 3e-4 --updates 4000 --eval-every 250 --eval-n 10000
    done
    wait
}

# 19. Internal noise: the one regime the supervisor proposed that was never run, because the
#     recurrent-noise ladder was a no-op. A pre-test without training found that as the noise grows
#     behaviour falls to the floor (d' 0.96 at x1, 0.21 at x2, -0.26 at x3) and the positional
#     gradient lines up with the task gradient (cosine +0.06, +0.18, +0.22). At the floor the
#     reward learner has no gradient; the positional term has one and it points the right way.
#     Primary: reward plus positional term against reward alone, label-free. Secondary: joint
#     against the supervised control. Prediction: reward_aux > reward at both levels; joint >= control.
block19() {
    for scale in 2 3; do for net in $NETS; do
        run "nf_c1_reward_rn${scale}_net$net" -m experiments.nf --task c1 --cond reward \
            --net $net --rec-noise-scale $scale --updates 2000 --eval-every 200 --eval-n 10000
        run "nf_c1_reward_aux_rn${scale}_net$net" -m experiments.nf --task c1 --cond reward_aux \
            --net $net --rec-noise-scale $scale --lam 20 --target cue --updates 2000 \
            --eval-every 200 --eval-n 10000
        run "nf_c1_control_rn${scale}_net$net" -m experiments.nf --task c1 --cond control \
            --net $net --rec-noise-scale $scale --updates 2000 --eval-every 200 --eval-n 10000
        run "nf_c1_joint_rn${scale}_net$net" -m experiments.nf --task c1 --cond joint \
            --net $net --rec-noise-scale $scale --target cue --updates 2000 --eval-every 200 \
            --eval-n 10000
    done; done
    wait
}

# 20. The confirmatory repeat of the four primary contrasts on networks 5-9, with the contrasts,
#     the direction of each and the Holm correction fixed before any of these runs. 20a: the two
#     c1 contrasts; 20b: the ladder; 20c: the context contrast.
block20a() {
    for net in $NEW_NETS; do for rep in 0 1; do
        run "nf_c1_reward_10k_net${net}_r$rep" -m experiments.nf --task c1 --cond reward \
            --net $net --rep $rep --eval-n 10000
        run "nf_c1_control_lr1e-4_net${net}_r$rep" -m experiments.nf --task c1 --cond control \
            --net $net --rep $rep --lr 1e-4 --eval-n 10000
    done; done
    for net in $NEW_NETS; do
        run "nf_c1_expand_k2_slow_net$net" -m experiments.nf --task c1 --cond expand --net $net \
            --lr 7.5e-5 --kappa 2 --eval-n 10000 --panel --all-trials
        run "nf_c1_expand_shuf_fast_net$net" -m experiments.nf --task c1 --cond expand --net $net \
            --lr 1.3e-4 --kappa 1 --shuffle --eval-n 10000 --panel --all-trials
        run "nf_c1_expand_k1_net$net" -m experiments.nf --task c1 --cond expand --net $net \
            --lr 1e-4 --kappa 1 --eval-n 10000 --panel --all-trials
    done
    wait
}

block20b() {
    for net in $NEW_NETS; do for c in 0 1 2; do
        run "manifold_outside_net${net}_c$c" -m experiments.manifold --net $net --kind outside \
            --angle 90 --candidate $c
        run "manifold_auto_net${net}_c$c" -m experiments.manifold --net $net --kind outside \
            --angle 90 --candidate $c --auto --updates 6000
    done; done
    wait
}

block20c() {
    for net in $NEW_NETS; do for rep in 0 1; do for cond in control reward; do
        run "nf_context_${cond}_net${net}_r$rep" -m experiments.nf --task context --cond $cond \
            --net $net --rep $rep --criterion
    done; done; done
    wait
}

# 21. The threshold, everywhere a behavioural contrast is claimed. On c1 the whole reward advantage
#     turned out to be the release threshold of the frozen decoder: with each condition at its
#     best threshold, reward - control is -0.006 (p = 0.84), and both raise discriminability by the
#     same amount. The same check for the representational targets on c1 (21a) and for the
#     context contrast the regime axis rests on (21b), networks 0-4.
block21a() {
    for net in $NETS; do
        run "nf_c1_expand_k2_slow_crit_net$net" -m experiments.nf --task c1 --cond expand \
            --net $net --lr 7.5e-5 --kappa 2 --eval-n 10000 --panel --all-trials --criterion
        run "nf_c1_expand_shuf_fast_crit_net$net" -m experiments.nf --task c1 --cond expand \
            --net $net --lr 1.3e-4 --kappa 1 --shuffle --eval-n 10000 --panel --all-trials --criterion
        run "nf_c1_expand_k1_crit_net$net" -m experiments.nf --task c1 --cond expand --net $net \
            --lr 1e-4 --kappa 1 --eval-n 10000 --panel --all-trials --criterion
        run "nf_c1_joint_crit_net$net" -m experiments.nf --task c1 --cond joint --net $net \
            --eval-n 10000 --criterion
    done
    wait
}

block21b() {
    for net in $NETS; do for cond in control reward joint; do
        run "nf_context_${cond}_crit_net$net" -m experiments.nf --task context --cond $cond \
            --net $net --target cue --criterion
    done; done
    wait
}

# 21c. The threshold under internal noise. Block 19 found reward far above the supervised control
#      at both noise levels (+0.88 at x2, +1.02 at x3), several times the intact c1 advantage that
#      block 17 traced entirely to the release threshold. Before that gap means anything it gets
#      the same check.
block21c() {
    for scale in 2 3; do for cond in reward control; do for net in $NETS; do
        run "nf_c1_${cond}_crit_rn${scale}_net$net" -m experiments.nf --task c1 --cond $cond \
            --net $net --rec-noise-scale $scale --updates 2000 --eval-every 200 --eval-n 10000 \
            --criterion
    done; done; done
    wait
}

# 22. The reward advantage without a threshold, and a supervised learner that is not
#     miscalibrated. Pretraining used plain cross-entropy and so does the control; the ceiling arms
#     add a false-alarm penalty (0.4), and the control is the one learner here left releasing too
#     readily. Reward, the control and the control with that penalty, all at lr 1e-4, with the
#     best threshold and the area under the ROC hull at the start and the end. A third repeat
#     (rep 2), so the names do not collide with block 17. Prediction: the penalised control matches
#     reward at the decoder's own threshold, and all three match in area.
block22() {
    for cond in reward control control_fa; do for net in $NETS; do
        run "nf_c1_${cond}_auc_net$net" -m experiments.nf --task c1 --cond $cond --net $net \
            --rep 2 --lr 1e-4 --eval-n 10000 --criterion
    done; done
    wait
}

# 23. Neurofeedback proper, with a perfect learner. The reward is a scalar the frozen position
#     decoder computes from the delay state; the network learns it by reinforcement on its own
#     recurrent noise, with back-propagation carrying the credit. Nothing is supervised and there is
#     no task term. The node-perturbation rule failed to move the code, and this separates the two
#     reasons it could have: if the code moves here, the local rule's credit assignment was the
#     problem; if it does not, a decoder-computed scalar cannot steer the code at this scale. The null
#     scores each trial against another trial's position. Two learning rates.
block23() {
    for lr in 1e-4 3e-4; do for net in $NETS; do
        run "nf_c1_operant_lr${lr}_net$net" -m experiments.nf --task c1 --cond operant \
            --net $net --lr $lr --target cue --eval-n 10000 --panel --all-trials
        run "nf_c1_operant_shuf_lr${lr}_net$net" -m experiments.nf --task c1 --cond operant \
            --shuffle --net $net --lr $lr --target cue --eval-n 10000 --panel --all-trials
    done; done
    wait
}

# 24. Adam, not credit? Adam scales each parameter's step by the running size of its gradient, so a
#     noisy gradient moves the weights less whatever its direction, and "reward moves the weights
#     less and gets more per unit moved" (blocks 13b, 15) could be the optimiser. The same contrast
#     under plain gradient descent, with the threshold sweep and the policy shift. At lr 1e-2 both
#     signals wreck the network within 30 updates (the control's gradient is clipped on 80% of them),
#     and matching the per-update movement of the Adam runs puts both near 3e-4, so the two rates
#     are 1e-4 and 3e-4 for both. What has to survive is the ratio of behaviour, and of policy
#     shift, to weight movement; the clipped fraction is recorded, since a clipped step is a
#     normalised one.
block24() {
    for lr in 1e-4 3e-4; do for cond in control reward; do for net in $NETS; do
        run "nf_c1_${cond}_sgd_lr${lr}_net$net" -m experiments.nf --task c1 --cond $cond \
            --net $net --optimizer sgd --lr $lr --eval-n 10000 --criterion --retention
    done; done; done
    wait
}

# 25. The operant learner's positive control. Block 23 found that a reward computed by the position
#     decoder does not move the code, but the same score-function learner was never shown to learn
#     anything; a learner that learns nothing says nothing about the signal. Here it is paid the
#     task's outcome instead. It passes if behaviour gains more than the resolution on at least four
#     networks at either rate.
block25() {
    for lr in 1e-4 3e-4; do for net in $NETS; do
        run "nf_c1_operant_task_lr${lr}_net$net" -m experiments.nf --task c1 --cond operant \
            --operant-reward task --net $net --lr $lr --eval-n 10000
    done; done
    wait
}

# 26. The resolution of the threshold sweep itself, for the equivalence margins: ten sweeps of one
#     frozen network per task.
block26() {
    for task in c1 context; do
        run "criterion_resolution_$task" -m experiments.criterion_resolution --task $task --net 0
    done
    wait
}

# 27. Criterion theory. Reward on the trial outcome should leave the decoder's own threshold at the
#     payoff-optimal point of the network's final ROC, for whatever the payoffs are; cross-entropy
#     has no payoffs in it and should not. The false-alarm cost of the reward and the weight of
#     the false-alarm penalty in the supervised loss are the doses. A fourth repeat (rep 3), so
#     the names do not collide with blocks 17 and 22. Predictions are registered before the runs.
block27() {
    for net in $NETS; do
        for cost in 1 2 4; do
            run "nf_c1_reward_fac${cost}_net$net" -m experiments.nf --task c1 --cond reward \
                --net $net --rep 3 --fa-cost $cost --eval-n 10000 --criterion
        done
        for weight in 0 0.4 1.6; do
            run "nf_c1_control_fa_fap${weight}_net$net" -m experiments.nf --task c1 \
                --cond control_fa --net $net --rep 3 --fa-penalty $weight --lr 1e-4 \
                --eval-n 10000 --criterion
        done
    done
    wait
}

# 28. The first step, split into criterion and separation, for every signal and every angle of the
#     axis target, on the frozen networks. Cheap: gradients and a held-out read-out, no training.
block28() {
    for net in $NETS; do
        run "components_net$net" -m experiments.components --net $net
    done
    wait
}

# 29. The axis target as a dose: a frozen one-dimensional classifier separating response steps from
#     false-alarm steps along a direction at cosine -1, -0.5, 0, 0.5 and 1 to the decision axis,
#     inside the activity manifold, with no task term. The prediction is a slope of discriminability
#     on the cosine that changes sign, and a criterion shift that block 28 predicts per angle.
block29() {
    for cos in -1 -0.5 0 0.5 1; do for net in $NETS; do
        run "nf_c1_axis_cos${cos}_net$net" -m experiments.nf --task c1 --cond axis --cosine $cos \
            --net $net --lr 1e-4 --eval-n 10000 --criterion
    done; done
    wait
}

# 30. The second batch: networks 10-24, trained with the same curriculum, registered as an
#     independent replication with two primary contrasts -- the context contrast in area under the
#     ROC, and the c1 criterion shift (the best offset at the end, reward against control).
BATCH2="10 11 12 13 14 15 16 17 18 19 20 21 22 23 24"

block30pretrain() {
    for net in $BATCH2; do
        run "pretrain_net$net" -m experiments.pretrain --net $net
    done
    wait
    for task in c1 context; do for net in $BATCH2; do
        run "head_${task}_net$net" -m experiments.fit_head --task $task --net $net
    done; done
    wait
}

block30() {
    for net in $BATCH2; do for rep in 0 1; do for cond in control reward; do
        run "nf_context_${cond}_b2_net${net}_r$rep" -m experiments.nf --task context \
            --cond $cond --net $net --rep $rep --criterion
        run "nf_c1_${cond}_b2_net${net}_r$rep" -m experiments.nf --task c1 --cond $cond \
            --net $net --rep $rep --lr 1e-4 --eval-n 10000 --criterion
    done; done; done
    wait
}

# 31. Why the reward learner ends conservative. It is trained on sampled releases and scored with
#     the greedy read-out; a policy calibrated for sampling looks conservative when read greedily.
#     The reward learner at false-alarm cost 1 and 4 and the supervised learner, each swept with both
#     read-outs at the end. Predictions registered before the runs.
block31() {
    for net in $NETS; do
        for cost in 1 4; do
            run "nf_c1_reward_fac${cost}_samp_net$net" -m experiments.nf --task c1 --cond reward \
                --net $net --rep 4 --fa-cost $cost --eval-n 10000 --criterion --sampled-sweep
        done
        run "nf_c1_control_fa_fap0_samp_net$net" -m experiments.nf --task c1 --cond control_fa \
            --net $net --rep 4 --fa-penalty 0 --lr 1e-4 --eval-n 10000 --criterion --sampled-sweep
    done
    wait
}

# 32. The axis dose with the angle as the only dose: every direction is asked for the same rise of
#     one pooled standard deviation in the separation along it, instead of the classifier's loss,
#     whose demand grew with how far the direction started from being satisfied (block 29 could
#     not separate the two, r = 0.97).
block32() {
    for cos in -1 -0.5 0 0.5 1; do for net in $NETS; do
        run "nf_c1_axis_inc_cos${cos}_net$net" -m experiments.nf --task c1 --cond axis \
            --cosine $cos --axis-increment 1 --net $net --lr 1e-4 --eval-n 10000 --criterion
    done; done
    wait
}

# 33. The c1 signals again, keeping the whole network at 0, 150, 300, 600 and 1200 updates, so the
#     questions the end point cannot answer can be asked of the saved states (block 34): is
#     discriminability the same before saturation, what does noise at evaluation do, how many
#     dimensions does the change occupy, and where does it go relative to the decision read-out.
#     A seventh repeat (rep 6), so the names do not collide with earlier blocks.
block33() {
    for net in $NETS; do
        run "steps_reward_net$net" -m experiments.nf --task c1 --cond reward --net $net --rep 6 \
            --eval-n 10000 --save-at 0,150,300,600,1200 --ckpt-dir nf_c1_steps
        run "steps_control_net$net" -m experiments.nf --task c1 --cond control --net $net --rep 6 \
            --lr 1e-4 --eval-n 10000 --save-at 0,150,300,600,1200 --ckpt-dir nf_c1_steps
        run "steps_control_fa_net$net" -m experiments.nf --task c1 --cond control_fa --net $net \
            --rep 6 --lr 1e-4 --eval-n 10000 --save-at 0,150,300,600,1200 --ckpt-dir nf_c1_steps
        run "steps_joint_net$net" -m experiments.nf --task c1 --cond joint --net $net --rep 6 \
            --eval-n 10000 --save-at 0,150,300,600,1200 --ckpt-dir nf_c1_steps
        run "steps_k2_net$net" -m experiments.nf --task c1 --cond expand --kappa 2 --net $net \
            --rep 6 --lr 1e-4 --eval-n 10000 --save-at 0,150,300,600,1200 --ckpt-dir nf_c1_steps
        run "steps_shuf_net$net" -m experiments.nf --task c1 --cond expand --kappa 1 --shuffle \
            --net $net --rep 6 --lr 1e-4 --eval-n 10000 --save-at 0,150,300,600,1200 \
            --ckpt-dir nf_c1_steps
    done
    wait
}

block34() {
    for net in $NETS; do
        run "stepwise_net$net" -m experiments.stepwise --net $net
    done
    wait
}

# 35. A decoder fitted the way a protocol fits one: a fresh decision read-out trained on the frozen
#     state for 2000 read-out-only updates (the instrument's arm A), then frozen, and reward against
#     supervision through it. The question is whether "one ROC, different operating points" survives
#     a decoder that was not trained together with the network.
block35() {
    for net in $NETS; do for cond in reward control; do
        run "refit_${cond}_net$net" -m experiments.nf --task c1 --cond $cond --net $net --rep 7 \
            --lr 1e-4 --refit-decoder 2000 --eval-n 10000 --criterion
    done; done
    wait
}

# 36. The reward signal's entropy bonus (0.05 throughout) pulls every step's release probability
#     toward one half, and may set part of the reward learner's operating point. The reward learner
#     at bonus 0 and 0.01, swept with both read-outs.
block36() {
    for ent in 0 0.01; do for net in $NETS; do
        run "entropy_${ent}_net$net" -m experiments.nf --task c1 --cond reward --net $net --rep 8 \
            --entropy $ent --eval-n 10000 --criterion --sampled-sweep
    done; done
    wait
}

# 37. The reward learner on the context task with its decision threshold moved the way an
#     adaptive-threshold protocol moves it, from the learner's own sampled training trials with the
#     noise on: held at c = 0 among the trials that reach the response window, held at a hit rate of
#     0.7 among them, or replayed from another network's clamped run (yoked: the threshold moves but
#     does not follow this network). Against the same learner at a fixed threshold and supervision.
#     States kept at 0, 2000, 4000 and 8000 updates for the step split (block 37e).
B37="--task context --rep 9 --criterion --save-at 0,2000,4000,8000 --ckpt-dir nf_context_b37"
TRACK="--track-step 1 --track-every 25"
block37() {
    for net in $NETS; do
        run "b37_balance_net$net" -m experiments.nf $B37 --cond reward --net $net \
            --track balance $TRACK
    done
    for net in $NETS; do
        run "b37_rate_net$net" -m experiments.nf $B37 --cond reward --net $net \
            --track rate --track-rate 0.7 $TRACK
        run "b37_fixed_net$net" -m experiments.nf $B37 --cond reward --net $net
        run "b37_control_net$net" -m experiments.nf $B37 --cond control --net $net
    done
    wait
    for net in $NETS; do
        src=$(( (net + 1) % 5 ))
        run "b37_yoke_net$net" -m experiments.nf $B37 --cond reward --net $net --track yoke \
            --track-every 25 --yoke "$(ls results/nf_context_reward_*_balance_net${src}_r9.csv)"
    done
    wait
}

block37e() {
    for net in $NETS; do
        run "b37_split_net$net" -m experiments.criterion_share --net $net \
            --ckpt-dir nf_context_b37 --pattern "nf_context_*_net{net}_r9_u[248]000.pt" \
            --out results/criterion_share_b37_net$net.csv
    done
    wait
}

# 38. Is the reward learner's deficit on the context task the variance of its gradient estimate?
#     The same learner with each trial type's own mean reward as the baseline, and with four times
#     the batch.
block38() {
    for net in $NETS; do
        run "b38_typebase_net$net" -m experiments.nf $B37 --cond reward --net $net --baseline type
        run "b38_batch192_net$net" -m experiments.nf $B37 --cond reward --net $net --batch 192
    done
    wait
}

# 39. The reward learners at false-alarm cost 1, 2 and 4 again, keeping the end state, so they can
#     be read the way they learn (39e: noise on, sampled releases) and the payoff-optimal threshold
#     set against the cost each one was trained with.
block39() {
    for net in $NETS; do for cost in 1 2 4; do
        run "b39_fac${cost}_net$net" -m experiments.nf --task c1 --cond reward --net $net \
            --rep 10 --fa-cost $cost --eval-n 10000 --criterion --save-at 1200 \
            --ckpt-dir nf_c1_cost
    done; done
    wait
}

block39e() {
    run "b39_calibration" -m experiments.calibration --ckpt-dir nf_c1_cost
    wait
}

# 40. The first-passage account of the threshold on networks it was not fitted on: reward,
#     supervision, the magnified and the shuffled target on networks 10-24, states kept at 0 and
#     1200 updates, read by 40e as block 34 read networks 0-4.
block40() {
    for net in $BATCH2; do
        run "b40_reward_net$net" -m experiments.nf --task c1 --cond reward --net $net --rep 6 \
            --eval-n 10000 --save-at 0,1200 --ckpt-dir nf_c1_steps
        run "b40_control_net$net" -m experiments.nf --task c1 --cond control --net $net --rep 6 \
            --lr 1e-4 --eval-n 10000 --save-at 0,1200 --ckpt-dir nf_c1_steps
        run "b40_k2_net$net" -m experiments.nf --task c1 --cond expand --kappa 2 --net $net \
            --rep 6 --lr 1e-4 --eval-n 10000 --save-at 0,1200 --ckpt-dir nf_c1_steps
        run "b40_shuf_net$net" -m experiments.nf --task c1 --cond expand --kappa 1 --shuffle \
            --net $net --rep 6 --lr 1e-4 --eval-n 10000 --save-at 0,1200 --ckpt-dir nf_c1_steps
    done
    wait
}

block40e() {
    for net in $BATCH2; do
        run "b40_stepwise_net$net" -m experiments.stepwise --net $net
    done
    wait
}

# 41. Block 37's collapse as a race between two adaptive systems: the same c = 0 tracker, moved
#     every 100, 400 and 1600 updates instead of every 25. A rate at which the tracker still holds
#     the balance and the learner does not collapse would make separation of time scales the rule.
B41="--task context --cond reward --rep 11 --criterion"
block41() {
    for every in 100 400 1600; do for net in $NETS; do
        run "b41_every${every}_net$net" -m experiments.nf $B41 --net $net --track balance \
            --track-step 1 --track-every $every
    done; done
    wait
}

# 42. The absorbing state is partly the learner's: when every trial of a batch aborts, the batch's
#     own baseline leaves no advantage. The block-37 tracker with a baseline that lags (a slow
#     average over past batches).
block42() {
    for net in $NETS; do
        run "b42_runbase_net$net" -m experiments.nf $B41 --net $net --track balance \
            --track-step 1 --track-every 25 --baseline running
    done
    wait
}

# 43. Block 39 on networks 10-24: calibration by cost, and whether a dearer false alarm also buys
#     discriminability. 43e reads them five processes at a time.
block43() {
    for net in $BATCH2; do for cost in 1 2 4; do
        run "b43_fac${cost}_net$net" -m experiments.nf --task c1 --cond reward --net $net \
            --rep 10 --fa-cost $cost --eval-n 10000 --criterion --save-at 1200 \
            --ckpt-dir nf_c1_cost
    done; done
    wait
}

block43e() {
    local i=0
    for nets in 10,11,12 13,14,15 16,17,18 19,20,21 22,23,24; do
        run "b43_calibration_$i" -m experiments.calibration --ckpt-dir nf_c1_cost --nets $nets \
            --out results/calibration_c1_b2_$i.csv
        i=$((i + 1))
    done
    wait
}

# 44. Block 38's batch of 192 with the trials held equal: 2000 updates, as many trials as the
#     batch of 48 sees in 8000. If the gain survives it is the variance of the estimate, if not the
#     amount of experience.
block44() {
    for net in $NETS; do
        run "b44_batch192_u2000_net$net" -m experiments.nf --task context --cond reward --net $net \
            --rep 9 --criterion --batch 192 --updates 2000
    done
    wait
}

# 45. The magnified target as a dose on networks 10-24: kappa 0.5 asks for a compressed code, 1 for
#     the present one, 1.5 for a larger one; block 40 has kappa 2. The own-threshold curve is read
#     on 2000 trials only, since every behavioural number is recomputed on the kept states.
block45() {
    for kappa in 0.5 1 1.5; do for net in $BATCH2; do
        run "b45_k${kappa}_net$net" -m experiments.nf --task c1 --cond expand --kappa $kappa             --net $net --rep 6 --lr 1e-4 --eval-n 2000 --save-at 0,1200 --ckpt-dir nf_c1_steps
    done; done
    wait
}

# 45e. The kappa dose read on its kept states: induction, behaviour, first passage, per network.
block45e() {
    for net in $BATCH2; do
        run "b45e_net$net" -m experiments.induction --nets $net \
            --glob "nf_c1_expand_lam1_n*_k*_lr0.0001_net{net}_r6_u1200.pt" --exclude _shuf \
            --sweep-n 1500,6000 \
            --out results/induction_dose_c1_net$net.csv
    done
    wait
}

READ_NETS="0 1 2 3 4 $BATCH2"

# 46. How much of c1's discriminability passes through location: distractors moved to the cue, to
#     the opposite edge, or no cue, on the start and the supervised end state.
block46() {
    for nets in 0,1,2,3 4,10,11,12 13,14,15,16 17,18,19,20 21,22,23,24; do
        run "b46_location_${nets//,/_}" -m experiments.location --nets $nets \
            --out results/location_c1_${nets//,/_}.csv
    done
    wait
}

# 47. Induction of the magnified and shuffled targets against behaviour, across networks.
block47() {
    for net in $READ_NETS; do
        run "b47_induction_net$net" -m experiments.induction --nets $net \
            --glob "nf_c1_expand_lam1_n10000_k*_net{net}_r6_u1200.pt" \
            --out results/induction_c1_net$net.csv
    done
    wait
}

# 48. Reaction-time signatures: each start network's criterion line against every trained state.
block48() {
    for nets in 0,1,2,3 4,10,11,12 13,14,15,16 17,18,19,20 21,22,23,24; do
        run "b48_rt_${nets//,/_}" -m experiments.rt_signatures --nets $nets \
            --out results/rt_signatures_c1_${nets//,/_}.csv
    done
    wait
}

# 49. Re-association or new patterns on the context task.
block49() {
    for net in $NETS; do
        run "b49_repertoire_net$net" -m experiments.repertoire --nets $net \
            --out results/repertoire_context_net$net.csv
    done
    wait
}

# 50. Target strength: the c1 comparison at five strengths, without retraining.
block50() {
    for net in $NETS; do
        run "b50_psychometric_net$net" -m experiments.psychometric --nets $net \
            --out results/psychometric_c1_net$net.csv
    done
    wait
}

# 51. The magnified target applied only to trials cued in one sector (0 deg, +-45); the rest of the
#     disk gets no signal. 51e reads each network inside and opposite the sector, so the
#     comparison is within the network.
block51() {
    for net in $BATCH2; do
        run "b51_focus_net$net" -m experiments.nf --task c1 --cond expand --kappa 2 --net $net             --rep 6 --lr 1e-4 --eval-n 2000 --focus-theta 0 --focus-width 45 --save-at 0,1200             --ckpt-dir nf_c1_steps
    done
    wait
}

block51e() {
    for nets in 10,11,12 13,14,15 16,17,18 19,20,21 22,23,24; do
        run "b51e_focus_${nets//,/_}" -m experiments.focus --nets $nets             --out results/focus_c1_${nets//,/_}.csv
    done
    wait
}

# 52. Closed-loop difficulty after deBettencourt et al. (2015): the target fades within the trial
#     when the delay decoding of the cued position lapses; supervised learner, networks 10-24.
#     52y: each network yoked to the next network's recorded strengths.
block52() {
    for net in $BATCH2; do
        run "b52_closed_net$net" -m experiments.closed_loop --net $net --rep 0 \
            --sweep-n 1500,6000
    done
    wait
}

block52y() {
    for net in $BATCH2; do
        src=$(( net == 24 ? 10 : net + 1 ))
        run "b52_yoked_net$net" -m experiments.closed_loop --net $net --rep 0 \
            --sweep-n 1500,6000 --yoke-from results/closed_loop_closed_net${src}_r0.npy
    done
    wait
}

# 53. Block 50 on networks 10-24: reward and supervised end states at five target strengths.
block53() {
    for net in $BATCH2; do
        run "b53_psychometric_net$net" -m experiments.psychometric --nets $net \
            --states reward,control --out results/psychometric_c1_net$net.csv
    done
    wait
}

# 54. After closed-loop or yoked training: is the positional code lost, or moved out of the
#     directions the frozen head reads, and does behaviour still use location?
block54() {
    for net in $BATCH2; do
        run "b54_relocated_net$net" -m experiments.relocated --nets $net \
            --out results/relocated_c1_net$net.csv
    done
    wait
}

# 55. Location with a control that keeps the distractors' distances from the centre: every
#     distractor mirrored through the centre from the cue, on the start, the supervised end, and
#     the closed-loop and yoked end states.
block55() {
    for net in $READ_NETS; do
        run "b55_mirror_net$net" -m experiments.location --nets $net --variants mirror \
            --conds control,closed,yoked --out results/location_mirror_c1_net$net.csv
    done
    wait
}

# 56. The context end states swept to +12: on the default grid the reward learner's curve stops
#     before its peak.
block56() {
    for net in $NETS; do
        run "b56_wide_net$net" -m experiments.wide_sweep --nets $net \
            --out results/wide_sweep_context_net$net.csv
    done
    wait
}

# 57. After the magnified and the shuffled target: is the lost discriminability lost from the
#     network, or from the frozen decision read-out? A fresh read-out is fitted on each end
#     state with the recurrent weights frozen.
block57() {
    for net in $READ_NETS; do
        run "b57_access_net$net" -m experiments.access --nets $net \n            --states theta0,control,k2,shuffled --select-n 1000 --report-n 4000 \n            --out results/access_c1_net$net.csv
    done
    wait
}

main() {
    for b in "$@"; do
        echo "=== block $b started $(date)"
        "block$b"
        echo "=== block $b finished $(date)"
    done
}

# one line, so bash has read the whole file before anything runs: a launch in progress keeps
# doing what it was started with even if blocks are added to this file meanwhile
main "$@"; exit
