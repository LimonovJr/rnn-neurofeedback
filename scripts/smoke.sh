#!/usr/bin/env bash
# Every entry point on a budget small enough to finish in a few minutes, so a change can be checked
# end to end before a multi-hour run is launched. Writes to results/smoke/.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
OUT=results/smoke
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
mkdir -p "$OUT"

run() { echo; echo "### $*"; "$@"; }

run $PY -m experiments.headroom --setting c1 --arm A --init incumbent --net 0 --cap 40 --chunk 20 --reeval 200 \
    --out $OUT/headroom_c1_A_net0_incumbent.csv
run $PY -m experiments.headroom --setting c1r1 --arm B --init fresh_a --net 0 --cap 40 --chunk 20 --reeval 200 \
    --out $OUT/headroom_c1r1_B_net0_fresh_a.csv
run $PY -m experiments.headroom --setting patch --arm Aplus --init fresh_a --net 0 --cap 40 --chunk 20 --reeval 200 \
    --hidden 8 --out $OUT/headroom_patch_Aplus_net0_fresh_a.csv

run $PY -m experiments.degradation --net 0 --sigmas 0.18 --reeval 200 --out $OUT/degradation_net0.csv
run $PY -m experiments.patch_integration --net 0 --reeval 200 --out $OUT/patch_net0.csv
run $PY -m experiments.competence --net 0 --rep 0 --marks 10,20 --trials 200 --panel-trials 300 --out $OUT/competence_net0_r0.csv
run $PY -m experiments.resolutions --task context --rep 0 --reeval 200 --panel-trials 300 --out $OUT/resolution_context_net0_r0.csv
run $PY -m experiments.positive_control --net 0 --decode-only --reeval 200

run $PY -m experiments.nf --task c1 --cond joint --net 0 --rep 0 --updates 6 --eval-every 3 \
    --eval-n 200 --panel --panel-trials 300 --out $OUT/nf_c1_joint_lam1_n200_net0_r0.csv
run $PY -m experiments.nf --task context --cond reward --net 0 --rep 0 --updates 6 --eval-every 3 \
    --eval-n 200 --out $OUT/nf_context_reward_lam1_n200_net0_r0.csv
run $PY -m experiments.nf --task c1 --cond reward_aux --rule local --net 0 --rep 0 --updates 6     --eval-every 3 --eval-n 200 --target cue --out $OUT/nf_c1_reward_aux_local_n200_net0_r0.csv
run $PY -m experiments.nf --task c1 --cond expand --kappa 2 --shuffle --net 0 --rep 0 --updates 6     --eval-every 3 --eval-n 200 --out $OUT/nf_c1_expand_shuf_n200_net0_r0.csv
run $PY -m experiments.manifold --net 0 --kind outside --angle 90 --auto --block 5 --updates 10     --eval-every 5 --out $OUT/manifold_outside_auto_net0_c0.csv

run $PY -m experiments.representation --cond joint --net 0 --rep 0 --panel-trials 300 --out $OUT/repr_context_joint_net0_r0.csv
run $PY -m experiments.gradients --cond joint --net 0 --rep 0 --out $OUT/grad_context_joint_net0_r0.csv
run $PY -m experiments.gradients --cond joint --net 0 --rep 0 --step 0.003     --out $OUT/grad_step_context_joint_L0.003_net0_r0.csv

echo
echo "### aggregation over the smoke results"
for table in headroom degradation patch nf-c1 nf-context manifold representation gradients projection; do
    $PY -m experiments.aggregate $table --results $OUT || true
done
