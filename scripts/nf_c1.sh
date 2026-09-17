#!/usr/bin/env bash
# The signal comparison at measured small headroom, then the lambda sweep on the same cell.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/pool.sh

for cond in control position joint reward; do
    for net in 0 1 2 3 4; do
        extra=""
        [ "$cond" = position ] && extra="--train-decoder"
        run "nf_c1_${cond}_net$net" -m experiments.nf --task c1 --cond $cond --net $net \
            --panel $extra
    done
done
wait

for lam in 0.5 1.0 2.0; do
    for net in 0 1 2 3 4; do
        for rep in 0 1; do
            run "nf_c1_joint_lam${lam}_net${net}_r$rep" -m experiments.nf --task c1 --cond joint \
                --net $net --rep $rep --lam $lam --eval-n 10000
        done
    done
done
for net in 0 1 2 3 4; do
    for rep in 0 1; do
        run "nf_c1_control_net${net}_r$rep" -m experiments.nf --task c1 --cond control \
            --net $net --rep $rep --eval-n 10000
    done
done
wait
