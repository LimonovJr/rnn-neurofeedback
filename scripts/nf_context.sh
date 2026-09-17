#!/usr/bin/env bash
# The comparison where the headroom is large: four conditions, five networks, two replicates, with
# the last checkpoints kept so the representation and gradient passes can read them.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/pool.sh

for cond in control joint reward position; do
    for net in 0 1 2 3 4; do
        for rep in 0 1; do
            run "nf_context_${cond}_net${net}_r$rep" -m experiments.nf --task context \
                --cond $cond --net $net --rep $rep --ckpt-dir nf_context
        done
    done
    wait
done

for cond in control joint reward position; do
    for net in 0 1 2 3 4; do
        for rep in 0 1; do
            run "repr_${cond}_net${net}_r$rep" -m experiments.representation \
                --cond $cond --net $net --rep $rep
        done
    done
done
for net in 0 1 2 3 4; do
    run "repr_theta0_net$net" -m experiments.representation --cond theta0 --net $net
done
wait

for cond in theta0 control joint reward position; do
    for net in 0 1 2 3 4; do
        run "grad_${cond}_net$net" -m experiments.gradients --cond $cond --net $net --rep 0
    done
done
wait
