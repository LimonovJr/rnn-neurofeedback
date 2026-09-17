#!/usr/bin/env bash
# The five networks, then the position head each task needs.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/pool.sh

for net in 0 1 2 3 4; do
    run "pretrain_net$net" -m experiments.pretrain --net $net
done
wait

for task in c1 context; do
    for net in 0 1 2 3 4; do
        run "head_${task}_net$net" -m experiments.fit_head --task $task --net $net
    done
done
wait
