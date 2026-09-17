#!/usr/bin/env bash
# Everything that calibrates the instrument rather than testing neurofeedback: the positive control,
# the resolutions, the degradation ladder, the integration measurement and the competence ladder.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/pool.sh

for net in 0 1 2; do
    run "control_net$net" -m experiments.positive_control --net $net
done
for net in 3 4; do
    run "control_decode_net$net" -m experiments.positive_control --net $net --decode-only
done
wait

for task in c1 context; do
    for rep in 0 1 2 3 4 5 6 7 8 9; do
        run "resolution_${task}_r$rep" -m experiments.resolutions --task $task --rep $rep
    done
done
wait

for net in 0 1 2 3 4; do
    run "degradation_net$net" -m experiments.degradation --net $net
    run "patch_net$net" -m experiments.patch_integration --net $net
done
wait

for net in 0 1 2 3 4; do
    for rep in 0 1; do
        run "competence_net${net}_r$rep" -m experiments.competence --net $net --rep $rep
    done
done
wait
