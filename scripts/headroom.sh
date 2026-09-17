#!/usr/bin/env bash
# The headroom table: four settings x two arms x five networks x four restarts, plus the nonlinear
# read-out on the intact cell. About 200 process-hours in total.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/pool.sh

for setting in c1 c1r1 c1r3 patch; do
    for arm in A B; do
        for net in 0 1 2 3 4; do
            for init in incumbent fresh_a fresh_b fresh_c; do
                run "headroom_${setting}_${arm}_net${net}_${init}" -m experiments.headroom \
                    --setting $setting --arm $arm --init $init --net $net
            done
        done
    done
done
wait

for net in 0 1 2 3 4; do
    for init in incumbent fresh_a fresh_b fresh_c; do
        run "headroom_c1_Aplus_net${net}_${init}" -m experiments.headroom \
            --setting c1 --arm Aplus --init $init --net $net
    done
done
wait
