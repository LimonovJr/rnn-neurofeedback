# Sourced by the launchers. Training is CPU-bound at this network size, so the runs are spread over
# processes rather than threads: five processes of two threads each fills the machine without the
# processes fighting over it.
PY=${PY:-python}
JOBS=${JOBS:-5}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
LOGS=results/logs
mkdir -p "$LOGS"

run() {
    local name=$1; shift
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
    echo "launching $name"
    ( $PY "$@" > "$LOGS/$name.log" 2>&1 ) &
}
