# Sourced by the launchers. Training is CPU-bound at this network size, so the runs are spread over
# processes rather than threads: five processes of two threads each fills the machine without the
# processes fighting over it.
PY=${PY:-python}
JOBS=${JOBS:-5}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
LOGS=results/logs
mkdir -p "$LOGS"

# a launcher that starts sixty runs against the wrong interpreter leaves sixty empty logs and no
# CSVs, which is a slow way to find out; ask once instead
$PY -c "import torch" 2>/dev/null || {
    echo "no interpreter with torch on PATH as '$PY'; set PY, e.g. PY=.venv/Scripts/python $0" >&2
    exit 1
}

run() {
    local name=$1; shift
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
    echo "launching $name"
    ( $PY "$@" > "$LOGS/$name.log" 2>&1 ) &
}
