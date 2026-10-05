#!/usr/bin/env bash
# One command to bring a machine from a fresh clone to a running training loop.
#
#   ./run.sh start     bootstrap if needed, then run the loop until stopped
#                      (must be spelled out: a bare ./run.sh only prints this)
#   ./run.sh status    leaderboard, benchmark trend, live worker progress
#   ./run.sh watch     the same, redrawing every 30s
#   ./run.sh stop      finish the current generation, then exit cleanly
#   ./run.sh kill      stop everything NOW, whatever stage each process is in
#   ./run.sh setup     only build the environment, do not launch
#
# Safe to re-run: every step checks whether it is already done.
set -euo pipefail

cd "$(dirname "$0")"
PY_VERSION="3.13"
MACHINE="${POKER_MACHINE:-$(hostname -s)}"

log() { printf '\033[1m==>\033[0m %s\n' "$*"; }

# The project directory is on NFS, so the .venv in it is visible to every
# machine -- but a venv's interpreter is a symlink into the *local* disk of
# whoever built it (a uv-managed CPython under ~/.local). Whether that symlink
# resolves on this host is therefore the only question that matters, and it can
# only be answered *after* the interpreter has been installed locally.
#
# Getting that order wrong is expensive and was the original mistake here:
# deciding first meant every fresh VM saw a dangling symlink, concluded the
# shared venv was unusable, and built its own 5.5 GB copy -- six machines, 33 GB
# wasted -- even though uv installs CPython at an identical path on each one, so
# the shared venv would have worked for all of them.
ensure_interpreter() {
    if command -v "python$PY_VERSION" >/dev/null 2>&1; then
        command -v "python$PY_VERSION"
        return
    fi
    if ! command -v uv >/dev/null 2>&1; then
        log "installo uv (serve solo a procurare Python $PY_VERSION)" >&2
        curl -LsSf https://astral.sh/uv/install.sh | sh >&2
        export PATH="$HOME/.local/bin:$PATH"
    fi
    # ~33 MB, and idempotent: a machine that already has it gets the path back.
    uv python install "$PY_VERSION" >&2
    uv python find "$PY_VERSION"
}

pick_venv() {
    if [ -x ".venv/bin/python" ] && .venv/bin/python -c "" >/dev/null 2>&1; then
        echo ".venv"
    else
        echo ".venv-$MACHINE"
    fi
}
VENV=".venv"

setup() {
    # Always first: the shared venv cannot be judged until the interpreter it
    # points at is present locally. See the comment on ensure_interpreter.
    local python; python="$(ensure_interpreter)"
    VENV="$(pick_venv)"

    if [ "$VENV" != ".venv" ]; then
        log "il venv condiviso non gira qui, ne creo uno per questa macchina: $VENV"
        if ! "$VENV/bin/python" -c "" >/dev/null 2>&1; then
            "$python" -m venv "$VENV"
            "$VENV/bin/pip" install --quiet --upgrade pip
        fi
    fi

    if ! "$VENV/bin/python" -c "import torch, pokerlab" >/dev/null 2>&1; then
        # The CUDA build on purpose, even with no GPU present: it runs fine on
        # CPU and the same environment keeps working if the machine gets a GPU.
        log "installo le dipendenze (torch CUDA + pokerlab, qualche GB)"
        "$VENV/bin/pip" install -e ".[dev,rl]"
    fi
    log "ambiente pronto: $VENV, $("$VENV/bin/python" --version)"
}

# This project directory lives on an NFS export mounted by several VMs, so every
# machine sees the same files. Every trained model lives in ONE shared store,
# checkpoints/models/: files are written once, with machine-prefixed names, and
# never modified, so any machine can read any model with no coordination. The
# ratings are one file per model under checkpoints/global/, updated under a
# per-model lock. Only this machine's own state (logs, scratch, loop_state.json)
# lives in its own subtree.
ROOT="checkpoints/machines/$MACHINE"
MODELS="checkpoints/models"
BENCHMARK="checkpoints/benchmark"   # shared, read-only: comparable across hosts

# Distinct seeds per machine. Without this every VM would run the identical
# seeds, train the identical models and contribute nothing to each other.
machine_seed() {
    printf '%d' "$(( 0x$(printf '%s' "$MACHINE" | sha256sum | cut -c1-6) ))"
}

# How many workers: `poker-loop --workers 0` means "as many as this machine
# holds", i.e. the fleet-wide `worker_ceiling` in `config.toml`, lowered on a small
# VM by its cores (all but two) and its free memory (`loop.auto_workers`).
# POKER_WORKERS pins a number for this machine only, for a one-off run.
WORKERS="${POKER_WORKERS:-0}"

# The training length and the pool size live in `config.toml`, with the reasoning
# for each value, not here: an explicit flag beats the file, so a value pinned in
# this script would silently override what the file says. POKER_ITERATIONS and
# POKER_POOL_MODELS still override it for a one-off run, and are passed to
# `status`/`watch` too -- the status scales every progress bar against the
# iterations, so a value that does not match the running loop's would draw every
# worker at the wrong fraction.
ITERATION_FLAGS=()
[ -n "${POKER_ITERATIONS:-}" ] && ITERATION_FLAGS=(--iterations "$POKER_ITERATIONS")
POOL_FLAGS=()
[ -n "${POKER_POOL_MODELS:-}" ] && POOL_FLAGS=(--pool-models "$POKER_POOL_MODELS")

start() {
    setup
    if pgrep -f "bin/poker-loop --workers --generations" >/dev/null 2>&1 \
       || pgrep -af "bin/poker-loop --workers" 2>/dev/null | grep -q "$ROOT"; then
        log "un loop è già in esecuzione su questa macchina; usa ./run.sh stop"
        exit 1
    fi
    rm -f "$ROOT/STOP"
    mkdir -p "$ROOT/logs/loop" "$MODELS"
    local seed; seed="$(machine_seed)"
    log "macchina $MACHINE | worker: $([ "$WORKERS" -gt 0 ] && echo "$WORKERS" || echo "auto, tetto da config.toml") | seed-base $seed"
    log "store condiviso dei modelli $MODELS, stato di questa macchina $ROOT"
    # **Flags deliberately absent: --hands, --eval-every and
    # --eval-sessions, and everything `config.toml` sets (game, iterations, pool
    # size, benchmark hands).** `--hands` is decided per
    # worker by `loop.hyperparameter_plan`, and passing it here would pin
    # nothing while looking like it did. The two --eval ones are absent so
    # `loop.py`'s own defaults govern the size and cadence of a validation pass.
    # The rule: an explicit flag beats the file, and the file beats a default, so
    # a flag added here silently overrides both -- only machine-local values
    # (workers, directories, machine, seed) belong on this line.
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    setsid nohup "$VENV/bin/poker-loop" \
        --workers "$WORKERS" --generations 0 \
        ${ITERATION_FLAGS[@]+"${ITERATION_FLAGS[@]}"} \
        --state-dir "$ROOT" --work-dir "$ROOT/work" --models-dir "$MODELS" \
        --log-dir "$ROOT/logs/loop" --machine "$MACHINE" \
        ${POOL_FLAGS[@]+"${POOL_FLAGS[@]}"} \
        --benchmark-dir "$BENCHMARK" \
        --seed-base "$seed" --top 15 \
        >> "$ROOT/logs/supervisor.log" 2>&1 < /dev/null &
    sleep 5
    log "avviato. segui con: ./run.sh watch"
}

# No default subcommand: a bare `./run.sh` must not put a production loop on the
# machine from a harmless-looking command. Nothing starts unless `start` is
# spelled out.
case "${1:-help}" in
    setup)  setup ;;
    start)  start ;;
    status) "$VENV/bin/poker-loop" --status ${ITERATION_FLAGS[@]+"${ITERATION_FLAGS[@]}"} \
                --state-dir "$ROOT" --models-dir "$MODELS" \
                --log-dir "$ROOT/logs/loop" --benchmark-dir "$BENCHMARK" ;;
    watch)  "$VENV/bin/poker-loop" --status --watch 30 ${ITERATION_FLAGS[@]+"${ITERATION_FLAGS[@]}"} \
                --state-dir "$ROOT" --models-dir "$MODELS" \
                --log-dir "$ROOT/logs/loop" --benchmark-dir "$BENCHMARK" ;;
    stop)   mkdir -p "$ROOT" && touch "$ROOT/STOP"
            log "richiesto arresto: il loop finisce la generazione in corso e poi esce" ;;
    # The brutal counterpart of `stop`: freezes the whole tree so the supervisor
    # cannot start another generation while the kill is in progress, then kills
    # it. The STOP file goes down too, or the next `start` would refuse to run.
    kill)   mkdir -p "$ROOT" && touch "$ROOT/STOP"
            "$VENV/bin/poker-kill" --state-dir "$ROOT" --work-dir "$ROOT/work" || true
            rm -f "$ROOT/STOP" ;;
    kill-dry) "$VENV/bin/poker-kill" --state-dir "$ROOT" --work-dir "$ROOT/work" --dry-run ;;
    help|-h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//' ; exit 0 ;;
    *)      sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//' ; exit 1 ;;
esac
