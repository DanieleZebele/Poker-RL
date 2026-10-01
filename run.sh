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

# Twenty workers by default, not as many as the machine can technically hold.
# Filling every core (nproc - 2, which is 30 on the 32-core boxes) is what the
# fleet ran before, and five of seven machines went unresponsive under it; the
# cause was never established, but a machine running at its own ceiling has no
# headroom for the end-of-run phases, where every worker of a generation
# arrives at the same moment and each one loads models it did not hold while
# training. Twenty still leaves a third of a 32-core box free for that, while
# ten was leaving two thirds of it idle.
#
# Watch this one. Every worker of a generation reaches the population round at
# about the same time and each loads ~55 models there, measured at 203 MB on top
# of its ~514 MB of training footprint, so twenty-five workers in that phase
# together want ~18 GB. Two things have made that window wider since the ceiling
# was first set: the round is ten times longer than it used to be
# (`global_arena.DEFAULT_HANDS_PER_GAME` is 1000 now), and
# `rate_against_benchmark` plays 500 rated sessions of 1000 hands against the
# frozen anchors, so 500,000 hands per published model. If machines start going
# unresponsive again, this and `--global-games-per-model` are the two dials --
# and POKER_WORKERS overrides this one without touching the file.
#
# The history is worth knowing before raising it further: the fleet ran at 30,
# five of seven machines went unresponsive, no cause was ever established, and
# the ceiling went to 10 and then to 20 as a compromise. 25 is a deliberate step
# back up, not a measured safe value.
#
# Cores and memory still cap it, for the small VMs: each worker is its own
# Python+torch process at roughly 700 MB once its pool is loaded, so a small VM
# runs out of RAM long before it runs out of cores.
DEFAULT_WORKER_CEILING=25
default_workers() {
    local cores free_mb by_cores by_memory chosen
    cores=$(nproc)
    free_mb=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
    by_cores=$(( cores - 2 ))
    by_memory=$(( (free_mb - 3000) / 700 ))
    chosen=$DEFAULT_WORKER_CEILING
    [ "$by_cores" -lt "$chosen" ] && chosen=$by_cores
    [ "$by_memory" -lt "$chosen" ] && chosen=$by_memory
    echo "$chosen"
}

WORKERS="${POKER_WORKERS:-$(default_workers)}"
[ "$WORKERS" -lt 1 ] && WORKERS=1

# Passed to `start` and to `status`/`watch` alike: the status scales every
# progress bar against it, so a value that does not match the running loop's
# would draw every worker at the wrong fraction.
#
# 1000, up from 100. Training measurably had not finished at 100: the learner's
# win rate against its own pool was still climbing (-94.5 -> -30.9 -> -19.3 ->
# -13.5 bb/100 at iterations 25/50/75/100), `clip` was still 0.072 and `kl`
# 0.009, and a duplicate-deck duel put the iteration-100 model +57 bb/100 ahead
# of the iteration-50 one from the same seed (t = 7.5). A run now takes ~2.8
# hours instead of ~25 minutes, so a machine produces roughly six times fewer
# models per day -- deliberately trading breadth for depth.
ITERATIONS="${POKER_ITERATIONS:-1000}"

# 50 opponents drawn per run, up from 20. A run ten times longer against the
# same fixed twenty would learn to beat those twenty rather than to play better
# -- and since the in-run evaluation uses that same pool, the number would keep
# rising while the model narrowed. A wider draw dilutes that. Each extra model
# costs ~3.7 MB resident in the worker.
POOL_MODELS="${POKER_POOL_MODELS:-50}"

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
    log "macchina $MACHINE | $WORKERS worker | seed-base $seed"
    log "store condiviso dei modelli $MODELS, stato di questa macchina $ROOT"
    # **Flags deliberately absent: --inherit-fraction, --hands, --eval-every and
    # --eval-sessions.** The first two are decided per worker by
    # `loop.hyperparameter_plan`, and passing them here pinned nothing while
    # looking like it did. The two --eval ones are absent for the sibling reason:
    # a validation round is 10 rated sessions every 100 iterations, that interval
    # is `poker-loop`'s own default, and this line used to carry
    # `--eval-every 250 --eval-hands 10000` -- which would have quietly kept the
    # fleet on 4 rounds a run instead of 10 after the defaults moved, exactly the
    # failure the paragraph below is about. (Two more, --self-share and
    # --eval-hands, no longer exist on `poker-loop` at all.)
    # `--inherit-fraction 0.5` was worse than misleading:
    # an explicit flag beats the default, so it kept half the fleet starting
    # from random networks for a whole day after from-scratch runs were
    # supposed to be gone -- measured on zebele-slaves-2 generation 425, 12 of
    # 25 workers inheriting and 13 at --fresh-lr, exactly the 0.5 this line
    # asked for. The rule it cost: when a default moves in `loop.py`, check
    # whether this command line is overriding it.
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    setsid nohup "$VENV/bin/poker-loop" \
        --workers "$WORKERS" --generations 0 --iterations "$ITERATIONS" \
        --players 6 --stack 200 --sb 1 --bb 2 \
        --state-dir "$ROOT" --work-dir "$ROOT/work" --models-dir "$MODELS" \
        --log-dir "$ROOT/logs/loop" --machine "$MACHINE" \
        --pool-models "$POOL_MODELS" \
        --archive-every 500 \
        --benchmark-dir "$BENCHMARK" --benchmark-hands 3000 \
        --seed-base "$seed" --top 15 \
        >> "$ROOT/logs/supervisor.log" 2>&1 < /dev/null &
    sleep 5
    log "avviato. segui con: ./run.sh watch"
}

# No default subcommand: `./run.sh` on its own used to mean `start`, which put a
# 30-worker production loop on the machine from a bare, harmless-looking command
# (it happened). Nothing starts unless `start` is spelled out.
case "${1:-help}" in
    setup)  setup ;;
    start)  start ;;
    status) "$VENV/bin/poker-loop" --status --iterations "$ITERATIONS" \
                --state-dir "$ROOT" --models-dir "$MODELS" \
                --log-dir "$ROOT/logs/loop" --benchmark-dir "$BENCHMARK" ;;
    watch)  "$VENV/bin/poker-loop" --status --watch 30 --iterations "$ITERATIONS" \
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
