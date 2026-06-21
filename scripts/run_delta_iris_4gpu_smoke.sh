#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="zhenglu_delta_iris"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
TIMESTAMP="${DELTA_IRIS_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/logs"
MAIN_LOG="${LOG_DIR}/smoke_${TIMESTAMP}_main.log"
CONDA_ROOT_HINT="/data/share/hxd/miniconda3"

TASK_NAMES=(Breakout Boxing Seaquest RoadRunner)
ENV_IDS=(BreakoutNoFrameskip-v4 BoxingNoFrameskip-v4 SeaquestNoFrameskip-v4 RoadRunnerNoFrameskip-v4)

mkdir -p "${LOG_DIR}" "${REPO_ROOT}/outputs"
exec > >(tee -a "${MAIN_LOG}") 2>&1

declare -a GPUS=()
declare -a PIDS=()
declare -a TASK_GPUS=()
declare -a TASK_LOGS=()
declare -a STATUSES=()
declare -A PID_TO_INDEX=()

info() {
  echo "[INFO] $*"
}

warn() {
  echo "[WARN] $*" >&2
}

activate_conda_env() {
  local conda_sh=""

  if [[ -f "${CONDA_ROOT_HINT}/etc/profile.d/conda.sh" ]]; then
    conda_sh="${CONDA_ROOT_HINT}/etc/profile.d/conda.sh"
  elif [[ -n "${CONDA_EXE:-}" ]]; then
    conda_sh="$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
  fi

  if [[ -n "${conda_sh}" && -f "${conda_sh}" ]]; then
    # shellcheck source=/dev/null
    source "${conda_sh}"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo "[ERROR] conda was not found. Expected it near ${CONDA_ROOT_HINT} or on PATH." >&2
    return 1
  fi
  conda activate "${ENV_NAME}"
}

parse_visible_gpus() {
  local raw="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -z "${raw}" ]]; then
    raw="0,1,2,3"
  fi
  raw="${raw// /,}"

  local fields=()
  IFS=',' read -r -a fields <<< "${raw}"
  for gpu in "${fields[@]}"; do
    if [[ -n "${gpu}" ]]; then
      GPUS+=("${gpu}")
    fi
  done

  if (( ${#GPUS[@]} < 4 )); then
    echo "[ERROR] Need at least 4 visible GPUs, got '${raw}'." >&2
    return 1
  fi
  GPUS=("${GPUS[@]:0:4}")
}

terminate_children() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      if command -v pkill >/dev/null 2>&1; then
        pkill -TERM -P "${pid}" >/dev/null 2>&1 || true
      fi
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
}

on_signal() {
  local signal_name="$1"
  warn "Received ${signal_name}; terminating smoke-test child processes."
  terminate_children
  wait || true
  if [[ "${signal_name}" == "TERM" ]]; then
    exit 143
  fi
  exit 130
}

run_task() {
  local index="$1"
  local task="${TASK_NAMES[${index}]}"
  local env_id="${ENV_IDS[${index}]}"
  local gpu="${GPUS[${index}]}"
  local task_log="${LOG_DIR}/smoke_${TIMESTAMP}_${task}.log"
  local run_dir="${REPO_ROOT}/outputs/smoke_${TIMESTAMP}/${task}"

  TASK_GPUS[${index}]="${gpu}"
  TASK_LOGS[${index}]="${task_log}"

  {
    info "Task ${task} starting at $(date -Is)"
    info "Env id: ${env_id}"
    info "Assigned outer CUDA_VISIBLE_DEVICES=${gpu}; code uses params.common.device=cuda:0"
    info "Hydra run dir: ${run_dir}"

    CUDA_VISIBLE_DEVICES="${gpu}" python src/main.py \
      env=atari \
      params=atari \
      "env.train.id=${env_id}" \
      "params.common.device=cuda:0" \
      "params.common.epochs=1" \
      "params.common.do_checkpoint=false" \
      "params.training.should=false" \
      "params.evaluation.should=true" \
      "params.evaluation.every=1" \
      "params.evaluation.tokenizer.start_after_epochs=999999" \
      "params.evaluation.world_model.start_after_epochs=999999" \
      "params.evaluation.tokenizer.save_reconstructions=false" \
      "params.collection.test.num_envs=1" \
      "params.collection.test.num_episodes_to_save=0" \
      "params.collection.test.config.num_episodes=1" \
      "env.test.max_episode_steps=64" \
      "env.test.noop_max=1" \
      "wandb.mode=disabled" \
      "hydra.run.dir=${run_dir}"

    info "Task ${task} completed at $(date -Is)"
  } > >(tee -a "${task_log}") 2>&1
}

launch_tasks() {
  for i in "${!TASK_NAMES[@]}"; do
    run_task "${i}" &
    local pid="$!"
    PIDS+=("${pid}")
    PID_TO_INDEX["${pid}"]="${i}"
    info "Launched ${TASK_NAMES[${i}]} on GPU ${GPUS[${i}]} with PID ${pid}; log ${LOG_DIR}/smoke_${TIMESTAMP}_${TASK_NAMES[${i}]}.log"
  done
}

wait_for_tasks() {
  local remaining="${#PIDS[@]}"
  local overall=0

  while (( remaining > 0 )); do
    local finished_pid=""
    local status=0
    if wait -n -p finished_pid; then
      status=0
    else
      status="$?"
    fi

    if [[ -n "${finished_pid}" && -n "${PID_TO_INDEX[${finished_pid}]+x}" ]]; then
      local index="${PID_TO_INDEX[${finished_pid}]}"
      STATUSES[${index}]="${status}"
      info "Task ${TASK_NAMES[${index}]} with PID ${finished_pid} exited with status ${status}."
    else
      warn "A child exited with status ${status}, but its PID was not reported by wait."
    fi

    remaining=$((remaining - 1))
    if (( status != 0 && overall == 0 )); then
      overall="${status}"
      warn "A smoke task failed; terminating any remaining smoke tasks."
      terminate_children
    fi
  done

  info "Smoke task summary:"
  for i in "${!TASK_NAMES[@]}"; do
    local status="${STATUSES[${i}]:-not-reaped}"
    info "  ${TASK_NAMES[${i}]} | pid=${PIDS[${i}]} | gpu=${TASK_GPUS[${i}]} | status=${status} | log=${TASK_LOGS[${i}]}"
  done

  return "${overall}"
}

main() {
  trap 'on_signal INT' INT
  trap 'on_signal TERM' TERM

  cd "${REPO_ROOT}"
  export HYDRA_FULL_ERROR=1
  export WANDB_MODE=disabled
  export SDL_VIDEODRIVER=dummy
  export OMP_NUM_THREADS=8
  export MKL_NUM_THREADS=8
  export OPENBLAS_NUM_THREADS=8
  export NUMEXPR_NUM_THREADS=8
  export TOKENIZERS_PARALLELISM=false

  info "Repo root: ${REPO_ROOT}"
  info "Main log: ${MAIN_LOG}"
  activate_conda_env
  parse_visible_gpus
  info "Using first four visible GPUs: ${GPUS[*]}"

  launch_tasks
  wait_for_tasks
}

main "$@"
