#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="zhenglu_delta_iris"
PYTHON_VERSION="3.10"
PYPI_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
PYPI_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
ALIYUN_TORCH_FIND_LINKS="https://mirrors.aliyun.com/pytorch-wheels/cu121/torch_stable.html"
PYTORCH_CU121_INDEX_URL="https://download.pytorch.org/whl/cu121"
CONDA_ROOT_HINT="/data/share/hxd/miniconda3"
TMP_REQ=""

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
LOG_DIR="${REPO_ROOT}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/setup_delta_iris_env_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

warn() {
  echo "[WARN] $*" >&2
}

info() {
  echo "[INFO] $*"
}

cleanup() {
  if [[ -n "${TMP_REQ:-}" ]]; then
    rm -f "${TMP_REQ}"
  fi
}

activate_conda_base() {
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
}

install_system_packages_if_possible() {
  local packages=(
    libgl1
    libegl1
    libosmesa6
    libxrender1
    libsm6
    libice6
    libxcursor1
    libxi6
    libxrandr2
    libxinerama1
    libpulse0
    ffmpeg
    rsync
  )

  if ! command -v apt-get >/dev/null 2>&1; then
    warn "apt-get is not available; skipping system package installation."
    return 0
  fi

  local apt_cmd=()
  if [[ "${EUID}" -eq 0 ]]; then
    apt_cmd=(apt-get)
  elif command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    apt_cmd=(sudo apt-get)
  else
    warn "No root or passwordless sudo access; please install these packages if imports fail: ${packages[*]}"
    return 0
  fi

  info "Installing common system packages for OpenCV, Pygame, video export, and sync tools."
  if ! DEBIAN_FRONTEND=noninteractive "${apt_cmd[@]}" update; then
    warn "apt-get update failed; continuing with Python dependency setup."
    return 0
  fi

  if ! DEBIAN_FRONTEND=noninteractive "${apt_cmd[@]}" install -y "${packages[@]}"; then
    warn "apt-get install failed; continuing. Missing shared libraries may still break OpenCV/Pygame."
  fi
}

create_or_activate_env() {
  activate_conda_base

  if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    info "Conda env ${ENV_NAME} already exists; reusing it."
  else
    info "Creating conda env ${ENV_NAME} with Python ${PYTHON_VERSION}."
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
  fi

  conda activate "${ENV_NAME}"
  info "Using python: $(command -v python)"
  python - <<'PY'
import sys
major_minor = sys.version_info[:2]
print(f"Python version: {sys.version}")
if major_minor != (3, 10):
    raise SystemExit(f"Expected Python 3.10, got {major_minor[0]}.{major_minor[1]}")
PY
}

install_python_dependencies() {
  TMP_REQ="$(mktemp)"

  info "Pinning pip/setuptools/wheel for gym==0.21.0 compatibility."
  python -m pip install \
    -i "${PYPI_INDEX_URL}" \
    --trusted-host "${PYPI_TRUSTED_HOST}" \
    --upgrade \
    "pip==23.0" \
    "setuptools==65.5.0" \
    "wheel==0.38.4"

  info "Installing PyTorch CUDA 12.1 wheels from Aliyun mirror first."
  if ! python -m pip install \
    -i "${PYPI_INDEX_URL}" \
    --trusted-host "${PYPI_TRUSTED_HOST}" \
    -f "${ALIYUN_TORCH_FIND_LINKS}" \
    "torch==2.1.2+cu121" \
    "torchvision==0.16.2+cu121"; then
    warn "Aliyun PyTorch wheel mirror failed; falling back to official PyTorch CUDA 12.1 index."
    python -m pip install \
      -i "${PYPI_INDEX_URL}" \
      --trusted-host "${PYPI_TRUSTED_HOST}" \
      --extra-index-url "${PYTORCH_CU121_INDEX_URL}" \
      "torch==2.1.2+cu121" \
      "torchvision==0.16.2+cu121"
  fi

  awk 'BEGIN {IGNORECASE=1} !/^[[:space:]]*(torch|torchvision)==/' "${REPO_ROOT}/requirements.txt" > "${TMP_REQ}"
  info "Installing official dependencies except torch/torchvision via temporary requirements: ${TMP_REQ}"
  python -m pip install \
    -i "${PYPI_INDEX_URL}" \
    --trusted-host "${PYPI_TRUSTED_HOST}" \
    -r "${TMP_REQ}"
}

install_atari_roms() {
  info "Installing Atari ROMs with AutoROM --accept-license."
  if command -v AutoROM >/dev/null 2>&1; then
    if ! AutoROM --accept-license; then
      warn "AutoROM command failed. Atari environments may fail until ROMs are installed."
    fi
  elif python -m AutoROM --help >/dev/null 2>&1; then
    if ! python -m AutoROM --accept-license; then
      warn "python -m AutoROM failed. Atari environments may fail until ROMs are installed."
    fi
  else
    warn "AutoROM was not found after dependency installation. Check gym[accept-rom-license] installation."
  fi
}

run_smoke_check() {
  info "Running import, CUDA, and BreakoutNoFrameskip-v4 smoke checks."
  python - <<'PY'
import importlib

modules = [
    "ale_py",
    "cv2",
    "einops",
    "gym",
    "hydra",
    "matplotlib",
    "numpy",
    "pygame",
    "torch",
    "torchvision",
    "tqdm",
    "wandb",
]

for name in modules:
    mod = importlib.import_module(name)
    version = getattr(mod, "__version__", "unknown")
    print(f"{name}: {version}")

import gym
import torch

print(f"torch.version.cuda: {torch.version.cuda}")
print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the conda env.")
print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")

env = gym.make("BreakoutNoFrameskip-v4")
obs = env.reset()
print(f"gym Breakout reset observation shape: {getattr(obs, 'shape', None)}")
env.close()
PY
}

main() {
  trap cleanup EXIT

  cd "${REPO_ROOT}"
  info "Repo root: ${REPO_ROOT}"
  info "Log file: ${LOG_FILE}"

  install_system_packages_if_possible
  create_or_activate_env
  install_python_dependencies
  install_atari_roms
  run_smoke_check

  info "Environment setup completed successfully."
  info "Activate it with: conda activate ${ENV_NAME}"
}

main "$@"
