#!/bin/bash
export BUILD_ENVIRONMENT=local-rocm-gfx942-wheel
export INSTALL_WHEEL=0
export PYTORCH_ROCM_ARCH=gfx942
set -ex -o pipefail
pip uninstall -y torch
rm -rf dist/

# Incremental wheel rebuild for local libkineto development.
#
# This script is intentionally narrower than .ci/pytorch/build.sh:
# - it assumes a successful full build already populated build/
# - it is meant for edits under third_party/kineto/**
# - it avoids setup.py clean so Ninja/CMake can rebuild incrementally
# - it still repackages the torch wheel so the resulting whl includes
#   the updated kineto bits and any dependent relinks

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# shellcheck source=./common-build.sh
source "$(dirname "${BASH_SOURCE[0]}")/common-build.sh"

script_dir="$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit ; pwd -P )"
repo_root="$( cd "${script_dir}/../.." || exit ; pwd -P )"
cd "${repo_root}"

BUILD_DIR="build"
KINETO_TARGET="${KINETO_TARGET:-kineto}"
INSTALL_WHEEL="${INSTALL_WHEEL:-1}"
ENSURE_CI_NUMPY="${ENSURE_CI_NUMPY:-1}"
UNIQUE_BUILD_METADATA="${UNIQUE_BUILD_METADATA:-1}"

if [[ -z "${BUILD_ENVIRONMENT:-}" ]]; then
  fatal "BUILD_ENVIRONMENT must be set"
fi

if [[ "${BUILD_ENVIRONMENT}" == *-mobile-*build* ]]; then
  fatal "Mobile builds are not supported by the incremental kineto wheel script"
fi

if [[ "${BUILD_ENVIRONMENT}" == *-android* ]]; then
  fatal "Android builds are not supported by the incremental kineto wheel script"
fi

if [[ "${BUILD_ENVIRONMENT}" == *-bazel-* ]]; then
  fatal "Bazel builds are not supported by the incremental kineto wheel script"
fi

if [[ "${BUILD_ENVIRONMENT}" == *libtorch* ]]; then
  fatal "This script only repackages Python wheels, not libtorch artifacts"
fi

if [[ "${BUILD_ENVIRONMENT}" == *xla* ]]; then
  fatal "XLA builds need extra setup that this incremental script does not replicate"
fi

if [[ "${USE_SPLIT_BUILD:-false}" == "true" ]]; then
  fatal "USE_SPLIT_BUILD=true is not supported by this incremental script"
fi

if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  fatal "Missing ${BUILD_DIR}/CMakeCache.txt. Run .ci/pytorch/build.sh once first to create the initial build cache."
fi

if [[ ! -f "${BUILD_DIR}/build.ninja" && ! -f "${BUILD_DIR}/Makefile" ]]; then
  fatal "Missing native build files under ${BUILD_DIR}. Run .ci/pytorch/build.sh once first."
fi

if ! grep -Eq '^USE_KINETO:(BOOL|STRING)=ON$' "${BUILD_DIR}/CMakeCache.txt"; then
  fatal "The existing build cache was configured with USE_KINETO=OFF"
fi

if [[ "${UNIQUE_BUILD_METADATA}" == "1" ]]; then
  full_sha="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  short_sha="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
  build_stamp="${KINETO_LOCAL_BUILD_ID:-$(date -u +%Y%m%d%H%M%S)}"
  base_version="$(tr -d '\n' < version.txt)"

  if [[ -z "${PYTORCH_BUILD_VERSION:-}" ]]; then
    export PYTORCH_BUILD_VERSION="${base_version}+kineto.${short_sha}.${build_stamp}"
    export PYTORCH_BUILD_NUMBER=1
  fi

  if [[ -z "${PYTORCH_BUILD_GIT_VERSION:-}" ]]; then
    export PYTORCH_BUILD_GIT_VERSION="${full_sha}.kineto.${build_stamp}"
  fi
fi

echo "Python version:"
python --version

echo "GCC version:"
gcc --version || true

echo "CMake version:"
cmake --version

if [[ -n "${PYTORCH_BUILD_VERSION:-}" ]]; then
  echo "Wheel version: ${PYTORCH_BUILD_VERSION}"
fi
if [[ -n "${PYTORCH_BUILD_GIT_VERSION:-}" ]]; then
  echo "Wheel git_version: ${PYTORCH_BUILD_GIT_VERSION}"
fi

# Keep the cache reusable and avoid accidental full reconfigure requests
# from the shell environment.
export CMAKE_FRESH=0

# Match build.sh for environments that benefit from the cache wrappers.
if [[ -d /opt/cache/lib ]]; then
  export PATH="/opt/cache/lib:${PATH}"
fi

if [[ -z "${MAX_JOBS:-}" ]]; then
  if [[ "${BUILD_ENVIRONMENT}" == *rocm* ]]; then
    export MAX_JOBS="$(($(nproc) - 1))"
  elif [[ "${BUILD_ENVIRONMENT}" == *cuda* ]] && command -v sccache >/dev/null 2>&1; then
    export MAX_JOBS="$(($(nproc) - 1))"
  fi
fi

if [[ "${ENSURE_CI_NUMPY}" == "1" && "${BUILD_ENVIRONMENT}" != *rocm* ]]; then
  if ! python - <<'PY'
import sys
try:
    import numpy
except Exception:
    sys.exit(1)
sys.exit(0 if numpy.__version__ == "2.0.2" else 1)
PY
  then
    python -mpip install numpy==2.0.2
  fi
fi

cmake_build_args=(--build "${BUILD_DIR}" --target "${KINETO_TARGET}")
if [[ -n "${MAX_JOBS:-}" ]]; then
  cmake_build_args+=(-j "${MAX_JOBS}")
fi

echo "Incrementally rebuilding ${KINETO_TARGET}"
cmake "${cmake_build_args[@]}"

echo "Packaging torch wheel without cleaning ${BUILD_DIR}"
WERROR="${WERROR:-0}" python setup.py bdist_wheel

shopt -s nullglob
wheels=(dist/*.whl)
if [[ ${#wheels[@]} -eq 0 ]]; then
  fatal "No wheel was produced under dist/"
fi

if [[ "${INSTALL_WHEEL}" == "1" ]]; then
  pip_install_whl "${wheels[@]}"
  python - <<'PY'
import torch
print(f"Installed torch.__version__={torch.__version__}")
print(f"Installed torch.version.git_version={torch.version.git_version}")
PY
fi

mkdir -p dist
if [[ -f "${BUILD_DIR}/.ninja_log" ]]; then
  cp "${BUILD_DIR}/.ninja_log" dist
fi

print_sccache_stats
