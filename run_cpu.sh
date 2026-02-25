#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ROOT_DIR}/sponge_cpu_mainpath"

if [[ ! -x "${BIN}" ]]; then
    echo "[cpusponge] binary missing, building sponge_cpu_mainpath ..."
    (cd "${ROOT_DIR}" && make -f Makefile.cpu -j"$(nproc)" sponge_cpu_mainpath)
fi

MDIN_PATH=""
if [[ $# -ge 2 && "$1" == "-mdin" ]]; then
    MDIN_PATH="$2"
elif [[ $# -ge 1 ]]; then
    MDIN_PATH="$1"
else
    echo "Usage: $0 -mdin <mdin_path>"
    exit 2
fi

[[ -f "${MDIN_PATH}" ]] || { echo "mdin not found: ${MDIN_PATH}"; exit 2; }

# Runtime hardening: keep deterministic CPU compatibility mode unless user overrides.
if [[ -z "${SPONGE_CPU_OMP_LAUNCH+x}" ]]; then
    export SPONGE_CPU_OMP_LAUNCH=0
fi

if [[ -d "${ROOT_DIR}/covid-tip4p" ]]; then
    sed -i 's/\r$//' "${ROOT_DIR}"/covid-tip4p/* || true
fi

exec "${BIN}" -mdin "${MDIN_PATH}"
