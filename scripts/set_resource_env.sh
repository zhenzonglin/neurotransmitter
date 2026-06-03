#!/usr/bin/env bash
set -euo pipefail

config_path="${1:-config/dat_config.yaml}"

read_yaml_value() {
  local key="$1"
  python - "$config_path" "$key" <<'PY'
import sys
import yaml

path, dotted = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
value = data
for part in dotted.split("."):
    value = value.get(part, {}) if isinstance(value, dict) else {}
if isinstance(value, dict) or value is None:
    print("")
else:
    print(value)
PY
}

blas_threads="$(read_yaml_value resources.blas_threads)"
lqt_dsi_cpus="$(read_yaml_value resources.lqt_dsi_cpus)"
lqt_dsi_memory="$(read_yaml_value resources.lqt_dsi_memory)"

blas_threads="${blas_threads:-4}"
lqt_dsi_cpus="${lqt_dsi_cpus:-3}"
lqt_dsi_memory="${lqt_dsi_memory:-4500m}"

# 限制底层数值库线程，避免多进程时过度占用CPU
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$blas_threads}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$blas_threads}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$blas_threads}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$blas_threads}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$blas_threads}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"

# DSI Studio Docker单容器资源
export NT_DSI_CPUS="${NT_DSI_CPUS:-$lqt_dsi_cpus}"
export NT_DSI_MEMORY="${NT_DSI_MEMORY:-$lqt_dsi_memory}"

echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
echo "MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS}"
echo "NT_DSI_CPUS=${NT_DSI_CPUS}"
echo "NT_DSI_MEMORY=${NT_DSI_MEMORY}"
