#!/usr/bin/env bash
set -euo pipefail

config_path="config/dat_config.yaml"
dry_run=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      config_path="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

# 从YAML配置转为MATLAB可读取的环境变量
eval "$(
  python - "${config_path}" <<'PY'
from pathlib import Path
import shlex
import sys
import yaml

config_path = Path(sys.argv[1]).resolve()
with config_path.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

project_dir = Path(config["project_dir"]).resolve()
values = {
    "NT_CONFIG": config_path,
    "NT_PROJECT_DIR": project_dir,
    "NT_NIISTAT_REPO": project_dir / config["niistat"]["repo_dir"],
    "NT_NODE_INPUT": project_dir / config["outputs"]["node_dir"] / "dat_node_niistat_input.mat",
    "NT_NODE_OUTPUT": project_dir / config["outputs"]["node_dir"] / "niistat_node_results",
    "NT_NODE_HDR": project_dir / config["atlases"]["outputs"]["atlas4s156_2mm"],
    "NT_WM_INPUT": project_dir / config["outputs"]["wm_voxel_dir"] / "dat_wm_voxel_niistat_input.mat",
    "NT_WM_OUTPUT": project_dir / config["outputs"]["wm_voxel_dir"] / "niistat_wm_results",
    "NT_WM_HDR": project_dir / config["atlases"]["outputs"]["dat_wm_mask_2mm"],
    "NT_P_THRESH": config["niistat"]["p_thresh"],
    "NT_NUM_PERMUTE": config["niistat"]["num_permute"],
}
for key, value in values.items():
    print(f"export {key}={shlex.quote(str(value))}")
PY
)"
project_dir="${NT_PROJECT_DIR}"
cd "${project_dir}"

if [[ "${dry_run}" == "true" ]]; then
  env | grep '^NT_' | sort
  exit 0
fi

# 用脚本封装MATLAB调用，避免命令行引号问题
matlab -nodisplay -nosplash -r "try, run('${project_dir}/scripts/run_niistat_node_wm.m'), catch e, disp(getReport(e,'extended')), exit(1), end, exit(0)"
