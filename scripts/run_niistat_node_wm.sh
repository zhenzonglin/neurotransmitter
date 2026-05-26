#!/usr/bin/env bash
set -euo pipefail

project_dir="/home/zhenzong2/analysis/neurotransmitter"
cd "${project_dir}"

# 用脚本封装MATLAB调用，避免命令行引号问题
matlab -nodisplay -nosplash -r "try, run('${project_dir}/scripts/run_niistat_node_wm.m'), catch e, disp(getReport(e,'extended')), exit(1), end, exit(0)"
