#!/usr/bin/env bash
set -euo pipefail

# 用Docker版DSI Studio，避免WSL图形依赖
docker run --rm \
  -e QT_PLUGIN_PATH=/opt/qt6/6.5.0/gcc_64/plugins \
  -e QT_QPA_PLATFORM=minimal \
  -v /home/zhenzong2:/home/zhenzong2 \
  -v /tmp:/tmp \
  dsistudio/dsistudio:latest \
  dsi_studio "$@"
