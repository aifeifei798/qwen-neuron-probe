#!/usr/bin/env bash
# 神经元热力探针 · 启动服务
# 用法: ./serve.sh   （首次启动模型加载约 10–30 秒，看状态栏）
#   PORT=9000 ./serve.sh   可改端口；QWEN_MODEL_PATH 可改模型路径
cd "$(dirname "$0")"
PORT="${PORT:-8790}"
exec .venv/bin/python -m uvicorn server.main:app --host 127.0.0.1 --port "$PORT"
