#!/usr/bin/env bash
# 神经元热力探针 · 启动服务
# 用法: ./serve.sh   （首次启动模型加载约 10–30 秒，看状态栏）
cd "$(dirname "$0")"
exec .venv/bin/python -m uvicorn server.main:app --host 127.0.0.1 --port 8790
