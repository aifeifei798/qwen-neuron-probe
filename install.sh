#!/usr/bin/env bash
# Qwen3-4B 神经元激活探针 · 环境安装
# 用法: ./install.sh [--mirror huawei|tuna|pypi] [--cpu]
#   --mirror huawei  国内网络用华为云 PyPI 镜像（tuna 同理），默认 pypi
#   --cpu            无 NVIDIA GPU 时装 CPU 版 torch（默认按 nvidia-smi 自动判断）
set -euo pipefail
cd "$(dirname "$0")"

MIRROR="pypi"; FORCE_CPU=0
while [ $# -gt 0 ]; do
  case "$1" in
    --mirror) MIRROR="$2"; shift 2 ;;
    --cpu)    FORCE_CPU=1; shift ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

INDEX_ARGS=()
case "$MIRROR" in
  pypi)   ;;
  huawei) INDEX_ARGS=(--index-url "https://mirrors.huaweicloud.com/repository/pypi/simple") ;;
  tuna)   INDEX_ARGS=(--index-url "https://pypi.tuna.tsinghua.edu.cn/simple") ;;
  *) echo "未知镜像: $MIRROR"; exit 1 ;;
esac

if ! command -v python3 >/dev/null; then
  echo "✗ 需要 python3（建议 ≥3.10）"; exit 1
fi

# 创建虚拟环境：优先 uv（快），退回 python3 -m venv
if command -v uv >/dev/null 2>&1; then
  echo "▸ 用 uv 创建 .venv"
  [ -d .venv ] || uv venv .venv
  install() { uv pip install --python .venv/bin/python "$@"; }
else
  echo "▸ 用 python3 -m venv 创建 .venv（想要更快可先安装 uv）"
  [ -d .venv ] || python3 -m venv .venv
  install() { .venv/bin/pip install "$@"; }
fi

# torch：CUDA 自动探测；CPU 强制或无 GPU 时用 CPU 轮子
if [ "$FORCE_CPU" = 0 ] && command -v nvidia-smi >/dev/null 2>&1; then
  echo "▸ 检测到 NVIDIA GPU，安装 CUDA 版 torch"
  install torch "${INDEX_ARGS[@]}"
else
  echo "▸ 安装 CPU 版 torch（--cpu 或未检测到 NVIDIA GPU）"
  install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "▸ 安装 transformers / fastapi / uvicorn / numpy"
install transformers fastapi "uvicorn[standard]" numpy "${INDEX_ARGS[@]}"

echo "▸ 验证导入"
.venv/bin/python -c "import torch, transformers, fastapi, uvicorn, numpy; print('  torch', torch.__version__, '| CUDA', torch.cuda.is_available())"

MODEL_PATH="${QWEN_MODEL_PATH:-/home/feifei/develop/models/Qwen3-4B-Thinking-2507}"
if [ ! -d "$MODEL_PATH" ]; then
  echo "⚠ 模型目录不存在: $MODEL_PATH"
  echo "  启动前请设置: export QWEN_MODEL_PATH=/path/to/Qwen3-4B-Thinking-2507"
  echo "  （或修改 server/probe.py 的默认路径）"
fi

echo
echo "✓ 安装完成。下一步:"
echo "    export QWEN_MODEL_PATH=/path/to/Qwen3-4B-Thinking-2507   # 如路径非默认"
echo "    ./serve.sh    # http://127.0.0.1:8790 （PORT 环境变量可改端口）"
