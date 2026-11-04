"""抑制退化机制实验：分层/分级钳位 + 生成期激活观测。

用法: .venv/bin/python experiments/suppress_study.py
（会加载一份独立模型实例，约 9GB 显存；不影响正在运行的服务）
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/feifei/develop/app/qwen_hidden")
from server.probe import NeuronProbe  # noqa: E402

QUESTION = "什么是注意力机制？"
N_NEW = 128
L = 36

p = NeuronProbe()
p.load()
tok, model, dev = p.tok, p.model, p.device
prompt, _, ids = p._build_ids(QUESTION)
print(f"问题: {QUESTION}  输入 {ids.shape[0]} token  每版生成 {N_NEW} token\n")

p._capture_on = False  # 观测/实验期间关掉探针的逐层拷贝


def gen(cfg, n=N_NEW, observe=None):
    """cfg: 每层阈值(float)或 None=不钳位；observe: 每层接收 max|v| 的列表。"""
    handles = []
    if observe is not None:
        for i in range(L):
            def obs(module, args, kwargs, _i=i):
                observe[_i].append(float(args[0].abs().max()))
            handles.append(model.model.layers[i].mlp.down_proj.register_forward_pre_hook(
                obs, with_kwargs=True))
    if cfg is not None:
        for i, thr in enumerate(cfg):
            if thr is None:
                continue
            def clamp(module, args, kwargs, _t=thr):
                x = args[0]
                return (x.clamp(-_t, _t),) + args[1:], kwargs
            handles.append(model.model.layers[i].mlp.down_proj.register_forward_pre_hook(
                clamp, with_kwargs=True))
    try:
        with torch.no_grad():
            out = model.generate(input_ids=ids.unsqueeze(0).to(dev), max_new_tokens=n,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(out[0][ids.shape[0]:], skip_special_tokens=True)
    finally:
        for h in handles:
            h.remove()


# 预填充捕获一次，三档阈值同源
p._capture_on = True
p._acts = [None] * L
with torch.no_grad():
    model(input_ids=ids.unsqueeze(0).to(dev))
acts = [a.astype(np.float32) for a in p._acts]
p._capture_on = False
thr995 = [float(np.quantile(np.abs(a), 0.995)) for a in acts]
thr999 = [float(np.quantile(np.abs(a), 0.999)) for a in acts]
thr9999 = [float(np.quantile(np.abs(a), 0.9999)) for a in acts]
print("钳位基准(99.5%%):  L0=%.2f L12=%.2f L24=%.2f L35=%.2f" %
      (thr995[0], thr995[12], thr995[24], thr995[35]))
print("钳位基准(99.99%%): L35=%.2f" % thr9999[35])

# 预填充期的固定 massive 神经元（每层 |v| 最大的神经元，看是否跨问题一致由前面已验证）
deep_top = sorted(range(9728), key=lambda n: -abs(acts[35][:, n]).max())[:3]
print("L35 预填充期最大激活神经元:", [(n, round(float(max(abs(acts[35][:, n]))), 1)) for n in deep_top], "\n")

variants = [
    ("基线 · 无抑制",              None),
    ("全层 @99.5%   (复现塌缩)",   thr995),
    ("全层 @99.9%",                thr999),
    ("全层 @99.99% (只钳最大值)",  thr9999),
    ("仅浅层 L00-17 @99.5%",       [thr995[i] if i < 18 else None for i in range(L)]),
    ("仅深层 L18-35 @99.5%",       [thr995[i] if i >= 18 else None for i in range(L)]),
]
for name, cfg in variants:
    text = gen(cfg)
    head = text[:90].replace("\n", "\\n")
    closed = "</think>" in text
    print(f"{name:24s} → 收尾={closed} | {head!r}")

# 正常生成时，每层激活是否常态超出"读题基准"
stats = [[] for _ in range(L)]
gen(None, observe=stats)
print("\n正常生成时各层 max|v| 中位数 vs 读题基准(99.5%)，以及超出基准的步数占比(前 6 名):")
rows = []
for i in range(L):
    med = float(np.median(stats[i]))
    frac = float(np.mean([m > thr995[i] for m in stats[i]]))
    rows.append((frac, i, med, thr995[i]))
for frac, i, med, base in sorted(rows, reverse=True)[:6]:
    print(f"  L{i:02d}: 生成期 max|v| 中位={med:7.2f}  基准={base:7.2f}  超出基准步数占比={frac*100:5.1f}%")
