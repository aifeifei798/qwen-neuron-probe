"""人格方向调向实验：礼貌客服（不道歉）方向提取 + α 扫描。

用法: .venv/bin/python experiments/steer_study.py
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/feifei/develop/app/qwen_hidden")
from server.probe import NeuronProbe  # noqa: E402

L = 36
SYS_A = "你是一名彬彬有礼、温和专业的客服。直接帮用户解决问题，不要使用'抱歉''对不起''不好意思'等道歉用语。"
SYS_B = "你是一名卑微的客服。遇到用户抱怨要反复道歉，不停地说'抱歉''对不起'。"
QUESTIONS = [
    "我买你们的音箱用了两天就坏了，你们卖的是什么破烂货？",
    "订单三天了还没发货，你们客服是摆设吗？",
]
APOLOGY_WORDS = ["抱歉", "对不起", "不好意思", "道歉"]
TAIL = 16  # 取预填充末尾 N 个 token 求均值（读完问题、准备开口的状态）

p = NeuronProbe()
p.load()
tok, model, dev = p.tok, p.model, p.device
p._capture_on = False


def capture(system_prompt, question):
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"][0]
    p._capture_on = True
    p._acts = [None] * L
    with torch.no_grad():
        model(input_ids=ids.unsqueeze(0).to(dev))
    p._capture_on = False
    return p._acts  # list of (seq, 9728) fp16 numpy


def gen_steer(ids, deltas_eff, n):
    handles = []
    if deltas_eff is not None:
        for i, d in enumerate(deltas_eff):
            if d is None:
                continue
            dt = torch.tensor(d, device=dev, dtype=torch.bfloat16)
            def hook(module, args, kwargs, _d=dt):
                return (args[0] + _d,) + args[1:], kwargs
            handles.append(model.model.layers[i].mlp.down_proj.register_forward_pre_hook(
                hook, with_kwargs=True))
    try:
        with torch.no_grad():
            out = model.generate(input_ids=ids.unsqueeze(0).to(dev), max_new_tokens=n,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(out[0][ids.shape[0]:], skip_special_tokens=True)
    finally:
        for h in handles:
            h.remove()


def apology_count(text):
    return sum(text.count(w) for w in APOLOGY_WORDS)


# ---- 方向提取 ----
q0 = QUESTIONS[0]
acts_a = capture(SYS_A, q0)
acts_b = capture(SYS_B, q0)
deltas, norms, stds = [], [], []
for i in range(L):
    a = acts_a[i].astype(np.float32)[-TAIL:].mean(axis=0)
    b = acts_b[i].astype(np.float32)[-TAIL:].mean(axis=0)
    d = a - b
    std = float(acts_b[i].astype(np.float32).std())
    norms.append(float(np.linalg.norm(d)))
    stds.append(std)
    deltas.append(d / max(np.linalg.norm(d), 1e-9))

top = sorted(range(L), key=lambda i: -norms[i])[:5]
print("方向强度 ||Δ|| 每层 Top5（人格信息集中的层）:")
for i in top:
    print(f"  L{i:02d}: ||Δ||={norms[i]:7.3f}  层内激活std={stds[i]:.3f}")
print()

# ---- 生成: ids 用"无 system"的纯问题，语气只由调向决定 ----
def plain_ids(question):
    prompt = tok.apply_chat_template([{"role": "user", "content": question}],
                                     tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    return enc["input_ids"][0], len(enc["input_ids"][0])

print("=== 实验 1：基线（无 system，无调向）===")
for q in QUESTIONS:
    ids, n = plain_ids(q)
    t = gen_steer(ids, None, 160)
    print(f"[道歉x{apology_count(t)}] {t[:110].replace(chr(10), ' ')}")
print()

print("=== 实验 2：只给礼貌 system prompt（不调向，对照组）===")
ids_a, _ = plain_ids(q0)  # 占位防误解
for q in QUESTIONS:
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYS_A}, {"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    t = gen_steer(enc["input_ids"][0], None, 160)
    print(f"[道歉x{apology_count(t)}] {t[:110].replace(chr(10), ' ')}")
print()

print("=== 实验 3：调向器 α 扫描（无 system prompt）===")
ids, _ = plain_ids(q0)
for alpha in [1, 2, 4, 8, 16]:
    eff = [deltas[i] * alpha * stds[i] for i in range(L)]
    t = gen_steer(ids, eff, 160)
    print(f"α={alpha:<3}[道歉x{apology_count(t)}] {t[:110].replace(chr(10), ' ')}")
print()

print("=== 实验 4：第二个问题复测（方向泛化）===")
ids2, _ = plain_ids(QUESTIONS[1])
for alpha in [2, 4]:
    eff = [deltas[i] * alpha * stds[i] for i in range(L)]
    t = gen_steer(ids2, eff, 160)
    print(f"α={alpha:<3}[道歉x{apology_count(t)}] {t[:110].replace(chr(10), ' ')}")
