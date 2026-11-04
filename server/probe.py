"""Qwen3-4B-Thinking-2507 神经元激活探针。

原理：在每层 Transformer 的 ``mlp.down_proj`` 上挂 forward pre-hook，
捕获它的输入 —— 即 SwiGLU 之后的 MLP 中间激活 (seq_len, intermediate_size)。
中间维度上的每一个分量就是一个"神经元"，Qwen3-4B 每层 9728 个、共 36 层。
"""
from __future__ import annotations

import base64
import threading
import time
from collections import OrderedDict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/feifei/develop/models/Qwen3-4B-Thinking-2507"
MAX_INPUT_TOKENS = 1024          # 输入 token 上限，防止激活缓存爆内存
MAX_CACHED_RESULTS = 2           # 内存里保留最近 N 次分析（每次最多约 700MB）
AGG_KINDS = ("last", "mean", "max")

# 调向器：人格方向 = 两种人设预填充激活之差（激活末尾 TAIL 个 token 求均值）
STEER_PRESET = {
    "label": "礼貌客服 · 不道歉",
    # 安全硬规则：人身安全场景下风格约束让位（router_case 实验 F 变体验证）
    "sys_a": ("你是一名彬彬有礼、温和专业的客服。直接帮用户解决问题，"
              "不要使用'抱歉''对不起''不好意思'等道歉用语。"
              "最高优先级规则：若用户描述涉及人身安全（起火、爆炸、冒烟、烧焦、触电、受伤等），"
              "先确认安全——请用户立即断电、远离设备、必要时联系消防或紧急售后，"
              "此场景下一切风格约束让位于安全指引。"
              "对情绪激动的用户先共情安抚（共情不等于道歉），再给行动步骤。"),
    "sys_b": "你是一名卑微的客服。遇到用户抱怨要反复道歉，不停地说'抱歉''对不起'。",
}
STEER_TAIL = 16
STEER_ALPHA_RANGE = (0.0, 4.0)   # α=0 = 纯 prompt 对照（实测最稳）；>4 常见退化
STEER_DEEP_FROM = 31             # 实验实测：全层注入破坏语义理解，默认只注深层


class NeuronProbe:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.tok = None
        self.model = None
        self.ready = False
        self.error: str | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()          # 串行化前向计算
        self._acts: list[np.ndarray | None] = []
        self._handles: list = []
        self._capture_on = True                # 生成阶段关掉捕获，省 GPU→CPU 拷贝
        self._clamp_handles: list = []         # 抑制器 hooks
        self._clamp_acc: list | None = None    # 每层钳位元素计数（GPU 张量累加）
        # rid -> {"acts": [np.float16 (seq, 9728) × 36], "tokens": [...]}
        self._results: OrderedDict[str, dict] = OrderedDict()

    # ---------- 模型加载 ----------

    def load(self) -> None:
        """在后台线程调用；完成后置 ready 或 error。"""
        try:
            t0 = time.time()
            self.tok = AutoTokenizer.from_pretrained(self.model_path)
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, dtype=dtype)
            self.model.to(self.device)
            self.model.eval()
            self._install_hooks()
            self.ready = True
            print(f"[probe] 模型加载完成 device={self.device} 用时 {time.time() - t0:.1f}s")
        except Exception as exc:  # noqa: BLE001 —— 加载失败要把原因带回给前端
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[probe] 模型加载失败: {self.error}")

    def status(self) -> dict:
        layers = int(self.model.config.num_hidden_layers) if self.model else 36
        inter = int(self.model.config.intermediate_size) if self.model else 9728
        hidden = int(self.model.config.hidden_size) if self.model else 2560
        heads = int(self.model.config.num_attention_heads) if self.model else 32
        info = {
            "ready": self.ready,
            "loading": not self.ready and self.error is None,
            "error": self.error,
            "device": self.device,
            "gpu_name": torch.cuda.get_device_name(0) if self.device == "cuda" else None,
            "model": self.model_path.rsplit("/", 1)[-1],
            "layers": layers,
            "neurons": inter,
            "hidden_size": hidden,
            "num_heads": heads,
            "max_input_tokens": MAX_INPUT_TOKENS,
        }
        if self.device == "cuda":
            free_b, total_b = torch.cuda.mem_get_info()
            info["vram_free_gb"] = round(free_b / 2**30, 1)
            info["vram_total_gb"] = round(total_b / 2**30, 1)
        return info

    # ---------- hooks ----------

    def _install_hooks(self) -> None:
        for idx, block in enumerate(self.model.model.layers):
            def pre_hook(module, args, kwargs, _idx=idx):
                if not self._capture_on:
                    return
                # down_proj 的输入即该层 MLP 神经元激活: (1, seq, 9728)
                x = args[0] if args else kwargs.get("input")
                self._acts[_idx] = x[0].detach().to("cpu", torch.float16).numpy()
            self._handles.append(
                block.mlp.down_proj.register_forward_pre_hook(pre_hook, with_kwargs=True)
            )

    # ---------- 抑制器 ----------

    def _set_clamp(self, thresholds: list[float] | None) -> None:
        """装/卸生成干预 hooks（钳位或调向共用一个挂点列表）。"""
        for h in self._clamp_handles:
            h.remove()
        self._clamp_handles = []
        self._clamp_acc = None
        if thresholds is None:
            return
        n_layers = len(thresholds)
        self._clamp_acc = [torch.zeros((), device=self.device, dtype=torch.long)
                           for _ in range(n_layers)]
        acc = self._clamp_acc
        for idx, thr in enumerate(thresholds):
            def hook(module, args, kwargs, _i=idx, _t=float(thr)):
                x = args[0] if args else kwargs.get("input")
                hi = x > _t
                lo = x < -_t
                acc[_i] = acc[_i] + hi.sum() + lo.sum()
                xc = x.clamp(-_t, _t)
                if args:
                    return (xc,) + args[1:], kwargs
                return args, {**kwargs, "input": xc}
            self._clamp_handles.append(
                self.model.model.layers[idx].mlp.down_proj.register_forward_pre_hook(
                    hook, with_kwargs=True)
            )

    def _set_steer(self, deltas_eff: list) -> None:
        """调向 hooks：每层 MLP 输入加上 α 缩放后的人格方向向量。"""
        for idx, d in enumerate(deltas_eff):
            if d is None:
                continue
            dt = torch.tensor(d, device=self.device, dtype=torch.bfloat16)
            def hook(module, args, kwargs, _d=dt):
                x = args[0] if args else kwargs.get("input")
                if args:
                    return (x + _d,) + args[1:], kwargs
                return args, {**kwargs, "input": x + _d}
            self._clamp_handles.append(
                self.model.model.layers[idx].mlp.down_proj.register_forward_pre_hook(
                    hook, with_kwargs=True)
            )

    def _capture_prompt(self, system: str, question: str) -> list:
        """带 system 人设的预填充激活（每层 (seq, 9728) fp16）。"""
        prompt = self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True)
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False)
        if enc["input_ids"].shape[1] > MAX_INPUT_TOKENS:
            raise ValueError("问题太长：加上人设后超过 token 上限，请缩短")
        self._capture_on = True
        self._acts = [None] * len(self.model.model.layers)
        with torch.no_grad():
            self.model(input_ids=enc["input_ids"][0].unsqueeze(0).to(self.device))
        self._capture_on = False
        if any(a is None for a in self._acts):
            raise RuntimeError("有层没捕获到激活（hook 数量或模型结构和预期不符）")
        return self._acts

    # ---------- 分析 ----------

    def analyze(self, question: str, agg: str = "last") -> dict:
        if not self.ready:
            raise RuntimeError("模型还没加载完成")
        if agg not in AGG_KINDS:
            raise ValueError(f"agg 只能是 {'/'.join(AGG_KINDS)}")
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空")

        with self._lock:
            prompt, enc, ids = self._build_ids(question)
            offsets = enc["offset_mapping"][0].tolist()
            self._acts = [None] * len(self.model.model.layers)
            with torch.no_grad():
                self.model(input_ids=ids.unsqueeze(0).to(self.device))
            if any(a is None for a in self._acts):
                raise RuntimeError("有层没捕获到激活（hook 数量或模型结构和预期不符）")

            acts = self._acts  # 36 × (seq, 9728) float16，已在 CPU
            tokens = self._token_views(ids.tolist(), offsets, prompt, question)
            rid = self._cache(acts, tokens)
            payload = self._build_payload(rid, agg=agg, token=None)
            payload["prompt_preview"] = prompt
            return payload

    def _build_ids(self, question: str, system: str | None = None):
        """套 chat 模板并切 token；超限直接报错。"""
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": question}]
        prompt = self.tok.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = self.tok(prompt, return_tensors="pt",
                       return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"][0]
        if ids.shape[0] > MAX_INPUT_TOKENS:
            raise ValueError(
                f"这段话会被切成 {ids.shape[0]} 个 token，"
                f"超过上限 {MAX_INPUT_TOKENS}：请缩短问题再分析"
            )
        return prompt, enc, ids

    # ---------- 对比聊天（抑制前 / 抑制后） ----------

    def _generate(self, ids, max_new_tokens: int, clamp: list[float] | None = None,
                  steer: list | None = None):
        """贪心解码；clamp（钳位）与 steer（调向）二选一。"""
        self._capture_on = False
        self._set_clamp(None)
        if clamp is not None:
            self._set_clamp(clamp)
        elif steer is not None:
            self._set_steer(steer)
        try:
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=ids.unsqueeze(0).to(self.device),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                )
            text = self.tok.decode(out[0][ids.shape[0]:], skip_special_tokens=True)
            n_new = int(out.shape[1] - ids.shape[0])
            counts = ([int(c.item()) for c in self._clamp_acc] if clamp else None)
            return text, n_new, counts
        finally:
            self._set_clamp(None)
            self._capture_on = True

    def _steer_direction(self, question: str, alpha: float) -> tuple[list, dict]:
        """人格方向：礼貌客服 vs 卑歉客服 的预填充激活差（单位化后按层 std × α 缩放）。"""
        acts_a = self._capture_prompt(STEER_PRESET["sys_a"], question)
        acts_b = self._capture_prompt(STEER_PRESET["sys_b"], question)
        raw_norms, applied, deltas = [], [], []
        for i in range(len(acts_a)):
            a = acts_a[i].astype(np.float32)[-STEER_TAIL:].mean(axis=0)
            b = acts_b[i].astype(np.float32)[-STEER_TAIL:].mean(axis=0)
            d = a - b
            n = float(np.linalg.norm(d))
            raw_norms.append(n)
            std = float(acts_b[i].astype(np.float32).std())
            unit = d / max(n, 1e-9)
            deltas.append(unit * alpha * std)
            applied.append(float(np.linalg.norm(unit * alpha * std)))
        top = int(np.argmax(raw_norms))
        info = {
            "preset": STEER_PRESET["label"],
            "alpha": alpha,
            "raw_norms": raw_norms,
            "applied_norms": applied,
            "top_layer": top,
        }
        return deltas, info

    def compare(self, question: str, max_new_tokens: int = 512,
                mode: str = "steer", alpha: float = 2.0, combined: bool = True,
                steer_layers: str = "deep") -> dict:
        if not self.ready:
            raise RuntimeError("模型还没加载完成")
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空")
        max_new_tokens = int(max_new_tokens)
        if not (64 <= max_new_tokens <= 2048):
            raise ValueError("生成长度要在 64~2048 token 之间")
        if mode not in ("steer", "clamp"):
            raise ValueError("mode 只能是 steer（语气调整）或 clamp（钳位抑制）")
        if steer_layers not in ("all", "deep"):
            raise ValueError("steer_layers 只能是 all 或 deep（L31–35）")
        alpha = float(alpha)
        if not (STEER_ALPHA_RANGE[0] <= alpha <= STEER_ALPHA_RANGE[1]):
            raise ValueError(
                f"α 要在 {STEER_ALPHA_RANGE[0]}~{STEER_ALPHA_RANGE[1]} 之间："
                f"实验实测 α>4 常见输出退化")

        with self._lock:
            if mode == "clamp":
                _, _, ids = self._build_ids(question)
                # Pass 1：无抑制预填充，采样每层"正常范围"基准（|v| 的 99.5% 分位）
                self._capture_on = True
                self._acts = [None] * len(self.model.model.layers)
                with torch.no_grad():
                    self.model(input_ids=ids.unsqueeze(0).to(self.device))
                thresholds = [float(np.quantile(np.abs(a.astype(np.float32)), 0.995))
                              for a in self._acts]
                text_before, n_before, _ = self._generate(ids, max_new_tokens, clamp=None)
                text_after, n_after, counts = self._generate(ids, max_new_tokens, clamp=thresholds)
                clamp_total = sum(counts) if counts else 0
                return {
                    "mode": "clamp",
                    "question": question,
                    "max_new_tokens": max_new_tokens,
                    "input_tokens": int(ids.shape[0]),
                    "thresholds": thresholds,
                    "gen_before": {"text": text_before, "tokens": n_before},
                    "gen_after": {"text": text_after, "tokens": n_after,
                                  "clamp_counts": counts, "clamp_total": clamp_total},
                }

            # steer：人格方向调向（可选叠加礼貌＋安全规则 system prompt）
            _, _, ids = self._build_ids(
                question, system=STEER_PRESET["sys_a"] if combined else None)
            deltas, direction = self._steer_direction(question, alpha)
            gen_steer = deltas
            if alpha > 0 and steer_layers == "deep":
                gen_steer = [d if i >= STEER_DEEP_FROM else None
                             for i, d in enumerate(deltas)]
            elif alpha == 0:
                gen_steer = None  # 纯 prompt 对照
            direction["layers"] = steer_layers
            direction["combined"] = combined
            text_before, n_before, _ = self._generate(ids, max_new_tokens, clamp=None)
            text_after, n_after, _ = self._generate(ids, max_new_tokens, steer=gen_steer)
            return {
                "mode": "steer",
                "question": question,
                "max_new_tokens": max_new_tokens,
                "input_tokens": int(ids.shape[0]),
                "direction": direction,
                "gen_before": {"text": text_before, "tokens": n_before},
                "gen_after": {"text": text_after, "tokens": n_after},
            }

    def _token_views(self, ids: list[int], offsets, prompt: str, question: str) -> list[dict]:
        """每个 token 的显示文本 + 角色（user / template）。"""
        q0 = prompt.find(question)
        q1 = q0 + len(question) if q0 >= 0 else -1
        out = []
        for i, tid in enumerate(ids):
            s, e = offsets[i]
            role = "user" if q0 >= 0 and e > q0 and s < q1 else "template"
            out.append({"i": i, "t": self.tok.decode([tid], skip_special_tokens=False), "role": role})
        return out

    def _cache(self, acts, tokens) -> str:
        rid = f"r{int(time.time() * 1000) % 10**10:x}{id(acts) % 0xffff:04x}"
        self._results[rid] = {"acts": acts, "tokens": tokens}
        while len(self._results) > MAX_CACHED_RESULTS:
            self._results.popitem(last=False)
        return rid

    # ---------- 结果读取 ----------

    def get_result(self, rid: str) -> dict:
        res = self._results.get(rid)
        if res is None:
            raise KeyError("没有这个分析结果（可能被新分析挤掉了），请重新分析一次")
        return res

    def view_heat(self, rid: str, agg: str | None = None, token: int | None = None) -> dict:
        if agg is not None and agg not in AGG_KINDS:
            raise ValueError(f"agg 只能是 {'/'.join(AGG_KINDS)}")
        return self._build_payload(rid, agg=agg, token=token)

    def neuron_series(self, rid: str, layer: int, neuron: int) -> dict:
        res = self.get_result(rid)
        if not (0 <= layer < len(res["acts"])):
            raise ValueError(f"layer 取值 0~{len(res['acts']) - 1}")
        seq, inter = res["acts"][0].shape
        if not (0 <= neuron < inter):
            raise ValueError(f"neuron 取值 0~{inter - 1}")
        vals = res["acts"][layer][:, neuron].astype(np.float32).tolist()
        return {
            "layer": layer,
            "neuron": neuron,
            "values": vals,
            "tokens": res["tokens"],
        }

    # ---------- payload ----------

    def _build_payload(self, rid: str, agg: str | None, token: int | None) -> dict:
        res = self.get_result(rid)
        acts = res["acts"]
        seq, inter = acts[0].shape
        if token is not None:
            if not (0 <= token < seq):
                raise ValueError(f"token 取值 0~{seq - 1}")
            label, kind = f"第 {token} 个 token", "token"
            vecs = [a[token] for a in acts]
        else:
            kind = agg or "last"
            label = {"last": "最后一个 token", "mean": "全序列均值", "max": "全序列最大值"}[kind]
            vecs = [self._aggregate(a, kind) for a in acts]

        heat = base64.b64encode(
            np.concatenate([np.asarray(v, dtype="<f2").ravel() for v in vecs]).tobytes()
        ).decode("ascii")
        return {
            "rid": rid,
            "agg": kind,
            "agg_label": label,
            "token": token,
            "seq": seq,
            "layers": len(acts),
            "neurons": inter,
            "dtype": "fp16-le",
            "heat": heat,
            "tokens": res["tokens"],
        }

    @staticmethod
    def _aggregate(a: np.ndarray, kind: str) -> np.ndarray:
        if kind == "last":
            return a[-1]
        if kind == "mean":
            return a.astype(np.float32).mean(axis=0).astype(np.float16)
        return a.max(axis=0)  # signed max
