"""FastAPI 入口：uvicorn server.main:app"""
from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .probe import AGG_KINDS, MAX_INPUT_TOKENS, NeuronProbe

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="Qwen3-4B 神经元热力探针", docs_url=None, redoc_url=None)
probe = NeuronProbe()
threading.Thread(target=probe.load, daemon=True).start()


class AnalyzeIn(BaseModel):
    question: str
    agg: str = "last"


class CompareIn(BaseModel):
    question: str
    max_new_tokens: int = 512
    mode: str = "steer"      # steer=语气调整 clamp=钳位抑制
    alpha: float = 2.0       # 调向强度
    combined: bool = True    # steer 时是否叠加礼貌 system prompt（实测推荐）
    steer_layers: str = "deep"  # all=全部层 deep=仅深层 L31-35（实测全层注入伤语义）


@app.get("/api/status")
def api_status():
    return probe.status()


@app.post("/api/analyze")
def api_analyze(body: AnalyzeIn):
    if not probe.ready:
        raise HTTPException(503, "模型还没加载完成，等几秒再试")
    if len(body.question) > 4000:
        raise HTTPException(422, "问题太长：超过 4000 字符，请缩短后再分析")
    try:
        return probe.analyze(body.question, body.agg)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/view")
def api_view(
    rid: str,
    agg: str | None = None,
    token: int | None = Query(default=None, ge=0),
):
    if agg is None and token is None:
        agg = "last"
    if agg is not None and agg not in AGG_KINDS:
        raise HTTPException(422, f"agg 只能是 {'/'.join(AGG_KINDS)}")
    try:
        return probe.view_heat(rid, agg=agg, token=token)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(410, str(exc)) from exc


@app.post("/api/compare")
def api_compare(body: CompareIn):
    if not probe.ready:
        raise HTTPException(503, "模型还没加载完成，等几秒再试")
    if len(body.question) > 4000:
        raise HTTPException(422, "问题太长：超过 4000 字符，请缩短后再试")
    try:
        return probe.compare(body.question, body.max_new_tokens, body.mode,
                             body.alpha, body.combined, body.steer_layers)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/neuron")
def api_neuron(rid: str, layer: int = Query(ge=0), neuron: int = Query(ge=0)):
    try:
        return probe.neuron_series(rid, layer, neuron)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(410, str(exc)) from exc


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
