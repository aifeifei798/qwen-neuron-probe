/* 神经元热力探针 · 前端逻辑 */
"use strict";

const $ = (s) => document.querySelector(s);
const LAYERS = 36;
const NEURONS = 9728;

const state = {
  rid: null,
  seq: 0,
  tokens: [],
  vecs: null,          // Float32Array[36]，当前视图的原始激活
  mode: "last",        // last | mean | max | token
  token: null,         // mode=token 时的下标
  layer: null,         // 详情面板当前层
  aggLabel: "",
  ready: false,
};

/* ---------- 基础工具 ---------- */

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  let body = null;
  try { body = await res.json(); } catch { /* 非 JSON 响应 */ }
  if (!res.ok) {
    throw new Error((body && body.detail) ? body.detail : `HTTP ${res.status}`);
  }
  return body;
}

function b64ToUint8(b64) {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
}

function fp16(h) {
  const s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) { const x = (m / 1024) * Math.pow(2, -14); return s ? -x : x; }
  if (e === 31) return s ? -Infinity : Infinity;
  const x = (1 + m / 1024) * Math.pow(2, e - 15);
  return s ? -x : x;
}

function decodeHeat(b64) {
  const u8 = b64ToUint8(b64);
  const dv = new DataView(u8.buffer);
  const n = u8.byteLength >> 1;
  const flat = new Float32Array(n);
  for (let i = 0; i < n; i++) flat[i] = fp16(dv.getUint16(i * 2, true));
  const out = [];
  for (let l = 0; l < LAYERS; l++) out.push(flat.subarray(l * NEURONS, (l + 1) * NEURONS));
  return out;
}

/* ---------- 颜色：从 token 解析 oklch，建 256 级色带 ---------- */

function oklchToRgb(str) {
  const m = /oklch\(\s*([\d.]+)%?\s+([\d.]+)\s+([\d.]+)/.exec(str);
  if (!m) return [0, 0, 0];
  const L = parseFloat(m[1]) / (m[1].endsWith("%") || str.includes("%") ? 100 : 1);
  const C = parseFloat(m[2]);
  const H = (parseFloat(m[3]) * Math.PI) / 180;
  const a = C * Math.cos(H), b = C * Math.sin(H);
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.2914855480 * b;
  const l3 = l_ ** 3, m3 = m_ ** 3, s3 = s_ ** 3;
  const lin = [
     4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3,
    -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3,
    -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3,
  ];
  const g = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(Math.max(c, 0), 1 / 2.4) - 0.055);
  return lin.map((c) => Math.round(g(c) * 255));
}

function cssColor(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

let PALETTE = null; // Uint8Array(256*3)
function buildPalette() {
  const low = oklchToRgb(cssColor("--color-heat-low"));
  const mid = oklchToRgb(cssColor("--color-heat-mid"));
  const high = oklchToRgb(cssColor("--color-heat-high"));
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
  PALETTE = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    const rgb = t < 0.5 ? mix(low, mid, t * 2) : mix(mid, high, (t - 0.5) * 2);
    PALETTE.set(rgb, i * 3);
  }
  PALETTE._mid = mid;
  PALETTE._muted = oklchToRgb(cssColor("--color-muted"));
  PALETTE._danger = oklchToRgb(cssColor("--color-danger"));
}

/* ---------- 状态栏 ---------- */

async function pollStatus() {
  try {
    const s = await fetchJSON("/api/status");
    state.ready = s.ready;
    const el = $("#status");
    if (s.error) {
      el.textContent = `模型加载失败：${s.error}`;
      el.classList.add("text-danger");
    } else if (!s.ready) {
      el.textContent = "模型加载中…";
    } else {
      const where = s.device === "cuda" ? `CUDA ${s.gpu_name} · 显存空闲 ${s.vram_free_gb}/${s.vram_total_gb} GB` : `${s.device}（未检测到 CUDA）`;
      el.textContent = `${s.model} · ${where} · ${s.layers} 层 × ${s.neurons} 神经元`;
      return; // 就绪后停止轮询
    }
  } catch { $("#status").textContent = "探针服务没连上（启动 serve.sh 试试）"; }
  setTimeout(pollStatus, 1200);
}

/* ---------- 分析 ---------- */

const form = $("#form");

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const q = $("#q").value.trim();
  const qErr = $("#q-err");
  qErr.hidden = true;
  if (!q) {
    qErr.textContent = "问题还是空的：先在上面输入一句话，再点「开始分析」。";
    qErr.hidden = false;
    return;
  }
  const btn = $("#btn");
  btn.disabled = true;
  btn.setAttribute("aria-busy", "true");
  showLoading();
  const t0 = performance.now();
  try {
    const r = await fetchJSON("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, agg: "last" }),
    });
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    applyResult(r);
    $("#loading-note").textContent = "";
    console.info(`分析完成：${r.seq} token，${secs}s`);
  } catch (e) {
    showError(e);
  } finally {
    btn.disabled = false;
    btn.removeAttribute("aria-busy");
  }
});

$("#retry").addEventListener("click", () => form.requestSubmit());

function showLoading() {
  $("#error").hidden = true;
  $("#empty").hidden = true;
  $("#heat-section").hidden = true;
  $("#detail").hidden = true;
  const sk = $("#skeleton");
  sk.innerHTML = "";
  for (let l = 0; l < LAYERS; l++) {
    const row = document.createElement("div");
    row.className = "flex items-center gap-2";
    row.innerHTML =
      `<span class="w-14 shrink-0 rounded-sm bg-paper-2" style="height:8px"></span>` +
      `<span class="min-w-0 grow">` +
      `<span class="block w-full rounded-sm bg-paper-2 animate-pulse" style="height:8px"></span>` +
      `<span class="block w-full rounded-sm bg-paper-2 animate-pulse" style="height:3px;margin-top:1px"></span>` +
      `</span>` +
      `<span class="w-24 shrink-0 rounded-sm bg-paper-2" style="height:8px"></span>`;
    sk.appendChild(row);
  }
  $("#loading-note").textContent = "正在把问题送进模型，抓取 36 层的中间激活…";
  $("#loading").hidden = false;
}

function showError(e) {
  $("#loading").hidden = true;
  $("#error-msg").textContent = e.message;
  $("#error-hint").textContent = e.message.includes("挤掉")
    ? "结果被更新的分析覆盖了：点「重试」重新跑一次即可。"
    : "如果是模型刚启动，等顶部状态栏出现 CUDA 信息后再点「重试」。";
  $("#error").hidden = false;
}

/* ---------- 渲染热力图 ---------- */

const rowsEl = $("#rows");
let rowCanvases = [];   // canvas 元素
let layerScales = [];   // 每层归一化分母

function applyResult(r) {
  state.rid = r.rid;
  state.seq = r.seq;
  state.tokens = r.tokens;
  state.vecs = decodeHeat(r.heat);
  state.aggLabel = r.agg_label;
  state.mode = "last";
  state.token = null;
  state.layer = null;
  buildRows();
  buildTokenSelect();
  document.querySelectorAll('input[name="view"]').forEach((el) => { el.checked = el.value === "last"; });
  $("#token-sel").classList.add("hidden");
  $("#loading").hidden = true;
  $("#empty").hidden = true;
  $("#heat-section").hidden = false;
  render();
  updateViewNote(r);
}

function buildRows() {
  rowsEl.innerHTML = "";
  rowCanvases = [];
  layerScales = [];
  const frag = document.createDocumentFragment();
  for (let l = 0; l < LAYERS; l++) {
    const row = document.createElement("div");
    row.className = "flex items-center gap-2";
    row.dataset.layer = l;
    row.innerHTML =
      `<span class="w-14 shrink-0 text-right text-xs text-muted font-mono">L${String(l).padStart(2, "0")}</span>` +
      `<div class="relative min-w-0 grow">` +
      `<canvas class="heat-cv block w-full" width="${NEURONS}" height="1" style="height:8px"></canvas>` +
      `<canvas class="tick-cv block w-full" width="8" height="1" style="height:3px;margin-top:1px;pointer-events:none"></canvas>` +
      `<div class="pointer-events-none absolute top-0 hidden" style="height:8px;width:2px;background:var(--color-ink)" data-bar></div>` +
      `</div>` +
      `<span class="w-24 shrink-0 text-right text-xs text-muted font-mono" data-max></span>`;
    frag.appendChild(row);
    rowCanvases.push(row.querySelector("canvas"));
  }
  rowsEl.appendChild(frag);
}

function buildTokenSelect() {
  const sel = $("#token-sel");
  sel.innerHTML = "";
  const mk = (label, list) => {
    const g = document.createElement("optgroup");
    g.label = label;
    for (const t of list) {
      const o = document.createElement("option");
      o.value = t.i;
      const txt = t.t === "\n" ? "\\n" : t.t;
      o.textContent = `${t.i}: ${txt.length > 18 ? txt.slice(0, 18) + "…" : txt}`;
      g.appendChild(o);
    }
    sel.appendChild(g);
    return g;
  };
  mk("用户内容", state.tokens.filter((t) => t.role === "user"));
  mk("chat 模板", state.tokens.filter((t) => t.role !== "user"));
  sel.value = String(state.seq - 1);
}

function quantile(sorted, q) {
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(q * (sorted.length - 1))));
  return sorted[idx];
}

function computeScales() {
  const q = parseFloat($("#pct-sel").value);
  layerScales = state.vecs.map((arr) => {
    let s;
    if (q < 1) {
      const sorted = Float32Array.from(arr).sort();
      s = quantile(sorted, q);
    } else {
      s = arr.reduce((m, v) => Math.max(m, v), -Infinity);
    }
    return Math.abs(s) > 1e-9 ? Math.abs(s) : 1e-9;
  });
}

function render() {
  if (!state.vecs) return;
  computeScales();
  const useSqrt = $("#sqrt-tgl").checked;
  const danger = PALETTE._danger;
  const img = new ImageData(NEURONS, 1);
  const px = img.data;
  for (let l = 0; l < LAYERS; l++) {
    const arr = state.vecs[l];
    const scale = layerScales[l];
    let redCnt = 0;
    const redIdx = [];
    for (let n = 0; n < NEURONS; n++) {
      const v = arr[n];
      const o = n * 4;
      if (Math.abs(v) > scale) {
        // 超出本层截断基准 → 红
        px[o] = danger[0]; px[o + 1] = danger[1]; px[o + 2] = danger[2]; px[o + 3] = 255;
        redCnt++; redIdx.push(n);
      } else {
        let t = v / scale;
        t = t < 0 ? 0 : t > 1 ? 1 : t;
        if (useSqrt) t = Math.sqrt(t);
        const pi = Math.round(t * 255) * 3;
        px[o] = PALETTE[pi]; px[o + 1] = PALETTE[pi + 1]; px[o + 2] = PALETTE[pi + 2]; px[o + 3] = 255;
      }
    }
    rowCanvases[l].getContext("2d").putImageData(img, 0, 0);
    paintTick(l, redIdx);
    let mx = -Infinity;
    for (let n = 0; n < NEURONS; n++) if (arr[n] > mx) mx = arr[n];
    const maxEl = rowCanvases[l].closest("[data-layer]").querySelector("[data-max]");
    maxEl.textContent = redCnt > 0 ? `max ${mx.toFixed(2)} · 超${redCnt}` : `max ${mx.toFixed(2)}`;
    maxEl.classList.toggle("text-danger", redCnt > 0);
  }
  $("#legend-note").textContent =
    `颜色按各层 ${$("#pct-sel").value === "1" ? "最大值" : "99.5% 分位"}截断归一${useSqrt ? "，开方增强" : ""}；红 = |v| 超出本层截断基准`;
}

/* 红色刻度条：显示分辨率下保证超范围格子肉眼可见 */
function paintTick(layer, redIdx) {
  const tcv = rowCanvases[layer].parentElement.querySelector("canvas.tick-cv");
  const w = Math.max(1, Math.round(tcv.clientWidth));
  if (tcv.width !== w) tcv.width = w;
  const ctx = tcv.getContext("2d");
  ctx.clearRect(0, 0, w, 1);
  if (!redIdx.length) return;
  const marks = new Uint8Array(w);
  for (const n of redIdx) marks[Math.min(w - 1, Math.floor((n * w) / NEURONS))] = 1;
  const d = PALETTE._danger;
  ctx.fillStyle = `rgb(${d[0]},${d[1]},${d[2]})`;
  for (let x = 0; x < w; x++) if (marks[x]) ctx.fillRect(x, 0, 1, 1);
}

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (state.vecs) render(); }, 150);
});

/* 视图切换 */
document.querySelectorAll('input[name="view"]').forEach((el) => {
  el.addEventListener("change", () => switchView(el.value));
});
$("#token-sel").addEventListener("change", () => {
  if (state.mode === "token") switchView("token");
});
$("#sqrt-tgl").addEventListener("change", () => render());
$("#pct-sel").addEventListener("change", () => render());

async function switchView(mode) {
  if (!state.rid) return;
  state.mode = mode;
  $("#token-sel").classList.toggle("hidden", mode !== "token");
  try {
    let url = `/api/view?rid=${state.rid}`;
    if (mode === "token") {
      state.token = parseInt($("#token-sel").value, 10);
      url += `&token=${state.token}`;
    } else {
      state.token = null;
      url += `&agg=${mode}`;
    }
    const r = await fetchJSON(url);
    state.vecs = decodeHeat(r.heat);
    state.aggLabel = r.agg_label;
    $("#detail").hidden = true;
    state.layer = null;
    rowsEl.querySelectorAll("[data-layer]").forEach((el) => el.classList.remove("row-sel"));
    render();
    updateViewNote(r);
  } catch (e) {
    showError(e);
  }
}

function updateViewNote(r) {
  let desc = `视图：${r.agg_label}`;
  if (r.token !== null && r.token !== undefined && state.tokens[r.token]) {
    desc += `（#${r.token}「${displayTok(state.tokens[r.token].t)}」）`;
  }
  $("#view-note").textContent = desc + " · 同一问题可随时换视图";
}

function displayTok(t) {
  return t === "\n" ? "\\n" : t.length > 24 ? t.slice(0, 24) + "…" : t;
}

/* ---------- 悬停与点击 ---------- */

rowsEl.addEventListener("mousemove", (ev) => {
  const row = ev.target.closest("[data-layer]");
  const cv = row && row.querySelector("canvas.heat-cv");
  if (!row || !cv || !state.vecs) { hideTip(); return; }
  const layer = parseInt(row.dataset.layer, 10);
  const rect = cv.getBoundingClientRect();
  if (ev.clientX < rect.left || ev.clientX > rect.right) { hideTip(); return; }
  const n = Math.min(NEURONS - 1, Math.max(0, Math.floor(((ev.clientX - rect.left) / rect.width) * NEURONS)));
  const v = state.vecs[layer][n];
  const base = layerScales[layer];
  const oor = Math.abs(v) > base;
  const tip = $("#tip");
  tip.textContent = `L${layer} · #${n} · v=${v.toFixed(3)} · 基准 ${base.toFixed(3)}${oor ? " · 超范围" : ""}`;
  tip.hidden = false;
  const tw = tip.offsetWidth;
  tip.style.left = `${Math.min(ev.clientX + 14, window.innerWidth - tw - 8)}px`;
  tip.style.top = `${ev.clientY + 14}px`;
  const bar = cv.parentElement.querySelector("[data-bar]");
  bar.classList.remove("hidden");
  bar.style.left = `${((ev.clientX - rect.left) / rect.width) * 100}%`;
});

rowsEl.addEventListener("mouseleave", hideTip);
function hideTip() {
  $("#tip").hidden = true;
  rowsEl.querySelectorAll("[data-bar]").forEach((b) => b.classList.add("hidden"));
}

rowsEl.addEventListener("click", (ev) => {
  const row = ev.target.closest("[data-layer]");
  if (!row) return;
  const layer = parseInt(row.dataset.layer, 10);
  rowsEl.querySelectorAll("[data-layer]").forEach((el) => el.classList.toggle("row-sel", el === row));
  buildDetail(layer);
});

/* ---------- 层详情 ---------- */

async function buildDetail(layer) {
  state.layer = layer;
  $("#detail").hidden = false;
  $("#spark-box").hidden = true;
  const arr = state.vecs[layer];
  const idx = Array.from({ length: NEURONS }, (_, i) => i);
  idx.sort((a, b) => Math.abs(arr[b]) - Math.abs(arr[a]));
  let maxAbs = 0, nonzero = 0;
  for (let i = 0; i < NEURONS; i++) {
    const a = Math.abs(arr[i]);
    if (a > maxAbs) maxAbs = a;
    if (a > 1e-6) nonzero++;
  }
  $("#h-detail").textContent = `第 ${layer} 层 · 最活跃的 20 个神经元`;
  $("#detail-sub").textContent =
    `视图：${state.aggLabel} · 层内 |v| 最大 ${maxAbs.toFixed(3)} · 非零占比 ${((nonzero / NEURONS) * 100).toFixed(1)}%`;
  const tbody = $("#top-tbody");
  tbody.innerHTML = "";
  const frag = document.createDocumentFragment();
  idx.slice(0, 20).forEach((n, k) => {
    const v = arr[n];
    const oor = Math.abs(v) > layerScales[layer];
    const tr = document.createElement("tr");
    tr.className = "border-b border-border/60";
    tr.innerHTML =
      `<td class="py-1.5 pr-3 text-muted">${k + 1}</td>` +
      `<td class="py-1.5 pr-3 font-mono">#${n}</td>` +
      `<td class="py-1.5 pr-3 text-right font-mono${oor ? " text-danger" : ""}">${v.toFixed(3)}</td>` +
      `<td class="py-1.5 pr-3 text-right font-mono text-muted">${((Math.abs(v) / maxAbs) * 100).toFixed(1)}%</td>` +
      `<td class="py-1"><button type="button" data-n="${n}" ` +
      `class="inline-flex min-h-9 items-center rounded-md border border-border px-3 text-sm hover:bg-border/40">看曲线</button></td>`;
    frag.appendChild(tr);
  });
  tbody.appendChild(frag);
  tbody.onclick = (ev) => {
    const b = ev.target.closest("button[data-n]");
    if (b) showSpark(layer, parseInt(b.dataset.n, 10));
  };
}

async function showSpark(layer, neuron) {
  try {
    const r = await fetchJSON(`/api/neuron?rid=${state.rid}&layer=${layer}&neuron=${neuron}`);
    $("#spark-box").hidden = false;
    $("#spark-title").textContent = `第 ${layer} 层 · 神经元 #${neuron} 在 ${r.values.length} 个 token 上的激活`;
    drawSpark(r.values);
    $("#spark").onscroll = null;
    $("#spark").onmousemove = (ev) => {
      const rect = ev.target.getBoundingClientRect();
      const i = Math.min(r.values.length - 1, Math.max(0, Math.floor(((ev.clientX - rect.left) / rect.width) * r.values.length)));
      $("#spark-token").textContent =
        `#${i}「${displayTok(r.tokens[i].t)}」 v=${r.values[i].toFixed(3)} (${r.tokens[i].role === "user" ? "用户" : "模板"})`;
    };
    $("#spark").onmouseleave = () => { $("#spark-token").textContent = ""; };
  } catch (e) {
    showError(e);
  }
}

function drawSpark(values) {
  const cv = $("#spark");
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const mid = PALETTE._mid, gray = PALETTE._muted;
  let vmax = 0;
  for (const v of values) vmax = Math.max(vmax, Math.abs(v));
  if (vmax <= 0) vmax = 1;
  const barW = Math.max(1, W / values.length);
  const scale = state.layer != null && layerScales[state.layer] ? layerScales[state.layer] : Infinity;
  const danger = PALETTE._danger;
  ctx.fillStyle = "rgba(0,0,0,0.12)"; // 只当 0 基线弱参考 —— 非语义色
  ctx.fillRect(0, H / 2 - 0.5, W, 1);
  for (let i = 0; i < values.length; i++) {
    const h = (Math.abs(values[i]) / vmax) * (H / 2 - 4);
    const oor = Math.abs(values[i]) > scale;
    const c = oor ? danger : (state.tokens[i] && state.tokens[i].role === "user" ? mid : gray);
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
    const x = i * barW;
    if (values[i] >= 0) ctx.fillRect(x, H / 2 - h, Math.max(1, barW - 0.5), h);
    else ctx.fillRect(x, H / 2 + 1, Math.max(1, barW - 0.5), h);
  }
}

/* ---------- Tab 切换 ---------- */

const TABS = [
  { btn: $("#tab-probe"), panel: $("#view-probe") },
  { btn: $("#tab-compare"), panel: $("#view-compare") },
];

function selectTab(which) {
  for (const { btn, panel } of TABS) {
    const on = btn.id === `tab-${which}`;
    btn.setAttribute("aria-selected", String(on));
    btn.classList.toggle("bg-ink", on);
    btn.classList.toggle("text-paper", on);
    btn.classList.toggle("text-ink", !on);
    btn.classList.toggle("hover:bg-border/40", !on);
    panel.hidden = !on;
  }
}
TABS.forEach(({ btn }) => btn.addEventListener("click", () => selectTab(btn.id.replace("tab-", ""))));

/* ---------- 对比聊天 ---------- */

const cmpForm = $("#cmp-form");
let cmpTimer = null;

cmpForm.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const q = $("#q2").value.trim();
  const errEl = $("#q2-err");
  errEl.hidden = true;
  if (!q) {
    errEl.textContent = "问题还是空的：先输入一句话，再点「生成对比」。";
    errEl.hidden = false;
    return;
  }
  const btn = $("#cmp-btn");
  btn.disabled = true;
  btn.setAttribute("aria-busy", "true");
  $("#cmp-error").hidden = true;
  $("#cmp-result").hidden = true;
  $("#cmp-empty").hidden = true;
  $("#cmp-loading").hidden = false;
  let secs = 0;
  $("#cmp-timer").textContent = "0";
  cmpTimer = setInterval(() => { secs += 1; $("#cmp-timer").textContent = String(secs); }, 1000);
  const t0 = performance.now();
  try {
    const r = await fetchJSON("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: q,
        max_new_tokens: parseInt($("#cmp-len").value, 10),
        mode: document.querySelector('input[name="cmp-mode"]:checked').value,
        alpha: parseFloat($("#cmp-alpha").value),
        combined: $("#cmp-combined").checked,
        steer_layers: $("#cmp-layers").value,
      }),
    });
    applyCompare(r, ((performance.now() - t0) / 1000).toFixed(1));
  } catch (e) {
    $("#cmp-loading").hidden = true;
    $("#cmp-error-msg").textContent = e.message;
    $("#cmp-error").hidden = false;
  } finally {
    clearInterval(cmpTimer);
    btn.disabled = false;
    btn.removeAttribute("aria-busy");
  }
});

$("#cmp-retry").addEventListener("click", () => cmpForm.requestSubmit());

/* 调整方式切换：语气调整显示 α 滑条与叠加开关 */
document.querySelectorAll('input[name="cmp-mode"]').forEach((el) => {
  el.addEventListener("change", () => {
    $("#steer-opts").hidden =
      document.querySelector('input[name="cmp-mode"]:checked').value !== "steer";
  });
});
$("#cmp-alpha").addEventListener("input", () => {
  $("#alpha-val").textContent = $("#cmp-alpha").value;
});

const lastAnswer = { before: "", after: "" };

function buildCmpSkeleton() {
  const box = $("#cmp-skel");
  box.innerHTML = "";
  for (let p = 0; p < 2; p++) {
    const d = document.createElement("div");
    d.className = "space-y-2 rounded-lg border border-border p-4";
    d.innerHTML =
      `<p class="h-4 w-24 rounded-sm bg-paper-2"></p>` +
      [1, 1, 0.8, 0.65]
        .map((w) => `<p class="h-3 rounded-sm bg-paper-2 animate-pulse" style="width:${Math.round(w * 100)}%"></p>`)
        .join("");
    box.appendChild(d);
  }
}

function splitThink(text) {
  const open = text.indexOf("<think>");
  const close = text.indexOf("</think>");
  if (open === -1 && close === -1) return { think: null, answer: text, closed: false };
  const think = text.slice(open >= 0 ? open + 7 : 0, close >= 0 ? close : text.length);
  const answer = close >= 0 ? text.slice(close + 8).trim() : "";
  return { think, answer, closed: close >= 0 };
}

function fillPanel(prefix, gen) {
  const { think, answer, closed } = splitThink(gen.text);
  const box = $(`#${prefix}-think`);
  if (think && think.trim()) {
    box.hidden = false;
    $(`#${prefix}-think-text`).textContent = think.trim();
    box.querySelector("summary").textContent = closed
      ? `思考过程（${think.trim().length} 字，折叠）`
      : "思考过程（未收尾，已达生成长度上限）";
    box.open = !closed;
  } else {
    box.hidden = true;
  }
  $(`#${prefix}-answer`).textContent =
    answer || (closed ? "" : "—— 回答部分没有生成：思维链超出长度上限，试试更大的生成长度。");
  lastAnswer[prefix] = answer;
  return answer;
}

function miniBars(sel, values, colorFn, fmt) {
  const box = $(sel);
  box.innerHTML = "";
  if (!values.length) return;
  const max = Math.max(...values, 1e-9);
  const frag = document.createDocumentFragment();
  values.forEach((v, i) => {
    const b = document.createElement("div");
    b.className = "min-w-0 grow";
    b.style.height = `${Math.max(2, Math.round((v / max) * 62))}px`;
    b.style.backgroundColor = colorFn(v);
    b.title = `L${i}: ${fmt(v)}`;
    frag.appendChild(b);
  });
  box.appendChild(frag);
}

function applyCompare(r, secs) {
  fillPanel("before", r.gen_before);
  fillPanel("after", r.gen_after);
  const steer = r.mode === "steer";
  if (steer) {
    $("#before-title").textContent = "调整前 · 原版";
    $("#after-title").textContent = r.direction.combined
      ? `调整后 · 「${r.direction.preset}」方向 α=${r.direction.alpha} ＋ 礼貌规则`
      : `调整后 · 「${r.direction.preset}」方向 α=${r.direction.alpha}`;
    $("#after-meta").textContent =
      `${r.gen_after.tokens} token 生成 · 注入${r.direction.layers === "deep" ? "仅深层 L31–35" : "全部 36 层"}${r.direction.combined ? " · 叠加礼貌＋安全规则 prompt" : ""}`;
    $("#cmp-note").textContent =
      `问题 ${r.input_tokens} token · 人格方向 · 生成上限 ${r.max_new_tokens} · 用时 ${secs}s`;
    $("#stats-title").textContent = "调向器工作记录";
    $("#bars-label").textContent = "每层方向原始强度 ‖Δ‖";
    $("#thr-label").textContent = "每层实际加上的偏移量（α×std 缩放后）";
    $("#stats-caliber").textContent =
      "方向 = 「礼貌客服」与「卑歉客服」两种人设预填充激活之差（末尾 16 token 均值），单位化后按层激活 std×α 缩放。"
      + "归因（α=0 对照实测）：道歉词归零主要是 prompt 的功劳；全层注入会把字面语义（如「炸了」）压成比喻——默认只注深层；深层注入相比纯 prompt 多出共情句与安全确认倾向。";
    miniBars("#clamp-bars", r.direction.raw_norms, () => "var(--color-accent)", (v) => v.toFixed(2));
    miniBars("#thr-bars", r.direction.applied_norms, () => "var(--color-accent)", (v) => v.toFixed(2));
    $("#cmp-stats-note").textContent = `方向最集中在第 ${r.direction.top_layer} 层`;
  } else {
    $("#before-title").textContent = "抑制前 · 原版";
    $("#after-title").textContent = "抑制后 · 截断激活";
    $("#after-meta").textContent = `${r.gen_after.tokens} token 生成 · 钳位 ${r.gen_after.clamp_total} 次`;
    $("#cmp-note").textContent =
      `问题 ${r.input_tokens} token · 钳位基准 99.5% 分位 · 生成上限 ${r.max_new_tokens} · 用时 ${secs}s`;
    $("#stats-title").textContent = "抑制器工作记录";
    $("#bars-label").textContent = "每层钳位次数（生成全程累计）";
    $("#thr-label").textContent = "每层钳位基准（该问题预填充的 |v| 99.5% 分位）";
    $("#stats-caliber").textContent =
      "基准 = 读题那一遍各层激活的 |v| 99.5% 分位；钳位 = 每层把超出 ±基准 的值截回边界，预填充和每个生成步都生效。";
    const danger = PALETTE._danger;
    miniBars("#clamp-bars", r.gen_after.clamp_counts || [],
      () => `rgb(${danger[0]},${danger[1]},${danger[2]})`, (v) => `${v} 次`);
    miniBars("#thr-bars", r.thresholds, () => "var(--color-accent)", (v) => v.toFixed(3));
    const counts = r.gen_after.clamp_counts || [];
    if (r.gen_after.clamp_total > 0 && counts.length) {
      const top = counts.indexOf(Math.max(...counts));
      $("#cmp-stats-note").textContent =
        `共钳位 ${r.gen_after.clamp_total} 次激活 · 最集中在第 ${top} 层（${counts[top]} 次）`;
    } else {
      $("#cmp-stats-note").textContent = "本次生成没有任何激活超出基准";
    }
  }
  // 道歉词只统计回答段：思维链会"引用"禁词导致计数虚高
  const aw = ["抱歉", "对不起", "不好意思", "道歉"];
  const cnt = (s) => (s ? aw.reduce((n, w) => n + s.split(w).length - 1, 0) : 0);
  $("#apology-note").textContent =
    `道歉词（只统计回答段）：调整前 ${cnt(lastAnswer.before)} 次 · 调整后 ${cnt(lastAnswer.after)} 次`;
  $("#cmp-loading").hidden = true;
  $("#cmp-result").hidden = false;
}

/* ---------- 启动 ---------- */

buildPalette();
buildCmpSkeleton();
pollStatus();
