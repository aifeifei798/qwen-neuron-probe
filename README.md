# Qwen3-4B 神经元激活探针 & 行为干预实验台

输入一个问题，看 Qwen3-4B-Thinking-2507 读完问题的那一刻，**36 层 × 每层 9728 个 MLP 神经元**（共 350,208 个）各自的激活强度；同一个系统上做激活钳位、人格方向调向的对照实验——把「干预位置 → 副作用 → 归因边界」整条链路跑通的机制研究工具。

```
探针 Tab：1 像素 = 1 个神经元的热力图          对比聊天 Tab：同一问题跑两遍
┌ L00 ▁▂▁▁▃▁▁▁▁▁▁▁▁… max 0.61        ┌ 调整前 · 原版 ┐ ┌ 调整后 · 方向α=1.5 ┐
│ …                        │          │ <think>折叠   │ │ 钳位/调向统计图     │
└ L35 ▁▂▁▁▁▁█▁▁▁▁▁▁▁… 超 109          │ 共情+行动步骤  │ │ 道歉词 0 vs 基线    │
  红 = |v| 超出本层 99.5% 分位基准       └───────────────┘ └────────────────────┘
```

## 快速开始

```bash
./install.sh                       # 建 venv 装依赖（有 GPU 自动装 CUDA 版 torch）
./serve.sh                         # http://127.0.0.1:8790 ，模型首次加载约 10–30 秒
```

- 模型路径非默认时：`export QWEN_MODEL_PATH=/path/to/Qwen3-4B-Thinking-2507`（也可写进
  `server/probe.py` 默认值）；改端口：`PORT=9000 ./serve.sh`
- 国内网络：`./install.sh --mirror huawei`（或 tuna）；无 NVIDIA GPU：`./install.sh --cpu`
- 依赖：Python ≥3.10、torch ≥2.7（Blackwell 显卡需 sm_120 支持版本）、transformers 5.x、
  fastapi、uvicorn、numpy。改前端后重建 CSS：

```bash
cd web && ./node_modules/.bin/tailwindcss -i src/app.css -o dist/app.css --minify
```

## 两个 Tab

### 激活探针
- 热力条 1 像素 = 1 个神经元，36 行叠放；**红 = |v| 超出本层截断基准**（99.5% 分位），
  条下红色刻度条按显示分辨率汇总，行尾标注该层超范围个数
- 悬停看精确值；点击一层看该层最活跃的 20 个神经元；单神经元可看激活随 token 的曲线
- 视图：最后 token / 全序列均值 / 全序列最大值 / 指定 token；开方增强与分位截断可切换
- 口径：激活 = 每层 `mlp.down_proj` 的输入（SwiGLU 之后），prompt 套官方 chat 模板、
  只做 prefill 不生成；输入上限 1024 token；内存保留最近 2 次结果

### 对比聊天
同一问题、贪心解码跑两遍，唯一差异是干预 hook：
- **语气调整（steer）**：人格方向 = 「礼貌客服·不道歉」vs「卑微道歉客服」两种人设
  预填充激活之差（末尾 16 token 均值），单位化后按层 std×α 缩放；可叠加
  礼貌＋安全硬规则 system prompt（推荐）、可选注入层（默认仅深层 L31–35）
- **钳位抑制（clamp）**：把超出读题基准 ±99.5% 分位的激活截回边界（保底实验能力，
  已知会塌缩——这本身就是结论）
- 统计面板：每层方向强度/偏移量/钳位次数柱状图 + 回答段道歉词计数
  （只统计回答段——思维链会"引用"禁词，全文计数虚高）

**推荐配置**：α=1.5 · 仅深层 · 安全规则 prompt · 投诉/安全场景给 2048 生成预算
（验收电池 4/4，见 `reports/`）。

## 接口

| 接口 | 说明 |
|---|---|
| `GET /api/status` | 模型/设备/显存状态 |
| `POST /api/analyze` | `{question, agg}` → 36×9728 热度（fp16 base64）+ token 列表 |
| `GET /api/view?rid=&agg=last|mean|max` 或 `&token=i` | 换视图 |
| `GET /api/neuron?rid=&layer=&neuron=` | 单神经元随 token 的激活曲线 |
| `POST /api/compare` | `{question, max_new_tokens, mode, alpha, combined, steer_layers}` → 抑制/调整前后两版回答 + 干预统计 |

## 实验结论（详见 reports/）

1. **prompt 主导**：α=0 对照证实道歉词归零主要是 prompt 的功劳，方向向量独立贡献的
   强主张已撤回；深层注入仅留痕迹级增强（共情/安全确认倾向）
2. **位置决定副作用**：全层注入把字面语义压成比喻（"路由器炸了"→"断网"），
   仅深层 L31–35 不伤语义；浅中层是内容编码区，不可强干预
3. **钳位全档位塌缩**：99.5% ~ 99.99% 全部退化，深层巨大激活是结构组件
   （生成期 100% 步数超出读题基准）——"抑制超标激活"不是可行的干预方式
4. **风格约束会挤压安全语义**：唯一有真实事故潜力的发现；对策 = prompt 内置
   "安全 > 风格"让位条款（已验证）+ 数据/验收成对兜底

## 目录

```
server/probe.py     hook 捕获 + 钳位/调向 hooks + fp16 编码 + 结果缓存
server/main.py      FastAPI 路由 + 静态托管
web/                单页 UI（Tailwind v4；src/ 源码，dist/ 构建产物）
experiments/        可复跑实验脚本：suppress_study / steer_study / router_case
reports/            干预位置-副作用-归因边界 · 礼貌客服微调-数据准备报告
install.sh          环境安装（uv/pip 自适应、CUDA/CPU 探测、国内镜像开关）
serve.sh            启动脚本（默认端口 8790，PORT 可改）
```

## 工具链说明

本研究由国产模型工具链分工完成：**ZCode** 执行环境（探针服务、hook 注入、对照实验
脚本与验收电池，全部本机运行）；**DeepSeek** 批判性审查（α=0 对照设计、仅深层注入
假设、安全让位条款均源自审稿意见，随后由实验裁决）；**Qwen3-4B-Thinking-2507** 是
被试模型。实验与数据全程在本机环境内处理，未上传外部服务。

## 已知边界

- 全部结论在贪心解码、单一模型、Thinking 系 chat 模板下得出；归因核心案例 N=1–2
- 对比聊天一次占用 GPU ~15–50 秒（视生成长度），期间探针请求排队
- 生成上限 2048 token；思考链长的场景（安全投诉）建议用满预算
