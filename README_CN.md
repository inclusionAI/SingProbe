# SingProbe

[English](README.md) | **中文**

基于冻结 Base Model（如 Ling-3.0-flash）多层 hidden states 的轻量级 **token 级探针（probe）训练框架**：冻结的 Base Model 由带 token-probe 补丁的 SGLang 拉起做前向推理，把指定层的每 token hidden states 实时 dump 到 tmpfs；训练侧的 **SingProbe 模型**在这些 hidden states 上做流式训练，输出**每 token 的 10 维 logits**。

> Base Model 全部 `requires_grad=False`，仅做推理抽取 hidden states；**只有探针模型被训练**。

## 方案概览

```
     Base Model (冻结, SGLang token-probe 服务)
prompt tokens ──► forward with identity token probe
                      │  dump 各 tapped 层的 rmsnorm(hidden_states + residual)
                      ▼
          拼接 → [batch, seq, hidden_dim × num_layers]
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
       GuardMLP           GuardAttnProbe
      (2 层 MLP)         (因果 MQA token 探针)
           │                     │
           └──────────┬──────────┘
                      ▼
         per-token logits [batch, seq, 10]
```

### 三个任务（10 维输出，固定）

| 维度 | 任务 | 说明 |
|------|------|------|
| `0–6` | Query 风险多标签 | 7 类（A–G），多标签共存 |
| `7` | Query Safe | 与 `0–6` 互斥 |
| `8` | Response Safety | 该位 >0 ⇒ 不安全 |
| `9` | Response 幻觉 | token 级 |

每个数据集只标注自己的维度，其余位置无监督（标签 `-100`），格式细节见 [`data/README.md`](data/README.md)。

### 两种探针架构（`singprobe_model.arch`）

| `arch` | 类 | 文件 | 结构 |
|--------|------|------|------|
| `"mlp"` | `GuardMLP` | `models/guard.py` | 2 层 MLP |
| `"attn"` | `GuardAttnProbe` | `models/sglang_attn.py` | proj_q/k/v（MQA，共享 K/V 头）→ 因果 MQA → o_proj → query 残差 → RMSNorm → 逐 token 线性分类头 |

两种架构共用同一输入约定（拼接后的 hidden states），`train.py` 按 `arch` 分发。架构细节与自测命令见 [`models/README.md`](models/README.md)。

---

## 环境准备

**本仓库脚本不做任何安装**，运行前请按本节准备好环境。

### 1. Python 依赖

建议 Python 3.10+，以及与硬件匹配的 CUDA 环境。

```bash
pip install torch                      # CUDA 版，与你的硬件匹配
pip install transformers safetensors pyyaml numpy tqdm
```

### 2. 带 token probe 补丁的 SGLang（必需）

hidden states 抽取依赖带 **token-probe 补丁**的 SGLang（上游标准版没有该 dump 能力），从 token-probe 分支编译安装：

```bash
# 需先安装构建工具：pip install build
git clone -b token-probe-ling3-flash-main https://github.com/jinzhen-lin/sglang.git
cd sglang/python
python -m build --wheel --no-isolation
pip install dist/*.whl --force --no-deps
```

### 3. flash-linear-attention（Ling-3.0 必需）

Ling-3.0 含线性注意力层，SGLang 侧需要 `fla` 内核：

```bash
pip install fla
```

### 4. spaCy 模型（可选）

仅当 `training.use_hallu_assertion_mask: true` 时需要：

```bash
pip install spacy
# 再准备 en_core_web_sm 模型目录，二选一：
#   export SPACY_MODEL_PATH=path/to/en_core_web_sm   # 指向解压后的模型目录
#   export SPACY_MODEL_NAME=en_core_web_sm           # 指向已全局安装的包名
```

## 模型准备

`configs/all_models/` 下两个配置分别使用 HuggingFace 仓库 `inclusionAI/Ling-3.0-flash` / `inclusionAI/Ling-3.0-tiny`：SGLang 启动与 tokenizer 加载时会自动从 Hub 下载。离线环境可先自行下载，然后把 `base_model.name` 改成本地路径。

## 数据准备

训练需要两类数据集（格式细节见 [`data/README.md`](data/README.md)）：

- **Safety**（JSONL）：`Query` / `Response` / `Query_Label`（A–H 字符）/ `Response_Label`（`Safe`|`Unsafe`），标注 dims 0–8；
- **Hallucination**（JSONL）：response 内的字符区间 `spans`，映射为 token 级标签，标注 dim 9。

运行前把 YAML 里 `data:` 下的 4 个 `path/to/...` 占位符替换成真实路径；`train.py` / 流水线在预检查时会明确报出缺哪些路径。

## 快速开始：训练

```bash
# 先编辑 configs/all_models/ling-3.0-flash.yaml：
#   把 data: 下的数据集路径和 training.output_dir 改成你自己的
bash scripts/all/ling-3.0-flash.sh              # 默认 --ddp 4
bash scripts/all/ling-3.0-flash.sh --ddp 8
bash scripts/all/ling-3.0-tiny.sh
```

流水线 `scripts/run_train_pipeline.sh` 依次完成：

1. **[1/3]** 后台启动 SGLang hidden-state 服务（GPU 由 `sglang_gpus` 指定），等其就绪；
2. **[2/3]** `torchrun` 拉起 `train.py`（训练卡由 `TRAIN_GPUS` 指定，默认 `0,1,2,3`）；
3. **[3/3]** 把最终 checkpoint 转换为 HuggingFace 模型目录（写入该 checkpoint 目录下的 `safetensors/`）。

任何退出路径（成功/失败/中断）都会拆掉 SGLang 进程组释放 GPU。SGLang 侧与训练侧的 GPU **必须不相交**。

常用环境变量开关：`TP` `DP` `GPUS` `TRAIN_GPUS` `SGLANG_PORT` `SGLANG_SGLOG` `PROBE_CKPT` `SAVE_DIR` `MEM_FRACTION`，见脚本头注释（`bash scripts/run_train_pipeline.sh --help`）。

断点续训：`bash scripts/all/ling-3.0-flash.sh --resume path/to/checkpoint-N`

## 产物与推理

训练结束时流水线自动转换最终 checkpoint，产物为：

```
<output_dir>/checkpoint-N/safetensors/
├── model.safetensors              # SingProbe 权重
├── config.json                    # HuggingFace 风格 config（auto_map + 架构字段）
├── configuration_sing_probe.py    # SingProbeMlpConfig / SingProbeAttnConfig
└── modeling_sing_probe.py         # SingProbeMlpModel / SingProbeAttnModel（HF 模型）
```

产物目录就是标准的 HuggingFace 模型目录，无需 SingProbe 仓库即可直接加载：

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("path/to/singprobe-model", trust_remote_code=True)
model.eval()
out = model(hidden_states)   # TokenClassifierOutput
logits = out.logits          # [batch, seq, num_classes]
# hidden_states: [batch, seq, hidden_size × len(base_model_layer_ids)]，
# 即 Base Model 各 tapped 层特征沿特征维的拼接（与上方数据流一致）
```

也可手动转换任意训练检查点：

```bash
python scripts/convert_checkpoint_to_safetensors.py \
    --checkpoint path/to/checkpoint-N \
    --output-dir path/to/singprobe-model \
    --verify    # 往返校验：用 AutoModel 重载并对齐 .pt
```

## 配置概览

配置解析在 `config.py`（`Config.from_yaml`），顶层段落：

| 段落 | 主要字段 | 说明 |
|------|----------|------|
| `base_model` | `name` `hidden_layers` `hidden_size` | Base Model（HF 仓库名或本地路径）、抽取哪些层、每层宽度 |
| `base_model.inference` | `framework` `sglang_url` `sglang_save_dir` `sglang_probe_ckpt` `sglang_tp/dp/gpus` | SGLang 服务与客户端约定（`sglang_probe_ckpt` 需与流水线的 `PROBE_CKPT` 一致） |
| `singprobe_model` | `arch` `num_classes` `num_query_heads` `head_dim` `sliding_window` `init_bias` | 探针架构；`hidden_dim`/`num_layers` 留空、由 `base_model` 自动推导 |
| `training` | `epochs` `batch_size` `learning_rate` `output_dir` `task_ratios` … | 训练超参与保存节奏 |
| `data` | `train_safety_path` `train_hallu_path` `val_*` `max_seq_length` … | 数据路径与预处理 |

## 项目结构

```
SingProbe/
├── train.py                        # 训练主脚本（单进程 / torchrun DDP）
├── config.py                       # YAML → 嵌套 dataclass 配置
├── configs/all_models/
│   ├── ling-3.0-flash.yaml
│   └── ling-3.0-tiny.yaml
├── models/
│   ├── guard.py                    # GuardMLP（"mlp" 架构）
│   ├── sglang_attn.py              # GuardAttnProbe（"attn" 架构）
│   ├── sglang_client.py            # SGLang hidden-state 客户端
│   ├── base_model.py               # 进程内 HF device_map 后端
│   └── guardrail_model.py          # Base + 探针的组合封装
├── data/                           # 数据加载 / 格式转换（见 data/README.md）
├── trainers/                       # 损失函数
└── scripts/
    ├── all/ling-3.0-{flash,tiny}.sh     # 按模型的训练入口（薄封装）
    ├── run_train_pipeline.sh            # 流水线：SGLang 启动 → 训练 → 转换
    ├── convert_checkpoint_to_safetensors.py
    └── {configuration,modeling}_sing_probe.py # 随 checkpoint 注入的 HF modeling 文件
```

## 法律声明

使用前请阅读 [`LEGAL.md`](LEGAL.md)。
