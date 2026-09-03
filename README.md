# SingProbe

**English** | [中文](README_CN.md)

A lightweight **token-level probe training framework** built on the multi-layer hidden states of a frozen Base Model (e.g. Ling-3.0-flash). The frozen Base Model is served by a token-probe-patched SGLang build for forward inference, dumping the per-token hidden states of the configured layers to tmpfs in real time; the training-side **SingProbe model** trains on these hidden states in a streaming fashion and finally outputs **10-dim logits per token**.

> The Base Model is entirely `requires_grad=False` and only runs inference to extract hidden states; **only the probe model is trained**.

## Overview

```
     Base Model (frozen, SGLang token-probe server)
prompt tokens ──► forward with identity token probe
                      │  dumps rmsnorm(hidden_states + residual) per tapped layer
                      ▼
          concat → [batch, seq, hidden_dim × num_layers]
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
       GuardMLP           GuardAttnProbe
      (2-layer MLP)    (causal MQA token probe)
           │                     │
           └──────────┬──────────┘
                      ▼
         per-token logits [batch, seq, 10]
```

### Three tasks (10-dim output, fixed)

| Dims | Task | Notes |
|------|------|-------|
| `0–6` | Query risk (multi-label) | 7 classes (A–G), may co-occur |
| `7` | Query Safe | mutually exclusive with `0–6` |
| `8` | Response Safety | logit >0 ⇒ unsafe |
| `9` | Response Hallucination | token level |

Each dataset only labels its own dims; the rest is unsupervised (label `-100`). See [`data/README.md`](data/README.md) for the format details.

### Two probe architectures (`singprobe_model.arch`)

| `arch` | Class | File | Structure |
|--------|-------|------|-----------|
| `"mlp"` | `GuardMLP` | `models/guard.py` | 2-layer MLP |
| `"attn"` | `GuardAttnProbe` | `models/sglang_attn.py` | proj_q/k/v (MQA, one shared K/V head) → causal MQA → o_proj → query residual → RMSNorm → per-token linear classifier |

Both architectures share the same input contract (the concatenated hidden states); `train.py` dispatches on `arch`. See [`models/README.md`](models/README.md) for architecture details and self-test commands.

---

## Environment Setup

**The scripts in this repo do not install anything** — prepare the environment once following this section before running.

### 1. Python dependencies

Python 3.10+ recommended, with a CUDA environment matching your hardware.

```bash
pip install torch                      # CUDA build matching your hardware
pip install transformers safetensors pyyaml numpy tqdm
```

### 2. Token-probe-patched SGLang (required)

Hidden-state extraction requires the **token-probe-patched** SGLang build (upstream SGLang has no such dump capability). Build and install it from the token-probe branch:

```bash
# build tool required first: pip install build
git clone -b token-probe-ling3-flash-main https://github.com/jinzhen-lin/sglang.git
cd sglang/python
python -m build --wheel --no-isolation
pip install dist/*.whl --force --no-deps
```

### 3. flash-linear-attention (required for Ling-3.0)

Ling-3.0 contains linear-attention layers; the SGLang side needs the `fla` kernels:

```bash
pip install fla
```

### 4. spaCy model (optional)

Only needed when `training.use_hallu_assertion_mask: true`:

```bash
pip install spacy
# Then prepare the en_core_web_sm model directory, either:
#   export SPACY_MODEL_PATH=path/to/en_core_web_sm   # unzipped model dir
#   export SPACY_MODEL_NAME=en_core_web_sm           # globally-installed package name
```

## Model Preparation

The two configs under `configs/all_models/` use the HuggingFace repos `inclusionAI/Ling-3.0-flash` / `inclusionAI/Ling-3.0-tiny`: both the SGLang server launch and the tokenizer load auto-download them from the Hub on first use. For offline environments, download the models in advance and point `base_model.name` at the local path.

## Data Preparation

Training needs two dataset types (format details in [`data/README.md`](data/README.md)):

- **Safety** (JSONL): `Query` / `Response` / `Query_Label` (A–H chars) / `Response_Label` (`Safe`|`Unsafe`) — labels dims 0–8;
- **Hallucination** (JSONL): character `spans` inside the response, mapped to token-level labels — labels dim 9.

Before running, replace the four `path/to/...` placeholders under `data:` in the YAML with your real paths; `train.py` / the pipeline pre-check reports exactly which paths are missing.

## Quickstart: Training

```bash
# First edit configs/all_models/ling-3.0-flash.yaml:
#   replace the dataset paths under data: and training.output_dir with your own
bash scripts/all/ling-3.0-flash.sh              # defaults to --ddp 4
bash scripts/all/ling-3.0-flash.sh --ddp 8
bash scripts/all/ling-3.0-tiny.sh
```

The pipeline `scripts/run_train_pipeline.sh` runs in order:

1. **[1/3]** Start the SGLang hidden-state server in the background (GPUs from `sglang_gpus`) and wait until it is ready;
2. **[2/3]** Launch `train.py` via `torchrun` (training GPUs pinned by `TRAIN_GPUS`, default `0,1,2,3`);
3. **[3/3]** Convert the final checkpoint into a HuggingFace model directory (written to the checkpoint directory's `safetensors/`).

On any exit path (success / failure / interruption) the SGLang process group is always torn down so the GPUs release. The SGLang-side and train-side GPU sets **must be disjoint**.

Handy env-var knobs: `TP` `DP` `GPUS` `TRAIN_GPUS` `SGLANG_PORT` `SGLANG_SGLOG` `PROBE_CKPT` `SAVE_DIR` `MEM_FRACTION` — see the script header (`bash scripts/run_train_pipeline.sh --help`).

Resuming: `bash scripts/all/ling-3.0-flash.sh --resume path/to/checkpoint-N`

## Artifacts and Inference

When training finishes, the pipeline automatically converts the final checkpoint:

```
<output_dir>/checkpoint-N/safetensors/
├── model.safetensors              # SingProbe weights
├── config.json                    # HuggingFace-style config (auto_map + arch fields)
├── configuration_sing_probe.py    # SingProbeMlpConfig / SingProbeAttnConfig
└── modeling_sing_probe.py         # SingProbeMlpModel / SingProbeAttnModel (HF models)
```

The output dir is a standard HuggingFace model directory — load it directly with
`trust_remote_code` (no SingProbe repo needed):

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("path/to/singprobe-model", trust_remote_code=True)
model.eval()
out = model(hidden_states)   # TokenClassifierOutput
logits = out.logits          # [batch, seq, num_classes]
# hidden_states: [batch, seq, hidden_size × len(base_model_layer_ids)] — the
# tapped-layer features of the Base Model concatenated along the feature dim
# (matching the data flow above)
```

Any training checkpoint can also be converted manually:

```bash
python scripts/convert_checkpoint_to_safetensors.py \
    --checkpoint path/to/checkpoint-N \
    --output-dir path/to/singprobe-model \
    --verify    # round-trip check: reload via AutoModel and diff against the .pt
```

## Configuration Overview

Config parsing lives in `config.py` (`Config.from_yaml`); top-level sections:

| Section | Key fields | Notes |
|---------|------------|-------|
| `base_model` | `name` `hidden_layers` `hidden_size` | Base Model (HF repo id or local path), which layers to tap, per-layer width |
| `base_model.inference` | `framework` `sglang_url` `sglang_save_dir` `sglang_probe_ckpt` `sglang_tp/dp/gpus` | SGLang server and client contract (`sglang_probe_ckpt` must match the pipeline's `PROBE_CKPT`) |
| `singprobe_model` | `arch` `num_classes` `num_query_heads` `head_dim` `sliding_window` `init_bias` | Probe architecture; leave `hidden_dim`/`num_layers` unset — they are derived from `base_model` automatically |
| `training` | `epochs` `batch_size` `learning_rate` `output_dir` `task_ratios` … | Training hyperparameters and save cadence |
| `data` | `train_safety_path` `train_hallu_path` `val_*` `max_seq_length` … | Data paths and preprocessing |

## Repository Layout

```
SingProbe/
├── train.py                        # training entry (single-process / torchrun DDP)
├── config.py                       # YAML → nested-dataclass configuration
├── configs/all_models/
│   ├── ling-3.0-flash.yaml
│   └── ling-3.0-tiny.yaml
├── models/
│   ├── guard.py                    # GuardMLP ("mlp" arch)
│   ├── sglang_attn.py              # GuardAttnProbe ("attn" arch)
│   ├── sglang_client.py            # SGLang hidden-state client
│   ├── base_model.py               # in-process HF device_map backend
│   └── guardrail_model.py          # Base + probe composition wrapper
├── data/                           # dataset loading / format conversion (see data/README.md)
├── trainers/                       # loss functions
└── scripts/
    ├── all/ling-3.0-{flash,tiny}.sh     # per-model training entries (thin wrappers)
    ├── run_train_pipeline.sh            # pipeline: SGLang launch → train → convert
    ├── convert_checkpoint_to_safetensors.py
    └── {configuration,modeling}_sing_probe.py # HF modeling files injected next to checkpoints
```

## Legal

Please read [`LEGAL.md`](LEGAL.md) before use.

## Citation

```bibtex
@article{singteam2026singprobe,
  title = {SingProbe Technical Report},
  author = {Sing Team},
  journal = {arXiv preprint arXiv:2608.30703},
  year = {2026},
}
```
