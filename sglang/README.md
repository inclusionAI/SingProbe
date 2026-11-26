# SGLang: Fast inference for LLMs and multimodal models

<p align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang" width="400">
</p>

<p align="center">
  <a href="https://pypi.org/project/sglang/"><img src="https://img.shields.io/pypi/v/sglang?style=flat&amp;label=PyPI&amp;labelColor=555555&amp;color=orange" alt="PyPI version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green?style=flat&amp;labelColor=555555" alt="License: Apache 2.0"></a>
  <a href="https://pypistats.org/packages/sglang"><img src="https://img.shields.io/pypi/dm/sglang?style=flat&amp;label=Downloads&amp;labelColor=555555&amp;color=blue" alt="PyPI downloads per month"></a>
</p>

<p align="center">
  <a href="https://docs.sglang.io/">Docs</a> |
  <a href="https://cookbook.sglang.io/">Cookbook</a> |
  <a href="https://www.sglang.io/">Website</a> |
  <a href="https://lmsys.org/blog/">Blog</a> |
  <a href="https://slack.sglang.io/">Slack</a>
</p>

SGLang is an open-source inference framework for large language, vision-language, and diffusion models, optimized for agentic workloads, large-scale serving, and RL rollouts. [SGLang Diffusion](https://docs.sglang.io/docs/sglang-diffusion) is its built-in image and video generation engine, included in this repository and the `sglang` Python package.

👋 Get started below, or meet the community at [SGLang Events](https://www.sglang.io/events), including meetups, developer meetings, workshops, and office hours.

> **[SGLang Summit 2026: Meet the builders of open AI](https://www.sglang.io/summit)**<br>
> **Our flagship annual gathering · November 12–13 · San Francisco**
>
> Go deep on inference, training, and AI hardware through technical talks, live demos, and hands-on workshops. Bring your deployment challenges and meet the engineers and researchers building the stack.

## Token-probe inference

This SingProbe build of SGLang can return safety scores during generation or save the input features of selected decoder layers. Run the examples below from **this directory** (`sglang/` in the SingProbe repository). In a Linux CUDA environment with Python 3.12 and a Rust toolchain, install this checkout with `python -m pip install ./python`.

| Base model | Identity example (zero-based layers) | Feature width per token | Identity launcher |
| --- | --- | ---: | --- |
| DeepSeek-V4.1-Flash-1017 | [0, 14, 39] | 15360 | [launch_identity.sh](scripts/token_probe/launch_identity.sh) |
| Ling-3.1-flash-1019 | [0, 35, 71] | 12288 | [launch_ling_identity.sh](scripts/token_probe/launch_ling_identity.sh) |
| Qwen3.8-Flash-Next | [15, 31, 47] | 7680 | [launch_qwen38_identity.sh](scripts/token_probe/launch_qwen38_identity.sh) |
| GLM-5.3-Flash | [14, 29, 43] | 12288 | [launch_glm53_identity.sh](scripts/token_probe/launch_glm53_identity.sh) |

### Serve a trained probe

Set `MODEL_PATH` to the base model used for training and `PROBE_CKPT` to the matching exported probe directory containing `config.json` and `model.safetensors`. The eight-GPU example below uses the checkpoint's layer IDs.

For Qwen3.8-Flash-Next and GLM-5.3-Flash, set `CHUNK_SIZE=-1` and use non-speculative decoding. For Qwen, also export `SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV=0`.

```bash
export MODEL_PATH=/path/to/base-model
export PROBE_CKPT=/path/to/matching-singprobe
unset SGLANG_TOKEN_PROBE_SAVE_DIR
python -m sglang.launch_server \
  --model-path "$MODEL_PATH" --probe-ckpt "$PROBE_CKPT" --trust-remote-code \
  --tp-size 8 --ep-size 8 --chunked-prefill-size "${CHUNK_SIZE:-1024}" \
  --context-length 16384 --max-prefill-tokens 8192 --max-total-tokens 32768 \
  --max-running-requests 4 --mem-fraction-static 0.85 \
  --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled \
  --host 127.0.0.1 --port 31080
```

From another terminal, send a request. The JSON response contains one row per generated token in `meta_info.token_probe_probs`. For the standard 10-output SingProbe heads, the first eight scores cover query intent, followed by response unsafety and hallucination risk.

```bash
curl -fsS http://127.0.0.1:31080/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":8}}'
```

### Dump hidden states with an Identity probe

Choose the Identity launcher from the table for your base model. The example below starts a separate DeepSeek server. Use a dedicated scratch directory for `PROBE_CKPT` because the launcher writes an Identity `config.json` there. To extract features for a specific training run, set `LAYER_IDS` to the JSON list of layer IDs in its training config.

```bash
MODEL_PATH=/path/to/DeepSeek-V4.1-Flash-1017 \
PYTHON="$(command -v python)" PORT=31081 \
PROBE_CKPT="$PWD/probe-work/identity" SAVE_DIR="$PWD/probe-work/dumps" \
bash scripts/token_probe/launch_identity.sh
```

In a second terminal in the same directory, send a short request and verify the saved tensor:

```bash
python scripts/token_probe/check_identity.py --smoke-only \
  --url http://127.0.0.1:31081 --checkpoint "$PWD/probe-work/identity" \
  --save-dir "$PWD/probe-work/dumps" --report "$PWD/probe-work/report.json"
```

Each request writes `<request-id>.safetensors` to `SAVE_DIR`. Its `scores` tensor has shape `[processed tokens, hidden size × number of selected layers]`. Each row contains the RMS-normalized layer inputs in checkpoint order. Cached prompt tokens are omitted.

## Get Started

Pull the Docker image, which includes SGLang and its dependencies:

```bash
docker pull lmsysorg/sglang:latest
```

Alternatively, install SGLang in an activated Python environment with uv:

```bash
uv pip install --prerelease=allow sglang
```

Next, launch your model:

- [Quickstart](https://docs.sglang.io/docs/get-started/quickstart): Run your first model and send a request.
- [Cookbook](https://cookbook.sglang.io/): Choose your model and hardware to get a ready-to-run launch command.

## Supported Hardware

SGLang supports a wide range of GPUs, TPUs, NPUs, CPUs, and Apple Silicon platforms.

| Platform | Representative hardware |
| --- | --- |
| [NVIDIA](https://docs.sglang.io/docs/hardware-platforms/nvidia-gpus) | B200/B300/GB200/GB300; H100/H200/H800/H20; A100; RTX 30/40/50 series, RTX 6000 Ada / PRO 6000; DGX Spark, Jetson Orin |
| [AMD](https://docs.sglang.io/docs/hardware-platforms/amd_gpu) | Instinct MI300X, MI325X, MI350X, MI355X |
| [Google TPU](https://docs.sglang.io/docs/hardware-platforms/tpu) | v6e, v7; [SGLang-TorchTPU](https://github.com/sgl-project/sglang-torchtpu) / [SGL-JAX](https://github.com/sgl-project/sglang-jax) |
| Intel ([GPU](https://docs.sglang.io/docs/hardware-platforms/xpu) / [CPU](https://docs.sglang.io/docs/hardware-platforms/cpu_server)) | Arc / Arc Pro B-Series GPUs, Xeon CPUs |
| [Apple Silicon](https://docs.sglang.io/docs/hardware-platforms/apple_metal) | Macs via Metal / MLX |
| [Huawei Ascend](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/getting-started/installation) | A2, A3, 950PR/DT NPUs |
| [Moore Threads](https://docs.sglang.io/docs/hardware-platforms/mthreads_gpu) | MTT S5000 GPUs |

Integrations in progress: AWS Trainium, [Alibaba T-Head PPU](https://github.com/sgl-project/sglang/issues/37519), [Cambricon MLU](https://github.com/sgl-project/sglang/pull/26898), Qualcomm QAIC, MetaX, Hygon HCU/DCU, Iluvatar CoreX, and more.

SGLang-TorchTPU brings the SGLang serving engine to Google TPUs through PyTorch, torch_tpu, and Pallas kernels. It supports LLM and vision-language serving on TPU7x (Ironwood), including tensor and expert parallelism across hosts. SGL-JAX provides the JAX-based TPU serving path.

See the [Cookbook](https://cookbook.sglang.io/) and platform guides for model compatibility and setup.

## SGL Ecosystem

| Area | Projects | Purpose |
| --- | --- | --- |
| Diffusion | [SGLang Diffusion](https://docs.sglang.io/docs/sglang-diffusion/installation) | Built into SGLang for image and video generation with diffusion models. |
| Audio | [SGLang Omni](https://github.com/sgl-project/sglang-omni) | Audio model serving for text-to-speech (TTS) and automatic speech recognition (ASR). |
| Speculative Decoding | [SpecForge](https://github.com/sgl-project/SpecForge) | Train draft models for speculative decoding and deploy them with SGLang. |
| RL and Post-Training | [Miles](https://github.com/radixark/miles), [slime](https://github.com/THUDM/slime), [AReaL](https://github.com/inclusionAI/AReaL), [Tunix](https://github.com/google/tunix), [verl](https://github.com/volcengine/verl) | Training frameworks that integrate SGLang for rollout generation. |
| Deployment and Orchestration | [SMG](https://github.com/smg-project/smg), [RBG](https://github.com/sgl-project/rbg), [llm-d](https://llm-d.ai/docs/dev/operations/disaggregation/sglang), [Ray Serve](https://docs.ray.io/en/latest/serve/llm/user-guides/sglang.html), [NVIDIA Dynamo](https://docs.nvidia.com/dynamo/backends/sg-lang/reference-guide) | Deploy and scale SGLang inference services with routing, load balancing, and cluster orchestration. |
| Education | [Mini-SGLang](https://github.com/sgl-project/mini-sglang), [zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang), [DeepLearning.AI course](https://www.deeplearning.ai/short-courses/efficient-inference-with-sglang-text-and-image-generation/) | Learn inference engine design and efficient text and image generation through code and hands-on courses. |

## Development and Contributing

Contributions are welcome, from bug fixes and documentation to model support and performance improvements.

### Development setup

Start from the `lmsysorg/sglang:dev` Docker image, which provides development tools and most dependencies. Clone or mount your SGLang checkout inside the container, then install it in editable mode from the repository root so tests use your local Python changes:

```bash
pip install -e "python"
```

In an activated virtual environment, you can use `uv pip install --prerelease=allow -e "python"` instead. See the [development guide](https://docs.sglang.io/docs/developer_guide/development_guide_using_docker) for container setup and testing.

### Contribute

1. Fork the repository and create a branch for your changes. For larger changes, discuss your proposal in a [GitHub issue](https://github.com/sgl-project/sglang/issues) or on [Slack](https://slack.sglang.io/).
2. Make your changes, run the relevant tests, and add regression coverage for fixes or new behavior. Run `pre-commit run --all-files` before submitting.
3. Open a pull request describing the change and how you tested it. Include benchmarks or accuracy evaluations when relevant.

See the [contributor guide](https://docs.sglang.io/docs/developer_guide/contribution_guide) for formatting, testing, and pull request instructions. Documentation contributors can start with the [docs guide](docs/README.md).

## Community and Sponsorship

SGLang is hosted by [LMSYS](https://lmsys.org/about/), a non-profit open-source organization.

- **Community discussions:** Join [Slack](https://slack.sglang.io/) for technical questions and development discussions.
- **Events:** Find meetups, workshops, and office hours on [SGLang Events](https://www.sglang.io/events).
- **Updates:** Follow [X](https://x.com/lmsysorg) and [LinkedIn](https://www.linkedin.com/company/sgl-project/) for project updates, and the [LMSYS Blog](https://lmsys.org/blog/) for release announcements and technical articles.
- **Project resources:** Explore the [documentation](https://docs.sglang.io/), [Cookbook](https://cookbook.sglang.io/), [roadmap](https://roadmap.sglang.io/), [release notes](https://github.com/sgl-project/sglang/releases), [issue tracker](https://github.com/sgl-project/sglang/issues), and [contributor guide](https://docs.sglang.io/docs/developer_guide/contribution_guide).
- **Contact Us:** For enterprise adoption and deployment, technical consulting, sponsorship, or partnership inquiries, please contact [sglang@lmsys.org](mailto:sglang@lmsys.org).
- **Contributor sponsorship:** Long-term active SGLang contributors are eligible for coding agent sponsorship, including Cursor, Claude Code, or OpenAI Codex. To apply, email [sglang@lmsys.org](mailto:sglang@lmsys.org) with links to your key commits or pull requests.

## Trusted by Industry and Research

SGLang serves production workloads across AI labs, cloud platforms, enterprises, and universities.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="Organizations adopting SGLang" width="800">

## Acknowledgment
We learned the design and reused code from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).
