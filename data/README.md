# Data Module

Unified dataset processing for SingProbe training: converts raw JSON Lines files into per-token, 10-dim label tensors, handling **left padding** and the **chat template** so that Response boundaries stay aligned with the model's tokenization.

## Supported dataset types

| `dataset_type` | Converter | Labels |
|----------------|-----------|--------|
| `safety` | `SafetyConverter` | dims 0–7 (query risk + query safe) and dim 8 (response safety) |
| `response_hallu` | `ResponseHalluConverter` | dim 9 (token-level hallucination) |

Each dataset only supervises its own dims; all other positions carry the ignore label (`-100`).

### Safety format

One record carries both the query-risk labels and the response-safety label:

```json
{
  "Query": "...",
  "Response": "...",
  "Query_Label": "BF",
  "Response_Label": "Unsafe"
}
```

- `Query_Label`: concatenated `A`–`H` characters. `A`–`G` → dims 0–6 (multi-label); `H` → dim 7 (Query Safe, mutually exclusive with the risk letters — if both appear, `H` wins and a warning is logged once).
- `Response_Label`: `"Safe"` / `"Unsafe"` → dim 8 (`Unsafe` → 1).
- Labels are sample-level and broadcast onto **all Response tokens** (query dims are also placed on the last query token; see the label contract below).

### Response_Hallu format

Token-level hallucination annotations; `spans` are character offsets inside the response, mapped to token labels on the fly:

```json
{
  "id": "...",
  "dataset": "...",
  "split": "train",
  "prompt": "...",
  "response": "...",
  "label": "hallucinated",
  "hallu_class": "factuality",
  "spans": [{"start": 12, "end": 30, "text": "..."}]
}
```

- Covered `spans` → dim 9 = 1 (hallucinated response tokens); other response tokens → 0.
- `split` field drives train/val carving: `"validation"` → val, `null`/`"train"` → train, `"test"` is held out and never used for training.

## Label contract (10 dims)

| Dims | Meaning |
|------|---------|
| 0–6 | Query risk classes A–G (multi-label) |
| 7 | Query Safe |
| 8 | Response Safety (logit > 0 ⇒ unsafe) |
| 9 | Response Hallucination (token level) |

Per-token assignment:

- **Padding tokens** → `-100` everywhere (ignored)
- **Query tokens (except the last)** → `-100` everywhere
- **Last query token** → query dims labeled, response dims `-100`
- **Response tokens** → full row: query dims + response safety + hallucination

## Usage

`train.py` builds these datasets internally from the `data:` section of your YAML; you normally don't touch this module directly. For custom pipelines:

```python
from transformers import AutoTokenizer
from data import GuardrailDataset, GuardrailCollator

tokenizer = AutoTokenizer.from_pretrained("inclusionAI/Ling-3.0-flash")  # or any local path
tokenizer.padding_side = "left"   # REQUIRED: boundary math assumes left padding

dataset = GuardrailDataset(
    data_path="path/to/safety_train.jsonl",
    dataset_type="safety",          # or "response_hallu"
    tokenizer=tokenizer,
    max_length=8192,
)
sample = dataset[0]
# sample["input_ids"]      [max_length]
# sample["labels"]         [max_length, 10]
# sample["response_mask"]  [max_length]

loader_batch = GuardrailCollator()([dataset[0], dataset[1]])
```

When both datasets are present, `train.py` balances them with `TaskRatioSampler` / `DistributedTaskRatioSampler` (`data/samplers/`) according to `training.task_ratios`.

## Implementation notes

- **Left padding is mandatory** (`tokenizer.padding_side = "left"`): the Response-boundary computation (`calculate_response_boundaries_with_padding` in `utils.py`) accounts for PAD tokens prepended on the left and the chat-template markers around query/response.
- **Character → token mapping**: hallucination `spans` use character offsets into the response text and are projected onto response tokens by `utils.py::map_spans_to_token_labels` at sample time.
- **POS assertion mask** (optional): with `training.use_hallu_assertion_mask: true`, `assertion_mask.py` computes a per-response-token content-word mask (spaCy `en_core_web_sm` POS tags) that is ANDed into the dim-9 hallucination mask, leaving function-word response tokens unsupervised for the hallucination task.
- **Tokenizer**: comes from `base_model.name` (the same model the SGLang server serves), so dataset tokens and dumped hidden states always align.

## Module layout

```
data/
├── dataset.py                  # GuardrailDataset (unified interface)
├── collator.py                 # GuardrailCollator (batch padding/alignment)
├── utils.py                    # boundary calculation, char→token mapping, chat-template helpers
├── assertion_mask.py           # spaCy POS assertion mask (optional, dim 9 only)
├── converters/
│   ├── base.py                 # BaseConverter
│   ├── safety.py               # SafetyConverter
│   └── response_hallu.py       # ResponseHalluConverter
└── samplers/
    └── task_ratio_sampler.py   # TaskRatioSampler / DistributedTaskRatioSampler
```
