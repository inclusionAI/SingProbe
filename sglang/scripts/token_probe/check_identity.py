"""Send short, batched and chunked requests and inspect saved Identity features."""

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import torch
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:31080")
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((args.checkpoint / "config.json").read_text())
    width = config["hidden_size"] * len(config["base_model_layer_ids"])
    long_prompt = (
        "Please remember these words: "
        + "red green blue yellow. " * 90
        + "\nName one color."
    )
    cases = [
        ("probe-short", "The capital of France is"),
        (
            ["probe-batch-a", "probe-batch-b"],
            ["What is two plus two?", "Write one sentence about the moon."],
        ),
        ("probe-chunked", long_prompt),
        ("probe-cached", long_prompt),
    ]
    if args.smoke_only:
        cases = cases[:1]
    records = []
    for rid, text in cases:
        payload = {
            "rid": rid,
            "text": text,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 8,
                "min_new_tokens": 8,
            },
        }
        req = Request(
            args.url + "/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=300) as response:
            result = json.load(response)
        responses = result if isinstance(result, list) else [result]
        for output in responses:
            meta = output["meta_info"]
            if meta["id"] == "probe-cached":
                assert meta.get("cached_tokens", 0) > 0, meta
            path = args.save_dir / (meta["id"] + ".safetensors")
            with safe_open(path, framework="pt", device="cpu") as saved:
                scores = saved.get_tensor("scores")
                metadata = saved.metadata()
            assert scores.ndim == 2 and scores.shape[1] == width, scores.shape
            assert scores.shape[0] > 0 and torch.isfinite(scores).all()
            rms = (
                scores.view(
                    -1, len(config["base_model_layer_ids"]), config["hidden_size"]
                )
                .square()
                .mean(-1)
            )
            torch.testing.assert_close(rms, torch.ones_like(rms), atol=0.02, rtol=0.02)
            expected = (
                meta["prompt_tokens"]
                - meta.get("cached_tokens", 0)
                + meta["completion_tokens"]
                - 1
            )
            assert scores.shape[0] == expected, (meta, scores.shape, expected)
            records.append(
                {
                    "id": meta["id"],
                    "shape": list(scores.shape),
                    "prompt_tokens": meta["prompt_tokens"],
                    "completion_tokens": meta["completion_tokens"],
                    "cached_tokens": meta.get("cached_tokens", 0),
                    "max_rms_error": float((rms - 1).abs().max()),
                    "metadata": metadata,
                    "text": output["text"],
                }
            )
    args.report.write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
