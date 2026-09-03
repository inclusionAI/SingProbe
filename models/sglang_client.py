# English-only convention enforced here.
"""
SGLang hidden-state client -- a drop-in replacement for BaseModelWrapper that
fetches per-token hidden states from an external SGLang server instead of
running a frozen HF base model on the training GPUs.

How SGLang provides hidden states (verified against sglang 1.x):
  - An *identity* token probe (`probe_type: "identity"`) emits the tapped
    layers' per-token hidden states concatenated to
    `[num_tokens, len(layer_ids) * hidden_size]` unchanged -- no MLP, no head.
  - With `sampling_params.max_new_tokens = 1`, SGLang scores EVERY prompt
    token during prefill (the probe records the full extend, not just the last
    row) and then decodes exactly one token. That single decoding step makes
    `check_finished()` mark the request finished, which is what triggers the
    dump to be written -- `max_new_tokens = 0` does NOT reliably reach the
    write path (the request can be flagged finished before the prefill scores
    are collected), which is why we use 1, not 0. The decode row is NOT added
    to the dump (the decode-score path skips already-finished requests), so
    the saved `scores` still hold exactly the prompt-token hidden states.
  - When `SGLANG_TOKEN_PROBE_SAVE_DIR` is set, those hidden states are written
    to `<save_dir>/<rid>.safetensors` (key `"scores"`), and are NOT returned
    over the API. The API path is unusable for identity probes anyway (empty
    label_names -> rows collapse to `{}`), so dump+read+delete is the only
    viable online path.

Request / rid contract (matches the verified-correct reference client):
  - We do NOT send a client-chosen `rid`. The POST body is the minimal
    `{"input_ids": [...], "sampling_params": {"max_new_tokens": 1,
    "temperature": 0}}`. SGLang assigns a rid server-side (its own uuid when
    none is supplied) and echoes it back in the response body as
    `meta_info["id"]`. The server names the dump file from that same rid, so
    reading `meta_info["id"]` is the canonical way to locate the file -- it
    never depends on the server preserving a client-supplied rid.

Dump read / lifecycle:
  Each forward() sends B `/generate` requests (one per non-padded sample),
  reads B `<rid>.safetensors` files (rid taken from each response), and
  DELETES each file the instant it is read successfully. The read uses a
  bounded retry loop (file may not be flushed yet) followed by one final
  blocking read, exactly like the reference; on any error we best-effort
  sweep the in-flight rids so a failure never leaks files for the next batch.
  Point `save_dir` at a tmpfs (/dev/shm, the default) so dumps never touch
  disk -- the client reads+deletes files identically regardless of backing
  store, and steady-state RAM use is just the in-flight (batch-sized) files.

Token alignment:
  Tokenization is the dataset's job (data/dataset.py), producing left-padded
  input_ids with content right-aligned. We strip the left padding (take the
  last `attention_mask.sum()` ids) before sending, so SGLang returns hidden
  states for exactly the dataset's tokens -- the Guard's loss masks
  (trainers/loss.py) line up directly. We then left-pad each layer's hidden
  states back to the batch sequence length with zeros (pad positions are
  loss-masked, so their values are irrelevant).
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from safetensors import SafetensorError


class SGLangHiddenStateClient(nn.Module):
    """Drop-in BaseModelWrapper replacement backed by an SGLang server.

    Implements the same surface train.py calls on BaseModelWrapper:
    forward(input_ids, attention_mask, ...), get_hidden_dim(), get_vocab_size(),
    eval(), _clear_hooks(), _hidden_states_cache, and a no-op to()/cuda().
    It carries no parameters (the base model lives in the SGLang server).
    """

    def __init__(
        self,
        model_name: str,
        hidden_layers: List[int],
        url: str = "http://127.0.0.1:6000",
        save_dir: str = "/dev/shm/save_probe",
        probe_ckpt: str = "path/to/probe_ckpt",
        hidden_size: Optional[int] = None,
        timeout: float = 300.0,
        max_concurrency: int = 8,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        retries: int = 3,
        file_wait_tries: int = 50,
        file_wait_secs: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name  # informational only (server owns the model)
        self.hidden_layers = list(hidden_layers)
        self.url = url.rstrip("/")
        self.save_dir = save_dir
        self.probe_ckpt = probe_ckpt
        self.timeout = timeout
        self.max_concurrency = max(1, int(max_concurrency))
        self.dtype = dtype
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # request / dump-read robustness knobs (mirror the reference client).
        self.retries = max(1, int(retries))
        self.file_wait_tries = max(1, int(file_wait_tries))
        self.file_wait_secs = float(file_wait_secs)

        # resolve hidden_size: probe_ckpt config.json first, else explicit arg.
        self._hidden_size = hidden_size or self._read_hidden_size()
        # number of feature columns the probe dumps per token.
        self._feature_dim = len(self.hidden_layers) * self._hidden_size

        # compatibility shims used by train.py's cleanup block
        self._hidden_states_cache: Dict[int, torch.Tensor] = {}
        self._hooks: List = []

        # Per-forward step-phase timing (seconds), populated by forward() so the
        # training loop can attribute the base-model cost to "send /generate"
        # vs "read dump files" vs "assemble + H2D". Train.py reads these for its
        # per-step profiling log. Reset at the start of every forward().
        self.last_send_secs: float = 0.0      # POST all /generate + collect rids
        self.last_read_secs: float = 0.0      # poll + read + delete dump files
        self.last_assemble_secs: float = 0.0  # zero-pad per layer + H2D + cast

        # Persistent thread pool for /generate sends. The SGLang prefill latency
        # dominates the step, and /generate blocks until the request finishes --
        # so we run many sends concurrently here (max_concurrency) AND allow the
        # training loop to dispatch the NEXT batch's sends on this same pool
        # while the current batch's guard training (read+asm+fwd+bwd) runs on
        # the main thread / GPU. This overlaps SGLang inference of batch N+1 with
        # Guard training of batch N. One shared pool avoids spawning/tearing down
        # threads per forward().
        from concurrent.futures import ThreadPoolExecutor
        self._send_pool = ThreadPoolExecutor(
            max_workers=self.max_concurrency, thread_name_prefix="sglang-send"
        )
        # A send dispatch already in flight for the next batch (prefetch), or None.
        # The training loop calls dispatch_send()/collect_and_assemble() directly
        # to drive overlap; forward() is the synchronous fallback path.
        self._prefetch_future = None  # type: ignore[var-annotated]

        os.makedirs(self.save_dir, exist_ok=True)

        print(f"[SGLang] Hidden-state client initialized:")
        print(f"  - url: {self.url}")
        print(f"  - save_dir: {self.save_dir}")
        print(f"  - hidden_layers: {self.hidden_layers}")
        print(f"  - hidden_size: {self._hidden_size} (feature_dim={self._feature_dim})")
        print(f"  - max_concurrency: {self.max_concurrency}")
        self._health_check()

    # ------------------------------------------------------------------ config

    def _read_hidden_size(self) -> int:
        """Read the per-layer hidden size from the probe checkpoint's config.json.

        Accepts either probe-config schema this repo's launch scripts write:
          * new (ProbeConfig): ``hidden_dim`` + ``base_model_layer_ids`` (+ num_layers)
          * legacy (make_probe_checkpoint): ``hidden_size`` + ``layer_ids``
        Both resolve to the same scalar. A present-but-mismatched layer list warns
        but still falls back to config.base_model.hidden_layers for the input split.
        """
        path = os.path.join(self.probe_ckpt, "config.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            # hidden_dim (new ProbeConfig schema) wins; fall back to hidden_size
            # (legacy make_probe_checkpoint schema) used by the older scripts.
            hs = cfg.get("hidden_dim")
            if hs is None:
                hs = cfg["hidden_size"]
            # base_model_layer_ids (new) wins; fall back to layer_ids (legacy).
            layer_ids = cfg.get("base_model_layer_ids")
            if layer_ids is None:
                layer_ids = cfg.get("layer_ids")
            if layer_ids is not None and list(layer_ids) != list(self.hidden_layers):
                print(f"[SGLang] WARNING: probe layers {layer_ids} differ from "
                      f"config base_model.hidden_layers {self.hidden_layers}; "
                      f"using base_model.hidden_layers for the Guard input split.")
            return int(hs)
        except Exception as e:
            raise RuntimeError(
                f"[SGLang] cannot read hidden size from probe ckpt {path}: {e}. "
                f"Create it via make_probe_checkpoint."
            )

    def _health_check(self) -> None:
        """Best-effort ping so a misconfigured URL fails fast at construction."""
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=10) as r:
                if r.status != 200:
                    print(f"[SGLang] health endpoint returned status {r.status} "
                          f"(continuing -- some builds omit /health)")
        except Exception as e:
            # Non-fatal: /health may be absent; real failures surface on forward.
            print(f"[SGLang] health check against {self.url}/health failed: {e} "
                  f"(continuing; will retry on the first forward).")

    # -------------------------------------------------------------- public API

    def get_hidden_dim(self) -> int:
        return self._hidden_size

    def get_vocab_size(self) -> int:
        # Not needed by SGLang path (warmup sends arbitrary ids), return 0 like
        # the HF wrapper's fallback so warmup_cuda guards correctly.
        return 0

    def _clear_hooks(self) -> None:
        # No hooks -- nothing to clear. Kept for train.py cleanup compatibility.
        self._hooks.clear()
        self._hidden_states_cache.clear()

    def reset_last_scores(self) -> None:  # noqa: API parity shim
        pass

    # ----------------------------------------------------------------- forward

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sample_ids: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[int, torch.Tensor]:
        """Fetch hidden states for a batch.

        Args:
            input_ids: [B, seq_len], left-padded (content right-aligned).
            attention_mask: [B, seq_len], 1 on real tokens.
            sample_ids: per-sample ids (informational only now -- the server
                assigns the rid; we no longer send a client-chosen one). Kept
                for call-site compatibility with BaseModelWrapper.

        Returns:
            {layer_idx: [B, seq_len, hidden_size]} -- left-padded to match the
            input grid, so downstream loss masks align exactly as in the HF path.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        input_ids_cpu = input_ids.detach().to("cpu", dtype=torch.long)
        attn_cpu = attention_mask.detach().to("cpu", dtype=torch.long)
        B, seq_len = input_ids_cpu.shape

        # sample_ids no longer drive rids; accept any value (incl. None) silently.

        # Reset per-forward phase timers (train.py reads them after this call to
        # attribute base-model cost to send vs read vs assemble).
        self.last_send_secs = 0.0
        self.last_read_secs = 0.0
        self.last_assemble_secs = 0.0

        jobs = self._build_jobs(attn_cpu, input_ids_cpu)

        # Synchronous fallback path: send -> (cleanup) -> read+assemble. The
        # training loop instead calls dispatch_send()/collect_and_assemble()
        # separately so it can overlap the next batch's send with this batch's
        # guard training (see train.py train_epoch).
        rids = self.dispatch_send(jobs)
        rids_in_flight = [rid for rid in rids.values() if rid]
        try:
            hidden_states = self.collect_and_assemble(rids, jobs, seq_len)
        finally:
            # Best-effort: delete any leftover in-flight files so a partial
            # failure never poisons the next batch. Successful reads already
            # self-delete, so this only sweeps the failure-path leftovers.
            self._cleanup_rids(rids_in_flight)

        return hidden_states

    # ------------------------------------------------------- request dispatch

    def _build_jobs(self, attn_cpu: torch.Tensor, input_ids_cpu: torch.Tensor) -> List[Dict]:
        """Strip left padding and turn a [B, seq_len] batch into per-sample jobs.

        Returns a list of {"content_ids": [...], "content_len": N}; degenerate
        (all-pad) rows get an empty job to be zero-filled at assemble time.
        """
        jobs: List[Dict] = []
        B = input_ids_cpu.shape[0]
        for i in range(B):
            content_len = int(attn_cpu[i].sum().item())
            if content_len <= 0:
                jobs.append({"content_ids": [], "content_len": 0})
                continue
            content = input_ids_cpu[i, -content_len:].tolist()
            jobs.append({"content_ids": content, "content_len": content_len})
        return jobs

    # ----------------------------------------------------------- overlapped API
    # The training loop calls the next two methods separately to overlap the
    # NEXT batch's SGLang inference (send) with the CURRENT batch's guard
    # training. dispatch_send() is safe to run on a background thread / the
    # shared _send_pool; collect_and_assemble() is the blocking GPU+IO step.

    def dispatch_send(self, jobs: List[Dict]) -> Dict[int, str]:
        """POST all /generate for these jobs (concurrency-bounded), return rids.

        Each /generate blocks until SGLang has finished prefill + the 1-token
        decode that triggers the dump write, so the wall time of this call IS
        the SGLang inference latency (not just network RTT). Safe to run in a
        background thread while the GPU trains the previous batch -- the shared
        _send_pool fans the B requests out concurrently. Records last_send_secs.

        Returns: {sample_index: rid} for non-degenerate jobs.
        """
        _t_send = time.perf_counter()
        # i -> server-assigned rid (only set for non-degenerate jobs).
        rids: Dict[int, str] = {}
        send_futures = []
        for i, job in enumerate(jobs):
            if job["content_len"] <= 0:
                continue
            send_futures.append((i, self._send_pool.submit(self._send_one, job)))

        # Collect rids; raise on any send error once all have been submitted
        # (so a single bad request does not abort the whole batch mid-flight and
        # leak the others' files -- the caller sweeps rids in a finally).
        send_errors = []
        for i, fut in send_futures:
            try:
                resp = fut.result()
                rids[i] = self._extract_rid(resp)
            except Exception as e:
                send_errors.append((i, e))
        if send_errors:
            i, e = send_errors[0]
            raise RuntimeError(f"[SGLang] /generate failed for sample {i}: {e}")
        self.last_send_secs = time.perf_counter() - _t_send
        return rids

    def dispatch_send_async(self, jobs: List[Dict]):
        """Submit dispatch_send to a background thread; return a Future.

        The SGLang inference for the next batch runs concurrently with the
        current batch's guard training. Caller masks exceptions until it joins.
        """
        from concurrent.futures import ThreadPoolExecutor
        # A single supervisor thread that itself fans out B sends over the
        # shared _send_pool. (We don't submit jobs directly to _send_pool here
        # because the caller needs ONE future whose result is the rid dict.)
        if not hasattr(self, "_dispatch_supervisor"):
            self._dispatch_supervisor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="sglang-prefetch"
            )
        return self._dispatch_supervisor.submit(self.dispatch_send, jobs)

    def collect_and_assemble(
        self,
        rids: Dict[int, str],
        jobs: List[Dict],
        seq_len: int,
    ) -> Dict[int, torch.Tensor]:
        """Read the dump files for `rids`, then assemble per-layer hidden states.

        Blocking: poll+load+delete each dump (read phase) then zero-pad + H2D
        (assemble phase). Records last_read_secs / last_assemble_secs.
        """
        _t_read = time.perf_counter()
        results: Dict[int, torch.Tensor] = {}  # i -> [content_len, feature_dim]
        for i, job in enumerate(jobs):
            if job["content_len"] <= 0:
                continue
            results[i] = self._load_scores(rids[i], job["content_len"])
        self.last_read_secs = time.perf_counter() - _t_read

        B = len(jobs)
        _t_asm = time.perf_counter()
        hidden_states: Dict[int, torch.Tensor] = {}
        # Precompute, per sample, the left-pad offset once (shared across layers).
        offsets = [seq_len - jobs[i]["content_len"] for i in range(B)]
        for li, layer_idx in enumerate(self.hidden_layers):
            per_layer = torch.zeros(B, seq_len, self._hidden_size, dtype=self.dtype)
            col0 = li * self._hidden_size
            col1 = col0 + self._hidden_size
            for i in range(B):
                content_len = jobs[i]["content_len"]
                if content_len <= 0 or i not in results:
                    continue
                per_layer[i, offsets[i]:, :] = results[i][:, col0:col1]
            hidden_states[layer_idx] = per_layer.to(self.device, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.last_assemble_secs = time.perf_counter() - _t_asm
        return hidden_states

    def _dispatch_and_collect(
        self,
        jobs: List[Dict],
        out: Dict[int, torch.Tensor],
        rids_seen: List[str],
    ) -> None:
        """Legacy synchronous send+collect (deprecated; kept for any external use).

        forward() now uses dispatch_send()/collect_and_assemble() directly.
        """
        rids = self.dispatch_send(jobs)
        rids_seen.extend(r for r in rids.values() if r)
        _t_read = time.perf_counter()
        for i, job in enumerate(jobs):
            if job["content_len"] <= 0:
                continue
            out[i] = self._load_scores(rids[i], job["content_len"])
        self.last_read_secs = time.perf_counter() - _t_read

    def _send_one(self, job: Dict) -> dict:
        """POST a single /generate request. Returns the parsed JSON body.

        Minimal payload (matches the verified-correct reference client): no
        client rid, no stream flag. The server assigns a rid and echoes it back
        in meta_info["id']. Retries on transient network errors.
        """
        payload = json.dumps(
            {
                "input_ids": job["content_ids"],
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_err = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as err:
                last_err = err
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(
            f"[SGLang] /generate failed after {self.retries} attempts: {last_err}"
        )

    @staticmethod
    def _extract_rid(resp: dict) -> str:
        """Pull the server-assigned rid from a /generate response body."""
        try:
            return resp["meta_info"]["id"]
        except (KeyError, TypeError) as e:
            raise RuntimeError(
                f"[SGLang] response missing meta_info.id (got: {resp!r}): {e}"
            )

    def _load_scores(self, rid: str, content_len: int) -> torch.Tensor:
        """Poll for <save_dir>/<rid>.safetensors, read `scores`, delete it.

        Mirrors the reference client: a bounded retry loop (the file may not be
        flushed yet), then one final blocking read; strict shape asserts; the
        file is removed immediately after a successful read.
        """
        from safetensors.torch import load_file

        path = os.path.join(self.save_dir, f"{rid}.safetensors")

        scores = None
        last_err = None
        for _ in range(self.file_wait_tries):
            try:
                # Read the whole file once so we load from a single consistent
                # snapshot (avoids a half-written header on unlucky timing).
                with open(path, "rb") as f:
                    raw = f.read()
                scores = self._load_scores_bytes(raw)
                break
            except (FileNotFoundError, SafetensorError) as e:
                last_err = e
                time.sleep(self.file_wait_secs)
            except Exception as e:
                # Partial-write / concurrent-access races can surface as other
                # errors; retry rather than abort the batch.
                last_err = e
                time.sleep(self.file_wait_secs)

        if scores is None:
            # Final blocking read (reference behavior): if still gone, surface.
            try:
                scores = load_file(path)["scores"]
            except Exception as e:
                raise RuntimeError(
                    f"[SGLang] timed out waiting for dump {path} "
                    f"(file_wait_tries={self.file_wait_tries}, "
                    f"content_len={content_len}, last_error={e or last_err}). "
                    f"Check the SGLang server log; the request may have failed."
                )

        # Strict shape validation (matches the reference): the dump must hold
        # exactly the prompt tokens we sent and exactly our feature columns.
        assert scores.shape[1] == self._feature_dim, (
            f"[SGLang] dump feature dim mismatch: expected {self._feature_dim}, "
            f"got {scores.shape[1]} (rid={rid})"
        )
        assert scores.shape[0] == content_len, (
            f"[SGLang] dump token count mismatch: dump has {scores.shape[0]} "
            f"tokens, sent {content_len} (rid={rid})"
        )

        # Delete immediately after a successful read so the save dir holds only
        # in-flight files (steady-state empty between batches).
        try:
            os.remove(path)
        except OSError as e:
            print(f"[SGLang] WARNING: probe remove failed: {path} -> {e!r} "
                  f"(exists={os.path.exists(path)})")

        return scores.to(self.dtype)

    @staticmethod
    def _load_scores_bytes(raw: bytes) -> torch.Tensor:
        """Load the `scores` tensor from raw safetensors bytes.

        safetensors 0.8 has no `load_bytes` in the torch namespace, so we use
        `safetensors.torch.load` on the bytes (same effect as the reference's
        `load_bytes`).
        """
        from safetensors.torch import load
        return load(raw)["scores"]

    def _cleanup_rids(self, rids: List[str]) -> None:
        """Remove any leftover dump files for the given rids (best-effort)."""
        for rid in rids:
            if not rid:
                continue
            path = os.path.join(self.save_dir, f"{rid}.safetensors")
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass


def _test_client():
    """Smoke test: requires a running SGLang probe server
    (e.g. the one scripts/run_train_pipeline.sh launches).

    Sends one short prompt and checks the returned per-layer hidden states have
    the expected shape, and that the dump file was deleted after read.
    """
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:6000")
    p.add_argument("--save-dir", default="/dev/shm/save_probe")
    p.add_argument("--probe-ckpt", default="path/to/probe_ckpt")
    p.add_argument("--hidden-layers", default="13,26,39")
    args = p.parse_args()

    hidden_layers = [int(x) for x in args.hidden_layers.split(",")]

    # Derive hidden_size from probe config for the assertion. Supports both the
    # new ProbeConfig schema (hidden_dim) and the legacy one (hidden_size).
    with open(os.path.join(args.probe_ckpt, "config.json")) as f:
        _probe_cfg = json.load(f)
    hidden_size = int(_probe_cfg.get("hidden_dim") or _probe_cfg["hidden_size"])

    client = SGLangHiddenStateClient(
        model_name="sglang-server",
        hidden_layers=hidden_layers,
        url=args.url,
        save_dir=args.save_dir,
        probe_ckpt=args.probe_ckpt,
        hidden_size=hidden_size,
    )

    # A trivial prompt: token ids [1,2,3,4,5], left-padded to seq_len=8.
    seq_len = 8
    ids = torch.zeros(1, seq_len, dtype=torch.long)
    ids[0, -5:] = torch.tensor([1, 2, 3, 4, 5])
    mask = torch.zeros(1, seq_len, dtype=torch.long)
    mask[0, -5:] = 1

    hs = client(ids, mask)

    print("Returned layers:", sorted(hs.keys()))
    for idx, t in hs.items():
        assert idx in hidden_layers, f"unexpected layer {idx}"
        assert t.shape == (1, seq_len, hidden_size), \
            f"layer {idx} bad shape {t.shape}, expected (1,{seq_len},{hidden_size})"
        # Pad positions must be zero.
        assert torch.all(t[0, :seq_len - 5] == 0), f"layer {idx} pad not zero"
        # Content positions must be non-trivial (not all zero).
        assert t[0, -5:].abs().sum() > 0, f"layer {idx} content is all zero"
        print(f"  layer {idx}: shape={tuple(t.shape)} "
              f"content_norm={t[0,-5:].float().norm().item():.3f}")

    leftovers = [f for f in os.listdir(args.save_dir) if f.endswith(".safetensors")]
    assert not leftovers, f"dump files not cleaned up: {leftovers}"
    print("✅ SGLang hidden-state client smoke test passed (dump file cleaned up).")


if __name__ == "__main__":
    _test_client()