#!/usr/bin/env python3
"""GuardAttnProbe -- "attn" token-probe Guard (causal MQA + o_proj + query-residual).

The "attn" Guard architecture. The Base Model's tapped layer features (per-layer
``rmsnorm(hidden_states + residual)``, dumped by the SGLang identity probe; see
models/sglang_client.py) are joined along the feature dim and projected by
separate ``proj_q`` / ``proj_k`` / ``proj_v`` (bias-free linears reading the
whole concatenated row), run through CAUSAL MULTI-QUERY ATTENTION along the
token sequence (``num_query_heads`` query heads share ONE K/V head), and then:

  1. an attention OUTPUT projection ``o_proj`` (bias-free Linear
     q_dim -> q_dim, the standard attention ``W_O``) transforms the SDPA
     context;
  2. an additive RESIDUAL uses the QUERY as the skip connection --
     ``h = q + o`` where ``q`` is the bias-free ``proj_q`` output and ``o`` is
     the ``o_proj``-transformed context. The query -- not the input features --
     is what is added back, so the probe cannot wash out each token's own
     signal under sequence mixing;
  3. a POST-RESIDUAL RMSNorm on ``h`` right before the classifier
     (``norm = nn.RMSNorm(q_dim, eps=1e-6)``), so the per-token classifier
     input is scale-stable. This is the only normalization in the Guard: there
     is NO pre-projection norm / reduction / activation -- the SGLang dump path
     already rms-normalizes ``(hidden+residual)`` per tapped layer, so the raw
     joined features go straight into proj_q/k/v.

WEIGHT CONTRACT (the convert script's injected modeling template reproduces
this layout exactly):
  - ``proj_q`` / ``proj_k`` / ``proj_v`` are SEPARATE bias-free linears
    (state-dict keys ``proj_q.weight`` / ``proj_k.weight`` / ``proj_v.weight``).
  - ``o_proj`` is a bias-free Linear (state-dict key ``o_proj.weight``).
  - ``norm`` is a single ``nn.RMSNorm(q_dim, eps=1e-6)`` (state-dict key
    ``norm.weight``) applied to the residual stream right before the
    classifier.
  - The classifier (state-dict keys ``classifier.weight`` + ``classifier.bias``)
    emits LOGITS (no sigmoid); the training loss (BCEWithLogitsLoss) applies
    the sigmoid+nll.

Input / output contract (matches the training framework's Guard interface):
  - Input:  [batch, seq_len, hidden_dim * num_layers]   (layer features
           CONCATENATED along the feature dim, as built by train.py from the
           SGLang identity-probe dumps)
  - Output: [batch, seq_len, num_classes]               (per-token LOGITS)

Args:
    hidden_dim: Per-layer hidden dimension of the Base Model.
    num_layers: Number of Base Model layers being aggregated. Sets the
        concatenated input width ``num_layers * hidden_dim``.
    num_classes: Output dimension per token (default 10: 8 query + 2 response).
    num_query_heads: Number of attention query heads. Defaults to 1. Each head
        has width ``head_dim``; the classifier input is
        ``num_query_heads * head_dim``.
    head_dim: Per-head width shared by Q, K, V. Defaults to 64. K/V always use
        a single head (MQA) regardless of ``num_query_heads``.
    sliding_window: Causal attention window size in tokens. 0 (default) = full
        causal attention. A positive W bounds attention to the last W tokens.
    init_bias: Constant init value for the classifier's bias. A negative value
        makes the Guard start from low positive-class predictions (sigmoid of
        -5 ~= 0.0067), so the model learns to "say no" by default and only
        flips to positive once there's evidence.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GuardAttnProbe(nn.Module):
    """"attn" token-probe Guard: causal MQA + o_proj + query-residual + post-norm.

    See module docstring for the full weight contract. The forward projects the
    joined tapped-layer features with separate proj_q/k/v, runs causal
    multi-query attention (optional sliding window), applies o_proj to the SDPA
    context, adds the query back as the residual, post-norms, and feeds a
    per-token linear classifier. NO input normalization (features are already
    rmsnormed at tap time on the SGLang side).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_classes: int = 10,
        num_query_heads: int = 1,
        head_dim: int = 64,
        sliding_window: int = 0,
        init_bias: float = 0.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_query_heads < 1:
            raise ValueError("num_query_heads must be >= 1")
        if head_dim < 1:
            raise ValueError("head_dim must be >= 1")
        if sliding_window < 0:
            raise ValueError("sliding_window must be >= 0 (0 = full causal)")
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")

        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_classes = int(num_classes)
        self.num_query_heads = int(num_query_heads)
        self.head_dim = int(head_dim)
        self.sliding_window = int(sliding_window)
        self.init_bias = float(init_bias)

        # Concatenated tapped-layer feature width; each of q/k/v reads the
        # whole row.
        self._d_in = self.num_layers * self.hidden_dim
        self._q_dim = self.num_query_heads * self.head_dim

        # Separate q/k/v (bias=False). ONE shared K/V head regardless of query
        # heads (MQA).
        self.proj_q = nn.Linear(self._d_in, self._q_dim, bias=False)
        self.proj_k = nn.Linear(self._d_in, self.head_dim, bias=False)
        self.proj_v = nn.Linear(self._d_in, self.head_dim, bias=False)

        # Attention OUTPUT projection (bias-free Linear q_dim -> q_dim), the
        # standard attention W_O.
        self.o_proj = nn.Linear(self._q_dim, self._q_dim, bias=False)

        # Post-residual RMSNorm on the q-stream, right before the classifier.
        # eps=1e-6 matches the Base Model/Llama-family RMSNorm the SGLang
        # identity probe taps. The ONLY normalization in this Guard -- there is
        # no pre-projection norm.
        self.norm = nn.RMSNorm(self._q_dim, eps=1e-6)

        # Linear per-token classifier (bias=True). Emits LOGITS.
        self.classifier = nn.Linear(self._q_dim, self.num_classes, bias=True)
        nn.init.constant_(self.classifier.bias, self.init_bias)

    # ------------------------------------------------------------- API parity

    @property
    def input_dim(self) -> int:
        """Concatenated tapped-layer feature width (num_layers * hidden_dim)."""
        return self._d_in

    @property
    def q_dim(self) -> int:
        """Query-side output width (num_query_heads * head_dim)."""
        return self._q_dim

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ----------------------------------------------------------------- forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Causal MQA + o_proj + query-residual + post-norm -> per-token logits.

        Args:
            x: [B, S, num_layers * hidden_dim] -- concatenated tapped+normed
               layer features (as produced by train.py from the SGLang dumps).

        Returns:
            [B, S, num_classes] per-token logits (sigmoid applied by the loss
            in training, by the eval path at inference).
        """
        # Match the param dtype so the Guard runs in the cast dtype selected by
        # train.py (bf16 by default).
        param_dtype = self.classifier.weight.dtype
        x = x.to(param_dtype)

        B, S, FD = x.shape
        expected = self._d_in
        if FD != expected:
            raise ValueError(
                f"Input feature dim {FD} != num_layers*hidden_dim "
                f"({self.num_layers}*{self.hidden_dim}={expected}). "
                f"Check base_model.hidden_layers / hidden_size config."
            )

        # q: [B, S, H*hd]  k/v: [B, S, hd]  (one shared KV head). q_feat is kept
        # in feature form as the RESIDUAL -- it is added back to the
        # o_proj-transformed attention output, so the per-token query signal
        # survives sequence mixing. (q goes into attention in its heads form,
        # q_feat stays in [B, S, q_dim] for the add.)
        q_feat = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        # Reshape to heads and transpose to [B, heads, S, hd]. Q has
        # num_query_heads; K/V keep a single head -- SDPA's enable_gqa handles
        # the head mismatch without the O(H) memory blow-up of repeat.
        q = q_feat.view(B, S, self.num_query_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, 1, self.head_dim).transpose(1, 2)
        v = v.view(B, S, 1, self.head_dim).transpose(1, 2)

        # Causal MQA. SDPA's default scale is 1/sqrt(head_dim). On GPU this
        # hits the flash / memory-efficient kernels, so the S*S matrix is never
        # materialized (unlike sliding_window, which materializes a per-block
        # mask). When Q has multiple heads but K/V have one (MQA), use the
        # native enable_gqa=True path -- it broadcasts the single KV head
        # without an O(H) memory copy. (train.py disables the cuDNN SDPA
        # backend at startup so SDPA lands on flash, the bf16+GQA path that
        # does not abort on Hopper -- see _configure_sdpa_backend in train.py.)
        if self.sliding_window and self.sliding_window < S:
            context = self._sliding_window_attention(q, k, v, S)
        elif self.num_query_heads != 1:
            context = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=True
            )
        else:
            context = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Heads back to feature dim: [B, S, H*hd], then the o_proj (W_O)
        # transforms the attention output, then the QUERY is added back as the
        # residual, then post-norm feeds the per-token classifier.
        o = context.transpose(1, 2).contiguous().view(B, S, self._q_dim)
        o = self.o_proj(o)                       # [B, S, q_dim] attention W_O
        h = q_feat + o                           # query-as-residual
        return self.classifier(self.norm(h))     # [B, S, num_classes] logits

    def _sliding_window_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, S: int
    ) -> torch.Tensor:
        """Causal sliding-window MQA via non-overlapping query blocks.

        Splits the sequence into consecutive query blocks of size
        ``sliding_window`` (the last block may be shorter); for query block
        [s, e) the attended key block is [max(0, s-W+1), e), masked so token t
        only sees keys in [t-W+1, t]. Each block's SDPA call materializes at
        most O(B * H * W * W) scores, so peak attention memory scales with the
        window, not the full length.

        enable_gqa is propagated per block so the single KV head broadcasts to
        every query head (matches the full-causal path).
        """
        W = self.sliding_window
        use_native_gqa = self.num_query_heads != 1
        device = q.device
        outs = []
        for s in range(0, S, W):
            e = min(s + W, S)
            ks = max(0, s - W + 1)          # earliest key still within the window
            ke = e
            Lq, Lkv = e - s, ke - ks
            # Per-block causal+window boolean mask (True = attend). The KV
            # block starts at ks (offset), so is_causal cannot be used here.
            idxq = torch.arange(Lq, device=device) + s
            idxkv = torch.arange(Lkv, device=device) + ks
            allow = (idxq[:, None] >= idxkv[None, :]) & (
                idxkv[None, :] >= idxq[:, None] - W + 1
            )
            mask = allow.view(1, 1, Lq, Lkv)
            qb, kb, vb = q[:, :, s:e], k[:, :, ks:ke], v[:, :, ks:ke]
            outs.append(
                F.scaled_dot_product_attention(
                    qb, kb, vb, attn_mask=mask,
                    enable_gqa=use_native_gqa or None,
                )
            )
        return torch.cat(outs, dim=2)


# Backward-compatibility alias (parity with other arch modules' Net alias).
GuardAttnProbeNet = GuardAttnProbe


if __name__ == "__main__":
    # --- End-to-end forward + backward smoke test (training path) ---
    torch.manual_seed(0)
    hidden_dim, num_layers = 8, 3
    model = GuardAttnProbe(
        hidden_dim=hidden_dim, num_layers=num_layers,
        num_classes=10, num_query_heads=4, head_dim=16,
    ).train()
    x = torch.randn(4, 64, hidden_dim * num_layers, requires_grad=False)
    y = model(x)
    print(f"[forward] output shape {tuple(y.shape)} (expected (4, 64, 10))")
    assert y.shape == (4, 64, 10)
    assert torch.isfinite(y).all(), "non-finite values in output"

    loss = y.float().sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradients"
    # Weight keys -- the convert key-disambiguator + the injected modeling
    # template both rely on these EXACT names. o_proj.weight + norm.weight
    # belong to the o_proj + query-residual + post-norm tail.
    keys = set(model.state_dict().keys())
    expected_keys = {
        "proj_q.weight", "proj_k.weight", "proj_v.weight",
        "o_proj.weight", "norm.weight",
        "classifier.weight", "classifier.bias",
    }
    assert keys == expected_keys, f"state_dict keys {sorted(keys)} != {sorted(expected_keys)}"
    assert model.proj_q.weight.shape == (
        model.num_query_heads * model.head_dim, model.input_dim
    )
    assert model.proj_k.weight.shape == (model.head_dim, model.input_dim)
    assert model.proj_v.weight.shape == (model.head_dim, model.input_dim)
    assert model.o_proj.weight.shape == (model.q_dim, model.q_dim)
    assert model.norm.weight.shape == (model.q_dim,)
    assert model.classifier.weight.shape == (model.num_classes, model.q_dim)
    print("[backward] forward + backward OK, all gradients finite; keys match the attn contract")
    print(f"[keys] {sorted(keys)}")

    # --- Causal-mask sanity check: prefix outputs must not change when a suffix
    # token is appended (position t depends only on positions <= t). The
    # query-residual (q_feat is per-token) and post-norm are both pointwise, so
    # causality is inherited from the causal MQA. ---
    model.eval()
    with torch.no_grad():
        x2 = torch.randn(1, 8, hidden_dim * num_layers)
        y_prefix = model(x2[:, :5])               # first 5 tokens
        y_full = model(x2)                        # all 8 tokens
        max_diff = (y_prefix - y_full[:, :5]).abs().max().item()
    print(f"[causal] max abs diff (prefix vs full first-5): {max_diff:.3e}")
    assert max_diff < 1e-5, f"causal mask violated: prefix outputs changed by {max_diff}"
    print("[causal] causal MQA verified (prefix outputs stable under suffix append)")

    # --- Sliding-window correctness: with window W, the first W tokens' window
    # already spans all preceding tokens, so their output matches full-causal;
    # the window path must also run without OOM at a longer-than-W sequence and
    # produce finite grads. ---
    W = 16
    model_sw = GuardAttnProbe(
        hidden_dim=hidden_dim, num_layers=num_layers,
        num_classes=10, num_query_heads=4, head_dim=16, sliding_window=W,
    ).eval()
    # Copy weights from the full-causal model so the two are directly comparable.
    model_sw.load_state_dict(model.state_dict())
    with torch.no_grad():
        xs = torch.randn(1, 48, hidden_dim * num_layers)
        y_full = model.eval()(xs)            # full-causal reference (same weights)
        y_win = model_sw(xs)                 # sliding-window
        # First W tokens are unaffected by clipping (their window covers all
        # preceding tokens), so they must match exactly.
        head_diff = (y_full[:, :W] - y_win[:, :W]).abs().max().item()
    print(f"[sliding-window] first-W window vs full-causal max abs diff: {head_diff:.3e}")
    assert head_diff < 1e-4, f"sliding-window diverged on first W tokens: {head_diff}"
    # A backward pass through the window path must produce finite grads.
    model_sw.train()
    out_sw = model_sw(xs)
    out_sw.float().sum().backward()
    grads_sw = [p.grad for p in model_sw.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads_sw), "non-finite window-path gradients"
    print("[sliding-window] window path matches full-causal on first W tokens + finite grads")

    # --- num_query_heads=1 sanity: full-causal path without enable_gqa must run
    # and produce finite logits. ---
    model_h1 = GuardAttnProbe(
        hidden_dim=hidden_dim, num_layers=num_layers,
        num_classes=10, num_query_heads=1, head_dim=16,
    ).eval()
    with torch.no_grad():
        y_h1 = model_h1(xs)
    assert y_h1.shape == (1, 48, 10) and torch.isfinite(y_h1).all()
    print("[heads=1] single-query-head full-causal path OK")

    # --- o_proj + residual actually wired: with o_proj zeroed, o=0 so logits
    # come ONLY from q_feat through the norm+classifier (proves the residual
    # path reaches the output). ---
    with torch.no_grad():
        model_o0 = GuardAttnProbe(
            hidden_dim=hidden_dim, num_layers=num_layers,
            num_classes=10, num_query_heads=4, head_dim=16,
        ).eval()
        nn.init.zeros_(model_o0.o_proj.weight)     # o == 0 identically
        model_o0.classifier.load_state_dict(model.classifier.state_dict())
        model_o0.norm.load_state_dict(model.norm.state_dict())
        y_o0 = model_o0(xs)
        # logits[0] must be classifier(norm(q_feat[0])) -- independent of keys,
        # so the same answer regardless of how the seq differs past token 0.
        qf0 = model_o0.proj_q(xs[:, 0:1])
        ref0 = model_o0.classifier(model_o0.norm(qf0))
        res_diff = (y_o0[:, 0:1] - ref0).abs().max().item()
    print(f"[residual] o_proj=0 -> logits==classifier(norm(q_feat)) max abs diff: {res_diff:.3e}")
    assert res_diff < 1e-4, f"residual path check failed: {res_diff}"
    print("[residual] query-residual path verified (o_proj=0 => output = classifier(norm(q_feat)))")

    print("\n[ok] All checks passed")
