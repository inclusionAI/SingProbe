"""
Utility functions for dataset processing

Core functions:
1. calculate_response_boundaries_with_padding: Find Query/Response boundaries with left padding
2. map_char_to_token_labels: Convert character-level offsets to token-level labels
"""

from typing import Dict, List, Optional, Tuple
import warnings
import torch
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Chat-template rendering (DeepSeek-V4 official encoder vs. HF apply_chat_template)
# ---------------------------------------------------------------------------
# DeepSeek-V4-Flash is the motivating case: its tokenizer_config.json sets
# tokenizer_class=PreTrainedTokenizerFast but carries NO `chat_template` field
# and ships no chat_template.jinja, so tokenizer.apply_chat_template(...) raises
#   ValueError: Cannot use chat template functions because
#   tokenizer.chat_template is not set and no template argument was passed!
# On the sglang backend the pipeline ships the tokenized input_ids straight to
# SGLang (no server-side template), so the token stream MUST be the exact prompt
# format the model was trained on. The simplified jinja fallback we used before
# dropped the official `` marker and ignored tool/thinking handling.
#
# render_chat_template_text() is the single entry point every call site uses:
#   - for a DeepSeek-V4 tokenizer -> delegates to the OFFICIAL encoder
#     models/utils/encoding_dsv4.encode_messages (byte-for-byte the V4 prompt);
#   - otherwise -> normal HF apply_chat_template (installing a generic text
#     fallback only when the tokenizer ships no chat_template at all).
# encode_messages always emits the assistant content verbatim, so the
# rfind(response_text) boundary lookup used downstream still works.

# Special-role markers that mark a tokenizer as DeepSeek-V4. These are the same
# tokens the V4 added vocab exposes and encode_messages wraps in its prompt.
_DSV4_SPECIAL_TOKENS = (
    "<｜User｜>",
    "<｜Assistant｜>",
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
)

_GENERIC_FALLBACK_CHAT_TEMPLATE = (
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}{{- 'User: ' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}{{- 'Assistant: ' + message['content'] + '\n' }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{- 'Assistant: ' }}{% endif %}"
)


def _is_deepseek_v4_tokenizer(tokenizer: AutoTokenizer) -> bool:
    """True iff `tokenizer` is a DeepSeek-V4 tokenizer rendered by the official encoder.

    A V4 tokenizer ships NO `chat_template` but registers the DeepSeek special role
    markers in its added vocab. Tokenizers that ship their own chat_template (Qwen,
    Gemma, Llama, ...) and non-V4 DeepSeek tokenizers that happen to expose these
    markers stay on the HF path.
    """
    if getattr(tokenizer, "chat_template", None):
        return False
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if not callable(get_added_vocab):
        return False
    try:
        added_vocab = get_added_vocab() or {}
    except Exception:
        return False
    return all(tok in added_vocab for tok in _DSV4_SPECIAL_TOKENS)


def install_fallback_chat_template(tokenizer: AutoTokenizer) -> None:
    """Set a generic chat_template on `tokenizer` if it has none.

    Used only for non-DSV4 tokenizers that ship no chat_template at all (so
    apply_chat_template works). DeepSeek-V4 is handled before reaching here by
    render_chat_template_text, which renders it with the official encoder.
    Idempotent: a tokenizer that already has a chat_template is returned untouched.
    """
    if getattr(tokenizer, "chat_template", None):
        return
    tokenizer.chat_template = _GENERIC_FALLBACK_CHAT_TEMPLATE
    warnings.warn(
        "tokenizer.chat_template was unset; installed a generic fallback "
        "chat_template (model family without a shipped chat_template). "
        "DeepSeek-V4 is rendered by the official encoder, not this fallback.",
        stacklevel=2,
    )


def render_chat_template_text(
    tokenizer: AutoTokenizer,
    messages: List[Dict],
    add_generation_prompt: bool = False,
    enable_thinking: Optional[bool] = None,
) -> str:
    """Render `messages` to the prompt string the frozen base model expects.

    DeepSeek-V4 tokenizers (no chat_template + DeepSeek special markers) are
    rendered by the official models/utils/encoding_dsv4.encode_messages, so the
    token stream is the exact V4 prompt format. Every other tokenizer uses the
    normal HF apply_chat_template (with a generic fallback template installed
    only when the tokenizer ships none).

    add_generation_prompt=True requests the continuation prefix (Query-only
    prompt): the official encoder always appends the <｜Assistant｜> continuation
    after the last user message, so any trailing assistant turn is stripped first.

    enable_thinking is the Qwen-style tri-state; for DSV4 it maps to the encoder's
    thinking_mode (True -> "thinking", otherwise "chat", which is the default).
    """
    if _is_deepseek_v4_tokenizer(tokenizer):
        from models.utils.encoding_dsv4 import encode_messages

        thinking_mode = "thinking" if enable_thinking is True else "chat"
        msgs = list(messages)
        if add_generation_prompt:
            while msgs and msgs[-1].get("role") == "assistant":
                msgs.pop()
        return encode_messages(
            msgs,
            thinking_mode=thinking_mode,
            add_default_bos_token=True,
        )

    install_fallback_chat_template(tokenizer)
    if enable_thinking is None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )


def calculate_response_boundaries_with_padding(
    tokenizer: AutoTokenizer,
    messages: List[Dict],
    max_length: int = 2048,
    enable_thinking: Optional[bool] = None
) -> Dict:
    """
    Calculate precise Query and Response boundaries considering left padding

    Single tokenization pass: the full conversation text is tokenized once (with
    offset mapping) to locate the response boundary, then left-padded/truncated
    manually. This avoids the previous approach of tokenizing both the full
    conversation AND the query-only part separately.

    Args:
        tokenizer: Tokenizer with padding_side='left'
        messages: Chat messages in OpenAI format
                  [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        max_length: Maximum sequence length (<=0 = no-truncation sentinel).
        enable_thinking: Optional forward to apply_chat_template (Qwen3 thinking
            toggle). None = omit (current default behavior preserved). Some
            tokenizers reject this kwarg when not None; on TypeError/ValueError
            we retry without it.

    Returns:
        {
            'query_end_pos': int,  # Position of Query's last token
            'response_start_pos': int,  # Position of Response's first token
            'input_ids': torch.Tensor,  # Tokenized sequence [max_length]
            'attention_mask': torch.Tensor,  # Attention mask [max_length]
            'padding_end': int,  # Position where padding ends
            'has_response': bool  # Whether there's a Response
        }
    """
    has_response = any(msg['role'] == 'assistant' for msg in messages)

    # Step 1: Apply chat template (without tokenization). DeepSeek-V4 tokenizers
    # are rendered by the official encoder; others use HF apply_chat_template
    # (install_fallback_chat_template fills in a generic template when none ships).
    text = render_chat_template_text(
        tokenizer,
        messages,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )

    # Step 2: Tokenize ONCE (no padding, no truncation) with offset mapping so we
    # can locate the assistant response's first token precisely.
    try:
        enc = tokenizer(
            text,
            truncation=False,
            padding=False,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        content_ids = enc['input_ids']
        offsets = enc.get('offset_mapping')
    except (TypeError, ValueError):
        # Tokenizer doesn't support return_offsets_mapping; fall back to plain encode.
        content_ids = tokenizer.encode(text, add_special_tokens=False)
        offsets = None

    # Step 3: Determine the response-start token index within the (unpadded) content.
    # Strategy: find the character position where the assistant response content
    # begins, then map it to a token index via offsets. If that's not available
    # (no offsets or response text not found), fall back to re-tokenizing the
    # query prefix to count its tokens -- the original, slower path.
    response_start_in_content = len(content_ids)  # default: no response region found
    if has_response:
        # The assistant message content
        assistant_msg = next(m for m in messages if m['role'] == 'assistant')
        response_text = assistant_msg['content']
        # rfind avoids false matches if the response text also appears in the query
        resp_char_start = text.rfind(response_text)
        if offsets is not None and resp_char_start != -1:
            # Find the first token whose span covers (or comes right after) resp_char_start
            found = False
            for tok_idx, (s, e) in enumerate(offsets):
                if s == 0 and e == 0:
                    # special token with no text span; skip unless we're already at/after target
                    continue
                if s <= resp_char_start < e or s >= resp_char_start:
                    response_start_in_content = tok_idx
                    found = True
                    break
            if not found:
                response_start_in_content = len(content_ids)
        else:
            # Fallback: re-tokenize the query prefix (with generation prompt) to count
            query_messages = [msg for msg in messages if msg['role'] != 'assistant']
            query_text = tokenizer.apply_chat_template(
                query_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            query_tokens = tokenizer.encode(query_text, add_special_tokens=False)
            response_start_in_content = len(query_tokens)

    # Step 4: Truncate to max_length.
    # max_length <= 0 is the "no truncation" sentinel: keep the full content and
    # pad to exactly the content length (i.e. no padding -- each sample becomes a
    # variable-length [content_len] tensor). The collator then does per-batch
    # dynamic left-padding to the batch's longest sample, so no information is
    # lost and forward cost stays minimal. With a positive max_length we match
    # HuggingFace default semantics: left padding + RIGHT truncation, i.e. the
    # FIRST max_length content tokens are kept (tail dropped).
    if max_length and max_length > 0:
        target_len = max_length
        if len(content_ids) > target_len:
            content_ids = content_ids[:target_len]
            # response_start_in_content is relative to the (front-anchored) content,
            # so it does not shift under right-truncation.
    else:
        target_len = len(content_ids)  # no truncation, no padding
    content_len = len(content_ids)

    # Step 5: Left-pad to target_len
    input_ids = torch.full((target_len,), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(target_len, dtype=torch.long)
    padding_end = target_len - content_len
    if content_len > 0:
        input_ids[padding_end:] = torch.tensor(content_ids, dtype=torch.long)
        attention_mask[padding_end:] = 1

    # Step 6: Compute final boundary indices in the padded space
    if content_len == 0:
        query_end_pos = 0
        response_start_pos = target_len
    elif has_response and response_start_in_content < content_len:
        response_start_pos = padding_end + response_start_in_content
        # query_end_pos = token immediately before response start
        query_end_pos = max(response_start_pos - 1, 0)
    else:
        # No Response (Query_Safety dataset): query spans all content tokens
        query_content_length = (attention_mask[padding_end:] == 1).sum().item()
        query_end_pos = padding_end + query_content_length - 1
        response_start_pos = target_len  # Invalid/unused

    return {
        'query_end_pos': query_end_pos,
        'response_start_pos': response_start_pos,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'padding_end': padding_end,
        'has_response': has_response
    }


def map_char_to_token_labels(
    tokenizer: AutoTokenizer,
    response_text: str,
    annotations: List[Dict],
    max_tokens: Optional[int] = None
) -> Tuple[List[int], int]:
    """
    Map character-level annotations to token-level labels

    Strategy:
    1. Tokenize the entire response text to get all tokens
    2. Build a mapping from token index to character range in the ORIGINAL response
    3. For each annotation, find all tokens that overlap with its character range
    4. Label overlapping tokens as hallucination (1) or not (0)

    Args:
        tokenizer: Tokenizer
        response_text: Response text (original, may be longer than truncated sequence)
        annotations: List of annotations
                     [{"index": char_offset, "label": "Supported/Not Supported/Insufficient Information", "span": text}, ...]
        max_tokens: Optional limit on number of tokens to return labels for.
                    If the response tokenizes to more than max_tokens, labels are truncated.
                    This handles the case where the response is truncated during full-sequence
                    tokenization due to max_length limits.

    Returns:
        token_labels: List of labels for each token [0, 0, 1, 1, 0, ...]
                     0 = Supported, 1 = Hallucination (Not Supported or Insufficient Information)
        total_tokens: Total number of tokens in the full response (before any truncation)
    """
    # Step 1: Tokenize the full response (not truncated)
    # This ensures correct character offset mapping
    tokens = tokenizer.encode(response_text, add_special_tokens=False)
    total_tokens = len(tokens)

    token_labels = [0] * len(tokens)  # Default: all supported (0)

    if not annotations:
        if max_tokens is not None and len(token_labels) > max_tokens:
            token_labels = token_labels[:max_tokens]
        return token_labels, total_tokens

    # Step 2: Build token-to-character-range mapping
    # We decode each token and track its character offset in the original response
    token_char_ranges = []  # [(start_char, end_char), ...]
    current_char = 0

    for token_id in tokens:
        token_text = tokenizer.decode([token_id])

        # Handle special cases: empty or invisible tokens
        if len(token_text) == 0:
            token_text = " "  # Default to one character

        start_char = current_char
        end_char = current_char + len(token_text)
        token_char_ranges.append((start_char, end_char))
        current_char = end_char

    # Step 3: Map annotations to tokens
    for ann in annotations:
        char_start = ann['index']
        label = ann['label']
        span_text = ann['span']
        char_end = char_start + len(span_text)

        # Only process hallucination labels
        if label in ['Not Supported', 'Insufficient Information']:
            # Find all tokens that overlap with [char_start, char_end]
            for token_idx, (tok_start, tok_end) in enumerate(token_char_ranges):
                # Check for overlap: token overlaps with annotation range
                if tok_start < char_end and tok_end > char_start:
                    token_labels[token_idx] = 1  # Mark as hallucination

    # Step 4: Truncate if needed (labels for truncated tokens are dropped)
    if max_tokens is not None and len(token_labels) > max_tokens:
        token_labels = token_labels[:max_tokens]

    return token_labels, total_tokens


def map_spans_to_token_labels(
    tokenizer: AutoTokenizer,
    response_text: str,
    spans: List[Dict],
    max_tokens: Optional[int] = None
) -> Tuple[List[int], int]:
    """
    Map response-relative character spans to token-level hallucination labels.

    Uses the tokenizer's char-level offset_mapping (produced by tokenizing the
    response text standalone) so each token's character range is exact -- unlike
    rebuild-by-decode, this is robust to leading-space / special tokens. A token
    is labeled hallucination (1) iff its character range [s, e) intersects any
    span [start, end) (half-open, README convention: response[start:end] == text).
    Matches the recommended method in
    dataset/Hallucination/README_token_level.md.

    Args:
        tokenizer: Tokenizer (must support return_offsets_mapping; fast tokenizer).
        response_text: The response string the spans are relative to.
        spans: List of {"start": int, "end": int, "text": str, ...}; clean rows
            pass an empty list -> all-zero labels.
        max_tokens: Optional cap; if the response tokenizes to more than this,
            labels are truncated (matches the legacy mapper's contract used by
            the Dataset's truncation handling).

    Returns:
        token_labels: List[int] of 0/1 per response token.
        total_tokens: Total response token count before any truncation.
    """
    # Tokenize the full response standalone (NOT truncated) so offsets are exact.
    enc = tokenizer(
        response_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = enc.get("offset_mapping") or []
    total_tokens = len(offsets)
    token_labels = [0] * total_tokens

    # Normalize spans to (start, end) int pairs; skip malformed entries.
    norm_spans = []
    for sp in spans or []:
        try:
            a = int(sp["start"])
            b = int(sp["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if b > a:
            norm_spans.append((a, b))

    if norm_spans:
        for i, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:
                # Special/zero-span token (no underlying text) -- leave 0.
                continue
            for a, b in norm_spans:
                if a < e and s < b:  # interval intersection (half-open)
                    token_labels[i] = 1
                    break

    if max_tokens is not None and len(token_labels) > max_tokens:
        token_labels = token_labels[:max_tokens]

    return token_labels, total_tokens


def verify_boundary_calculation(
    tokenizer: AutoTokenizer,
    sample: Dict,
    max_length: int = 512
) -> None:
    """
    Verification function to debug boundary calculation

    Prints:
        - Padding end position
        - Query end position
        - Response start position
        - Decoded tokens at key positions
    """
    boundaries = calculate_response_boundaries_with_padding(
        tokenizer,
        sample['messages'],
        max_length=max_length
    )

    print("=" * 80)
    print("Boundary Calculation Verification")
    print("=" * 80)
    print(f"Padding End Position: {boundaries['padding_end']}")
    print(f"Query End Position: {boundaries['query_end_pos']}")
    print(f"Response Start Position: {boundaries['response_start_pos']}")
    print(f"Has Response: {boundaries['has_response']}")

    input_ids = boundaries['input_ids']
    padding_end = boundaries['padding_end']

    # Decode first non-PAD token
    if padding_end < len(input_ids):
        first_token = tokenizer.decode([input_ids[padding_end].item()])
        print(f"\nFirst non-PAD token: '{first_token}'")

    # Decode Query end token
    query_end = boundaries['query_end_pos']
    if query_end < len(input_ids):
        query_end_token = tokenizer.decode([input_ids[query_end].item()])
        print(f"Query last token: '{query_end_token}'")

    # Decode Response start tokens
    if boundaries['has_response']:
        response_start = boundaries['response_start_pos']
        if response_start < len(input_ids):
            response_first_token = tokenizer.decode([input_ids[response_start].item()])
            print(f"Response first token: '{response_first_token}'")

            # Decode next 5 tokens
            if response_start + 5 < len(input_ids):
                response_first_5 = tokenizer.decode(
                    input_ids[response_start:response_start+5]
                )
                print(f"Response first 5 tokens: '{response_first_5}'")

    print("=" * 80)


def visualize_char_to_token_mapping(
    tokenizer: AutoTokenizer,
    response_text: str,
    annotations: List[Dict]
) -> None:
    """
    Visualization function to debug character-to-token mapping

    Prints:
        - Response text
        - Each token with its label
        - Original annotations
    """
    token_labels = map_char_to_token_labels(tokenizer, response_text, annotations)
    tokens = tokenizer.encode(response_text, add_special_tokens=False)

    print("=" * 80)
    print("Character-to-Token Mapping Visualization")
    print("=" * 80)
    print(f"Response Text: {response_text}")
    print(f"\nTotal tokens: {len(tokens)}")
    print(f"Hallucination tokens: {sum(token_labels)}")
    print("\nToken-level labels:")
    for i, (token_id, label) in enumerate(zip(tokens, token_labels)):
        token_text = tokenizer.decode([token_id])
        label_str = "HALLUCINATION" if label == 1 else "Supported"
        print(f"Token {i:3d}: '{token_text:20s}' -> {label_str}")

    print("\nOriginal Annotations:")
    for ann in annotations:
        print(f"  Char {ann['index']:3d}: '{ann['span']:20s}' -> {ann['label']}")
    print("=" * 80)