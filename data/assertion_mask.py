"""
POS assertion mask for the Hallucination task (dim 9).

Builds a per-response-token boolean mask that is True only on "content words" --
tokens whose spaCy POS tag is NOT a function/punctuation word -- so the Hallu
loss can be restricted to content tokens (function words become unsupervised).
Ported from the reference probe's `build_assertion_mask`, but the primary entry
point here (`build_assertion_mask_from_offsets`) takes the tokenizer's
`offset_mapping` over `response_text` directly (the exact call
`map_spans_to_token_labels` makes in `data/utils.py`), avoiding the reference's
lossy de-BPE (`Ġ`/`▁`) reconstruction of the surface text.

English-only: training data is English, so the spaCy backend is `en_core_web_sm`.
This module is imported LAZILY by `GuardrailDataset` only when
`use_hallu_assertion_mask` is True, so default-off runs pay no spaCy dependency.
The spaCy `nlp` object is loaded once per worker process via a module-level cache
(lazy on first call) and trimmed to the tagger pipe only (`disable` ner/parser/
lemmatizer).
"""

import os
from typing import List, Optional, Tuple

import numpy as np


# spaCy universal POS tags whose tokens are treated as non-content (function /
# punctuation) and thus masked OUT of Hallu supervision. Matches the reference
# probe's POS_BLACKLIST.
POS_BLACKLIST = {"PUNCT", "ADP", "DET", "AUX", "CCONJ", "SCONJ", "PRON", "SPACE", "PART"}

# Module-level cache for the spaCy nlp object, keyed per worker process. None
# means "not loaded yet"; a spaCy Language once loaded. This is NOT a global
# singleton across workers (each DataLoader worker process has its own), which
# is what we want -- 32 workers => 32 nlp objects, each loading once.
_NLP_CACHE = None


def get_spacy_nlp():
    """Lazily load and cache the en_core_web_sm tagger-pipeline.

    Model resolution order:
      1. `SPACY_MODEL_PATH` env var -- an explicit path to an unzipped spaCy model
         dir (e.g. 'path/to/en_core_web_sm'). Loaded via `spacy.load(path)` so
         the model does NOT need to be pip-installed globally; just downloaded to a
         directory.
      2. `SPACY_MODEL_NAME` env var -- a globally-installed model package name
         (e.g. 'en_core_web_sm').
      3. The default package name 'en_core_web_sm' (requires global install via
         `python -m spacy download en_core_web_sm`).

    Loads once per process (returns the cached object thereafter). The pipeline is
    trimmed to the tagger (+ attribute_ruler it depends on); ner/parser/lemmatizer
    are disabled for speed. Raises a clear ImportError if spaCy or the model are
    unavailable (this only happens when the assertion-mask flag is ON, so
    default-off runs never hit it).
    """
    global _NLP_CACHE
    if _NLP_CACHE is not None:
        return _NLP_CACHE
    try:
        import spacy
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "spaCy is required for use_hallu_assertion_mask=True "
            "(POS assertion mask). Install it with: pip install spacy "
            "&& python -m spacy download en_core_web_sm"
        ) from e

    # Resolve the model spec as described above. Prefer an explicit path so the
    # model can live anywhere without a global pip install.
    model_path = os.environ.get("SPACY_MODEL_PATH")
    model_name = os.environ.get("SPACY_MODEL_NAME") or "en_core_web_sm"
    if model_path:
        try:
            nlp = spacy.load(model_path, disable=["ner", "parser", "lemmatizer"])
        except (OSError, Exception) as e:  # pragma: no cover - env-dependent
            raise ImportError(
                f"Could not load spaCy model from SPACY_MODEL_PATH={model_path!r}. "
                f"Set it to a valid (unzipped) en_core_web_sm dir, or unset it to "
                f"fall back to the installed package '{model_name}'."
            ) from e
    else:
        try:
            nlp = spacy.load(model_name, disable=["ner", "parser", "lemmatizer"])
        except (OSError, Exception) as e:  # pragma: no cover - env-dependent
            raise ImportError(
                f"spaCy model {model_name!r} is required for "
                f"use_hallu_assertion_mask=True. Install it with: "
                f"python -m spacy download en_core_web_sm, or set "
                f"SPACY_MODEL_PATH to an unzipped model directory."
            ) from e
    _NLP_CACHE = nlp
    return nlp


def build_assertion_mask_from_offsets(
    response_text: str,
    offsets: List[Tuple[int, int]],
) -> np.ndarray:
    """Content-word mask per response token, from the tokenizer's offsets.

    Runs `nlp(response_text)` once to get spaCy token spans + POS tags, then for
    each HF tokenizer token (whose [s, e) char span came from `offsets`) marks it
    True iff it overlaps any spaCy token whose POS is not in POS_BLACKLIST.

    Args:
        response_text: The assistant response string (English).
        offsets: Per-token char spans (start, end) from
            tokenizer(response_text, add_special_tokens=False,
            return_offsets_mapping=True), the same source `map_spans_to_token_labels`
            uses. A (0, 0) span (special/empty token) is treated as non-content
            (False).

    Returns:
        np.ndarray[bool] of length len(offsets), True on content tokens.
    """
    n = len(offsets)
    mask = np.zeros(n, dtype=bool)
    if n == 0 or not response_text:
        # No tokens or no text -> vacuously empty mask (no opinion).
        return mask

    nlp = get_spacy_nlp()
    doc = nlp(response_text)

    # spaCy token intervals that are content words: (char_start, char_end_exclusive,
    # is_content). spaCy tokens carry .idx (char start) and len(.text).
    intervals = [
        (tok.idx, tok.idx + len(tok.text), tok.pos_ not in POS_BLACKLIST)
        for tok in doc
        if not tok.is_space and not tok.is_punct
    ]

    # Overlap-match (half-open interval intersection, same test as
    # map_spans_to_token_labels): HF token [s, e) is content if it overlaps any
    # content spaCy interval.
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            # Special / zero-span token -> not supervisable as a content word.
            continue
        for ss, se, ok in intervals:
            if ok and not (e <= ss or s >= se):
                mask[i] = True
                break
    return mask


def build_assertion_mask(token_strs: List[str]) -> np.ndarray:
    """Content-word mask from response subword token strings (reference variant).

    Reconstructs the surface text by de-BPE-ing leading `Ġ`/`▁` to a space, runs
    spaCy, then overlap-matches each subword's reconstructed char span against
    content spaCy intervals. This is the verbatim reference-probe path, kept for
    parity/testing; the repo uses `build_assertion_mask_from_offsets` which is
    exact against the tokenizer's offsets.

    Args:
        token_strs: Output of tokenizer.convert_ids_to_tokens(response_token_ids).

    Returns:
        np.ndarray[bool] of length len(token_strs), True on content tokens.
    """
    n = len(token_strs)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask

    text_parts, spans, cursor = [], [], 0
    for s in token_strs:
        if s.startswith("Ġ"):
            visible = s.replace("Ġ", " ", 1)
        elif s.startswith("▁"):
            visible = s.replace("▁", " ", 1)
        else:
            visible = s
        text_parts.append(visible)
        spans.append((cursor, cursor + len(visible)))
        cursor += len(visible)
    text = "".join(text_parts)
    if not text:
        return mask

    nlp = get_spacy_nlp()
    doc = nlp(text)
    intervals = [
        (tok.idx, tok.idx + len(tok.text), tok.pos_ not in POS_BLACKLIST)
        for tok in doc
        if not tok.is_space and not tok.is_punct
    ]
    for i, (s, e) in enumerate(spans):
        for ss, se, ok in intervals:
            if ok and not (e <= ss or s >= se):
                mask[i] = True
                break
    return mask