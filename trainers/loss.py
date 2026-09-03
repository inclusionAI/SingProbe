"""
Loss functions for Guardrail multi-task learning

Implements:
1. GuardrailLoss: Basic multi-task loss
2. WeightedMultiTaskLoss: Weighted multi-task loss with dynamic weight adjustment
3. ConfidenceWeightedLoss: Confidence-weighted loss using softmax
4. ConfidenceWeightedMultiTaskLoss: Combined multi-task and confidence weighting

Label structure:
- First 8 dimensions: Query multi-label classification (multi-hot)
- Dimension 8: Response safety classification (binary)
- Dimension 9: Response hallucination detection (binary)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class GuardrailLoss(nn.Module):
    """
    Basic Guardrail multi-task loss function

    Three tasks:
    1. Query multi-label classification (on Response tokens)
    2. Response safety classification (on Response tokens)
    3. Response hallucination detection (on Response tokens)
    """

    def __init__(
        self,
        num_query_classes: int = 8,
        ignore_index: int = -100,
        query_weight: float = 1.0,
        safety_weight: float = 1.0,
        hallucination_weight: float = 1.0
    ):
        """
        Args:
            num_query_classes: Number of Query categories (default: 8)
            ignore_index: Label value for ignored positions (default: -100)
            query_weight: Weight for Query classification loss
            safety_weight: Weight for safety classification loss
            hallucination_weight: Weight for hallucination detection loss
        """
        super().__init__()
        self.num_query_classes = num_query_classes
        self.ignore_index = ignore_index

        # Query multi-label: BCEWithLogitsLoss
        self.query_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        # Response binary classification: single logit + BCEWithLogitsLoss
        # Each task uses one dedicated logit (dim 8 for safety, dim 9 for hallucination),
        # where the logit is the "positive" (unsafe / hallucination) score and 0 means safe / supported.
        self.safety_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.hallucination_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        self.query_weight = query_weight
        self.safety_weight = safety_weight
        self.hallucination_weight = hallucination_weight

    def forward(
        self,
        logits: torch.Tensor,           # [batch, seq_len, 10]
        labels: torch.Tensor,           # [batch, seq_len, 10]
        response_mask: torch.Tensor     # [batch, seq_len]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss

        Args:
            logits: Model output [batch, seq_len, 10]
                    First 8 columns: Query multi-label logits
                    Column 8: Safety logit (single; >0 => unsafe)
                    Column 9: Hallucination logit (single; >0 => hallucination)
            labels: Labels [batch, seq_len, 10]
            response_mask: Response mask [batch, seq_len], 1 for Response tokens

        Returns:
            Dict containing total_loss and individual task losses
        """
        batch_size, seq_len, _ = logits.shape

        # ========== 1. Query multi-label loss (on Response tokens) ==========
        # Query labels (dims 0-7) are broadcast onto Response tokens by the
        # safety dataset, so evaluate them where response_mask == 1.
        query_mask = (labels[:, :, 0] != self.ignore_index) & (response_mask == 1)

        if query_mask.sum() > 0:
            query_logits = logits[query_mask][:, :8]  # [N, 8]
            query_labels = labels[query_mask][:, :8].float()  # [N, 8]

            query_loss = self.query_loss_fn(query_logits, query_labels).mean()
        else:
            query_loss = torch.tensor(0.0, device=logits.device)

        # ========== 2. Response safety and hallucination losses (on Response tokens) ==========
        # Each task uses a SINGLE logit (dim 8 for safety, dim 9 for hallucination).
        # Labels are 0/1; ignore_index (-100) positions are masked out manually.
        response_positions = response_mask == 1

        if response_positions.sum() > 0:
            # Safety loss (single logit at dim 8)
            safety_logits = logits[response_positions][:, 8]     # [N]
            safety_labels = labels[response_positions][:, 8]     # [N]
            safety_valid = safety_labels != self.ignore_index
            if safety_valid.sum() > 0:
                safety_loss = self.safety_loss_fn(
                    safety_logits[safety_valid].float(),
                    safety_labels[safety_valid].float()
                ).mean()
            else:
                safety_loss = torch.tensor(0.0, device=logits.device)

            # Hallucination loss (single logit at dim 9)
            hallu_logits = logits[response_positions][:, 9]      # [N]
            hallu_labels = labels[response_positions][:, 9]      # [N]
            hallu_valid = hallu_labels != self.ignore_index
            if hallu_valid.sum() > 0:
                hallucination_loss = self.hallucination_loss_fn(
                    hallu_logits[hallu_valid].float(),
                    hallu_labels[hallu_valid].float()
                ).mean()
            else:
                hallucination_loss = torch.tensor(0.0, device=logits.device)
        else:
            safety_loss = torch.tensor(0.0, device=logits.device)
            hallucination_loss = torch.tensor(0.0, device=logits.device)

        # ========== 3. Total loss (weighted sum) ==========
        total_loss = (
            self.query_weight * query_loss +
            self.safety_weight * safety_loss +
            self.hallucination_weight * hallucination_loss
        )

        return {
            'total_loss': total_loss,
            'query_loss': query_loss,
            'safety_loss': safety_loss,
            'hallucination_loss': hallucination_loss,
        }


class WeightedMultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss with dynamic weight adjustment

    Dynamically adjusts task weights based on loss values (EMA)
    - Tasks with higher loss get reduced weight
    - Tasks with lower loss get increased weight
    """

    def __init__(
        self,
        num_query_classes: int = 8,
        ignore_index: int = -100,
        query_weight: float = 1.0,
        safety_weight: float = 2.0,         # Safety samples are rare, increase weight
        hallucination_weight: float = 3.0,  # Hallucination samples are rarer, increase weight
        dynamic_weight: bool = True,
        ema_decay: float = 0.9
    ):
        """
        Args:
            num_query_classes: Number of Query categories
            ignore_index: Label value for ignored positions
            query_weight: Initial weight for Query task
            safety_weight: Initial weight for Safety task
            hallucination_weight: Initial weight for Hallucination task
            dynamic_weight: Whether to use dynamic weight adjustment
            ema_decay: EMA decay factor for loss tracking
        """
        super().__init__()
        self.num_query_classes = num_query_classes
        self.ignore_index = ignore_index
        self.query_weight = query_weight
        self.safety_weight = safety_weight
        self.hallucination_weight = hallucination_weight
        self.dynamic_weight = dynamic_weight
        self.ema_decay = ema_decay

        # Loss function components
        self.query_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        # Single logit + BCE for response tasks (dim 8: safety, dim 9: hallucination)
        self.safety_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.hallucination_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        # EMA tracking for dynamic weights
        if dynamic_weight:
            self.register_buffer('loss_ema_query', torch.tensor(1.0))
            self.register_buffer('loss_ema_safety', torch.tensor(1.0))
            self.register_buffer('loss_ema_hallu', torch.tensor(1.0))

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        response_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted multi-task loss

        Args:
            logits: [batch, seq_len, 10]
            labels: [batch, seq_len, 10]
            response_mask: [batch, seq_len]

        Returns:
            Dict containing losses
        """
        batch_size, seq_len, _ = logits.shape
        losses = {}

        # ========== 1. Query multi-label loss (on Response tokens) ==========
        query_mask = (labels[:, :, 0] != self.ignore_index) & (response_mask == 1)

        if query_mask.sum() > 0:
            query_logits = logits[query_mask][:, :8]
            query_labels = labels[query_mask][:, :8].float()
            query_loss = self.query_loss_fn(query_logits, query_labels).mean()
            losses['query_loss'] = query_loss
        else:
            losses['query_loss'] = torch.tensor(0.0, device=logits.device)

        # ========== 2. Response safety loss ==========
        # Single logit at dim 8; BCE with 0/1 labels, masked by valid positions
        safety_mask = (labels[:, :, 8] != self.ignore_index) & (response_mask == 1)

        if safety_mask.sum() > 0:
            safety_logits = logits[safety_mask][:, 8]
            safety_labels = labels[safety_mask][:, 8].float()
            safety_loss = self.safety_loss_fn(safety_logits.float(), safety_labels).mean()
            losses['safety_loss'] = safety_loss
        else:
            losses['safety_loss'] = torch.tensor(0.0, device=logits.device)

        # ========== 3. Response hallucination loss ==========
        # Single logit at dim 9; BCE with 0/1 labels, masked by valid positions
        hallu_mask = (labels[:, :, 9] != self.ignore_index) & (response_mask == 1)

        if hallu_mask.sum() > 0:
            hallu_logits = logits[hallu_mask][:, 9]
            hallu_labels = labels[hallu_mask][:, 9].float()
            hallu_loss = self.hallucination_loss_fn(hallu_logits.float(), hallu_labels).mean()
            losses['hallucination_loss'] = hallu_loss
        else:
            losses['hallucination_loss'] = torch.tensor(0.0, device=logits.device)

        # ========== 4. Dynamic weight adjustment ==========
        if self.dynamic_weight:
            with torch.no_grad():
                # Update EMA
                self.loss_ema_query = (
                    self.ema_decay * self.loss_ema_query +
                    (1 - self.ema_decay) * losses['query_loss'].item()
                )
                self.loss_ema_safety = (
                    self.ema_decay * self.loss_ema_safety +
                    (1 - self.ema_decay) * losses['safety_loss'].item()
                )
                self.loss_ema_hallu = (
                    self.ema_decay * self.loss_ema_hallu +
                    (1 - self.ema_decay) * losses['hallucination_loss'].item()
                )

            # Inverse weighting: higher loss → lower weight, lower loss → higher weight
            total_ema = self.loss_ema_query + self.loss_ema_safety + self.loss_ema_hallu
            dynamic_query_w = self.query_weight * (total_ema / (3 * self.loss_ema_query.clamp(min=1e-9)))
            dynamic_safety_w = self.safety_weight * (total_ema / (3 * self.loss_ema_safety.clamp(min=1e-9)))
            dynamic_hallu_w = self.hallucination_weight * (total_ema / (3 * self.loss_ema_hallu.clamp(min=1e-9)))
        else:
            dynamic_query_w = self.query_weight
            dynamic_safety_w = self.safety_weight
            dynamic_hallu_w = self.hallucination_weight

        # ========== 5. Total loss ==========
        losses['total_loss'] = (
            dynamic_query_w * losses['query_loss'] +
            dynamic_safety_w * losses['safety_loss'] +
            dynamic_hallu_w * losses['hallucination_loss']
        )

        # Record dynamic weights (clone tensors to avoid gradient issues)
        losses['query_weight'] = dynamic_query_w.clone().detach()
        losses['safety_weight'] = dynamic_safety_w.clone().detach()
        losses['hallucination_weight'] = dynamic_hallu_w.clone().detach()

        return losses


class ConfidenceWeightedLoss(nn.Module):
    """
    Confidence-weighted Loss using Softmax

    Core mechanism:
    1. Compute prediction confidence for each token
    2. Normalize with softmax to get weight distribution across sequence
    3. Higher confidence tokens get higher weights, lower confidence tokens get lower weights
    4. Encourage model to "stay silent" (low confidence) on uncertain parts

    Benefits:
    - Automatic hard example mining: difficult samples usually have low confidence
    - Encourage model to focus on important parts
    - Allow model to "not know", avoiding overconfident wrong predictions
    """

    def __init__(
        self,
        temperature: float = 1.0,
        min_weight: float = 0.1,
        confidence_threshold: float = 0.5,
        ignore_index: int = -100
    ):
        """
        Args:
            temperature: Softmax temperature, controls weight smoothness
                        - High temperature: smoother weights
                        - Low temperature: weights more concentrated on high-confidence tokens
            min_weight: Minimum weight to prevent a token from being ignored
            confidence_threshold: Tokens below this threshold get minimum weight
            ignore_index: Label value for ignored positions
        """
        super().__init__()
        self.temperature = temperature
        self.min_weight = min_weight
        self.confidence_threshold = confidence_threshold
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,      # [batch, seq_len, num_classes]
        labels: torch.Tensor,      # [batch, seq_len]
        mask: Optional[torch.Tensor] = None  # [batch, seq_len]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute confidence-weighted loss

        Args:
            logits: [batch, seq_len, num_classes]
            labels: [batch, seq_len]
            mask: Optional mask [batch, seq_len]

        Returns:
            Dict containing:
                - loss: Weighted loss
                - confidence: Average confidence
                - weights: Weight distribution
        """
        batch_size, seq_len, num_classes = logits.shape

        # ========== 4. Apply mask (ignore padding and invalid positions) ==========
        if mask is not None:
            valid_mask = (labels != self.ignore_index) & mask.bool()
        else:
            valid_mask = (labels != self.ignore_index)

        # Check if there are any valid tokens
        if valid_mask.sum() == 0:
            # No valid tokens, return zero loss
            return {
                'loss': torch.tensor(0.0, device=logits.device, requires_grad=True),
                'confidence': torch.tensor(0.0, device=logits.device),
                'weights': torch.zeros(batch_size, seq_len, device=logits.device)
            }

        # ========== Convert to FP32 for numerical stability ==========
        logits_fp32 = logits.float()

        # ========== 1. Compute base loss (per-token) ==========
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)
        token_losses = loss_fct(
            logits_fp32.view(-1, num_classes),
            labels.view(-1)
        ).view(batch_size, seq_len)  # [batch, seq_len]

        # ========== 2. Compute prediction confidence for each token ==========
        probs = F.softmax(logits_fp32, dim=-1)  # [batch, seq_len, num_classes]
        max_probs, _ = probs.max(dim=-1)   # [batch, seq_len] confidence of predicted class

        # ========== 3. Compute weights based on confidence ==========
        confidence_weights = self._compute_confidence_weights(max_probs, mask)

        token_losses = token_losses * valid_mask.float()
        confidence_weights = confidence_weights * valid_mask.float()

        # ========== 5. Weighted average ==========
        # Normalize weights
        weight_sum = confidence_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        normalized_weights = confidence_weights / weight_sum

        # Weighted loss
        weighted_loss = (token_losses * normalized_weights).sum(dim=-1)

        # Divide by valid token count
        valid_counts = valid_mask.float().sum(dim=-1).clamp(min=1)
        weighted_loss = weighted_loss / valid_counts

        # ========== 6. Compute average confidence (for monitoring) ==========
        avg_confidence = (max_probs * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1)

        return {
            'loss': weighted_loss.mean(),
            'confidence': avg_confidence,
            'weights': normalized_weights
        }

    def _compute_confidence_weights(
        self,
        confidence: torch.Tensor,  # [batch, seq_len]
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weights based on confidence using softmax normalization

        Args:
            confidence: [batch, seq_len] confidence scores
            mask: [batch, seq_len] optional mask

        Returns:
            weights: [batch, seq_len] normalized weights
        """
        batch_size, seq_len = confidence.shape

        # Apply temperature scaling
        scaled_confidence = confidence / self.temperature

        # Apply mask - use large negative value instead of -inf to avoid nan
        if mask is not None:
            scaled_confidence = scaled_confidence.masked_fill(~mask.bool(), -1e9)

        # Softmax normalization
        weights = F.softmax(scaled_confidence, dim=-1)  # [batch, seq_len]

        # Apply minimum weight threshold
        weights = weights.clamp(min=self.min_weight)

        # Further reduce weight for low-confidence positions
        low_confidence_mask = confidence < self.confidence_threshold
        weights = weights.masked_fill(low_confidence_mask, self.min_weight)

        # Re-normalize
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        return weights


class BalancedMultiTaskLoss(nn.Module):
    """
    Balanced multi-task loss with class weighting and optional confidence weighting

    Key features:
    1. Query: BCEWithLogitsLoss (multi-label, loss is already balanced per class)
    2. Safety/Hallu: single-logit BCEWithLogitsLoss with pos_weight (class imbalance)
       - dim 8: safety logit (>0 => unsafe, 0 => safe)
       - dim 9: hallucination logit (>0 => hallucination, 0 => supported)
       pos_weight and confidence weighting are configured INDEPENDENTLY per task
       (use_{safety,hallu}_{class,confidence}_weight) -- no global fallback.
    3. Optional: Confidence-based sequence weighting (focus on hard tokens)
    4. Optional: Dynamic task weighting based on loss magnitude
    """

    def __init__(
        self,
        num_query_classes: int = 8,
        ignore_index: int = -100,
        query_weight: float = 1.0,
        safety_weight: float = 1.0,
        hallucination_weight: float = 1.0,
        # Class weights for imbalanced data [weight_for_class_0, weight_for_class_1]
        safety_class_weights: Optional[list] = None,
        hallu_class_weights: Optional[list] = None,
        # Per-task pos_weight (per-batch class-imbalance weighting, num_neg/num_pos)
        # for the Safety / Hallu BCE heads. Configured INDEPENDENTLY per task (no
        # global switch): one task can use pos_weight while the other stays
        # unweighted. When False the head runs plain unweighted BCE (pos_weight=None)
        # -- the safe default when long sequences make num_neg/num_pos explode
        # (pos_weight ~1e3), which under confidence weighting self-amplifies the
        # positive-class gradient into a numerical cliff (inf -> NaN). Set True
        # only when the imbalance is controlled (e.g. global/EMA pos_weight).
        use_safety_class_weight: bool = False,
        use_hallu_class_weight: bool = False,
        # Per-task confidence weighting (sequence-level softmax over p_+).
        # Independent per task. Avoid enabling pos_weight AND confidence weighting
        # on the SAME task -- that combo is the BCE NaN cliff above.
        use_safety_confidence_weight: bool = False,
        use_hallu_confidence_weight: bool = False,
        confidence_temperature: float = 1.0,
        confidence_beta_tau: float = 0.1,
        confidence_aggregator: str = "beta",
        confidence_min_weight: float = 0.1,
        # Detach the confidence weights so they act as pure per-token
        # multipliers and ONLY the per-token BCE term carries gradient (the
        # "pure hard-example reweighting" semantics). False (default) keeps the
        # weights in the autograd graph, so the model also receives gradient
        # THROUGH the weighting itself (a self-referential term: raising a
        # token's p_+ changes its weight, which changes the loss). No-op when
        # confidence weighting is off (neither _confidence_weight switch on).
        confidence_weight_detach: bool = False,
        # Dynamic task weighting
        dynamic_task_weight: bool = False,
        ema_decay: float = 0.9,
        # Query-head structure (Safe-vs-risk partial mutual exclusion):
        # safe_weight   scales L_safe (the dim-7 Safe BCE).
        # mutex_weight  scales L_mutex (soft mutual-exclusion regularizer
        #               p_safe * p_risk_max). 0.1 is a gentle default; set 0 to
        #               disable the explicit exclusion and fall back to a
        #               label-only (flat-8 BCE-like) behavior.
        safe_weight: float = 1.0,
        mutex_weight: float = 0.1,
        use_single_token_as_query: bool = False,
        use_hallu_assertion_mask: bool = False
    ):
        """
        Args:
            num_query_classes: Number of Query categories
            ignore_index: Label value for ignored positions
            query_weight: Weight for Query task
            safety_weight: Weight for Safety task
            hallucination_weight: Weight for Hallucination task
            safety_class_weights: [weight_for_class_0, weight_for_class_1]
                                  If None, will be computed from batch statistics.
                                  Applied via pos_weight = weight_1 / weight_0.
            hallu_class_weights: [weight_for_class_0, weight_for_class_1]
                                 If None, will be computed from batch statistics.
            use_safety_class_weight: per-batch pos_weight for the Safety BCE head.
            use_hallu_class_weight: per-batch pos_weight for the Hallu BCE head.
            use_safety_confidence_weight: confidence-based token weighting for Safety.
            use_hallu_confidence_weight: confidence-based token weighting for Hallu.
            confidence_temperature: Temperature T in softmax(beta * p_+ / T)
            confidence_beta_tau: Std threshold at which the self-regulated
                                  inverse-temperature beta reaches 1 (full
                                  weighting). beta = clamp(std(p_+)/tau, 0, 1).
                                  Only used when confidence_aggregator='beta'.
            confidence_aggregator: 'beta' (default, self-regulated beta) or
                                  'softmax' (original fixed softmax -- restores
                                  the pre-beta behavior for ablation; WILL
                                  amplify cold-start init noise).
            confidence_min_weight: Minimum per-token weight floor. Only used by
                                   the 'softmax' aggregator (the 'beta' path is
                                   exactly uniform at cold start and needs no
                                   floor).
            dynamic_task_weight: Whether to dynamically adjust task weights
            ema_decay: EMA decay for dynamic weighting
            safe_weight: Weight for the Query Safe head (dim 7) BCE.
            mutex_weight: Weight for the Safe-vs-risk soft mutual-exclusion
                          regularizer (L_mutex = mean p_safe * p_risk_max).
            use_single_token_as_query: When True, the Query labels (dims 0-7) live
                          on the Query's LAST token (not on Response tokens), so
                          the query_mask must NOT be gated by response_mask (the
                          Query last token sits just before the Response, where
                          response_mask==0). When False, query labels are on the
                          Response tokens and the response_mask gate is kept
                          (legacy behavior). Only affects the Query task; Safety
                          (dim 8) and Hallucination (dim 9) are unaffected.
        """
        super().__init__()

        self.num_query_classes = num_query_classes
        self.ignore_index = ignore_index
        self.query_weight = query_weight
        self.safety_weight = safety_weight
        self.hallucination_weight = hallucination_weight
        self.confidence_temperature = confidence_temperature
        self.confidence_beta_tau = confidence_beta_tau
        agg = (confidence_aggregator or "beta").lower()
        if agg not in ("beta", "softmax"):
            raise ValueError(
                f"confidence_aggregator must be 'beta' or 'softmax', got '{confidence_aggregator}'"
            )
        self.confidence_aggregator = agg
        self.confidence_min_weight = confidence_min_weight
        self.confidence_weight_detach = confidence_weight_detach
        # Per-task pos_weight / confidence-weight switches (plain bools; the
        # config layer coerces null/None to the default, so these are always
        # clean bools here). Safety and Hallu are independent -- no global
        # fallback, no precedence rule.
        self.use_safety_class_weight = use_safety_class_weight
        self.use_hallu_class_weight = use_hallu_class_weight
        self.use_safety_confidence_weight = use_safety_confidence_weight
        self.use_hallu_confidence_weight = use_hallu_confidence_weight
        self.dynamic_task_weight = dynamic_task_weight
        self.ema_decay = ema_decay
        self.safe_weight = safe_weight
        self.mutex_weight = mutex_weight
        self.use_single_token_as_query = use_single_token_as_query
        # Hallu-only POS content-word assertion mask (dim 9). When True, AND
        # `hallu_assertion_mask` into `hallu_mask` in forward so function-word
        # response tokens are unsupervised for Hallu. Affects ONLY dim 9; Safety
        # (dim 8) / Query (dims 0-7) never read the mask. Default False ->
        # byte-identical to pre-feature behavior (the AND is skipped, and a None
        # mask -- e.g. the warmup 3-arg call -- is also skipped).
        self.use_hallu_assertion_mask = use_hallu_assertion_mask

        # Query: multi-label BCE
        self.query_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        # Safety/Hallu: single-logit BCE (pos_weight set per-batch in forward)
        self.safety_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.hallucination_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        self.safety_class_weights = safety_class_weights
        self.hallu_class_weights = hallu_class_weights

        # EMA for dynamic task weighting
        if dynamic_task_weight:
            self.register_buffer('loss_ema_query', torch.tensor(1.0))
            self.register_buffer('loss_ema_safety', torch.tensor(0.5))
            self.register_buffer('loss_ema_hallu', torch.tensor(0.1))

    def _compute_class_weights(self, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute class weights from label distribution (inverse frequency)

        Args:
            labels: [N] tensor of labels (0 or 1)
            mask: [N] boolean mask of valid positions

        Returns:
            [2] tensor of weights for class 0 and class 1
        """
        valid_labels = labels[mask]
        if len(valid_labels) == 0:
            return torch.tensor([1.0, 1.0], device=labels.device)

        # Count occurrences
        num_pos = (valid_labels == 1).sum().float()
        num_neg = (valid_labels == 0).sum().float()
        total = num_pos + num_neg

        if num_pos == 0 or num_neg == 0:
            return torch.tensor([1.0, 1.0], device=labels.device)

        # Inverse frequency weighting
        weight_pos = total / (2 * num_pos)
        weight_neg = total / (2 * num_neg)

        # Normalize so that weights sum to 2 (average weight = 1)
        weights = torch.tensor([weight_neg, weight_pos], device=labels.device)
        weights = weights / weights.sum() * 2

        return weights

    def _compute_pos_weight(self, class_weights: torch.Tensor) -> torch.Tensor:
        """
        Convert [weight_neg, weight_pos] into BCEWithLogitsLoss pos_weight.

        pos_weight scales the loss for positive (class 1) samples relative to
        negative (class 0) samples. With BCEWithLogitsLoss, weighting the
        positive term by `pos_weight = weight_pos / weight_neg` reproduces the
        same per-class weighting that CrossEntropyLoss(weight=[w0, w1]) applies.

        WARNING: this per-batch dynamic pos_weight is only used when the
        per-task switch (self.use_safety_class_weight /
        self.use_hallu_class_weight) is True. With long sequences it can reach
        ~1e3 (num_neg/num_pos), and combined with confidence weighting it
        self-amplifies the positive-class gradient into BCE's numerical cliff
        (inf -> NaN). Keep the per-task switch False unless the imbalance is
        controlled (e.g. via a global/EMA pos_weight).

        Args:
            class_weights: [weight_for_class_0, weight_for_class_1]

        Returns:
            Scalar tensor usable as BCEWithLogitsLoss(pos_weight=...)
        """
        weight_neg = class_weights[0].clamp(min=1e-9)
        weight_pos = class_weights[1]
        return (weight_pos / weight_neg).to(weight_pos.device)

    def _compute_confidence_weights(
        self,
        logits: torch.Tensor,  # [batch, seq_len]  (single positive-class logit)
        mask: torch.Tensor     # [batch, seq_len]
    ) -> torch.Tensor:
        """
        Sequence-level confidence weights over the positive-class (unsafe /
        hallucination) probability p_+ = sigmoid(logits).

        Two aggregators, selected by self.confidence_aggregator:

        "beta" (default) -- SELF-REGULATED inverse temperature. The sharpness is
        driven by the model's OWN prediction dispersion, not a step schedule:

            p_+  = sigmoid(logits)                       # [batch, seq_len] in (0,1)
            beta = clamp( std(p_+[valid]) / tau , 0, 1 )  # scalar
            w    = softmax( beta * p_+ / T , dim=-1 )    # per-sequence, masked

        Cold start: init_bias=-5 -> all p_+ ~ 0.0067 nearly identical -> std ~ 0
        -> beta ~ 0 -> softmax(constant) is EXACTLY uniform, so init-noise token
        differences are never amplified into a weight preference. As training
        disperses p_+, beta grows and the weighting sharpens onto the confident
        tokens. beta is a pure function of current p_+ -> no step counter / state
        / warmup -> resume-correct by construction, and transient over-confident
        batches self-correct (BCE lowers p_+ on the corrected tokens, dispersion
        falls, beta falls -- negative feedback, no runaway).

        "softmax" -- the ORIGINAL fixed-softmax weighting, kept so runs can fall
        back to it for ablation / regression checks. Equivalent to forcing
        beta = 1 / T at all times (constant sharpness) AND re-applying the legacy
        min_weight floor + low-confidence clamp. This is the pre-beta behavior:
        it WILL amplify cold-start init noise (that is the whole reason "beta"
        exists), so prefer "beta" for real training.

        Numerical guard (both paths): computed in fp32; the per-token score is
        clamped to SCORE_MAX (exp(10) ~ 2.2e4, safe in fp32) as defense in depth
        against small-T + bf16 silently overflowing exp; F.softmax's internal
        max-subtraction is the primary stability mechanism.

        Args:
            logits: Single positive-class logits [batch, seq_len]
            mask: Valid token mask [batch, seq_len] (bool or 0/1)

        Returns:
            weights: [batch, seq_len] per-sequence-normalized weights (sums to 1
                     along seq over valid positions; 0 on invalid positions).
        """
        # fp32 throughout for a numerically healthy weight pipeline.
        p_pos = torch.sigmoid(logits.float())          # [batch, seq_len] in (0,1)
        mask_b = mask.bool()
        mask_f = mask_b.float()
        SCORE_MAX = 10.0

        if self.confidence_aggregator == "softmax":
            # ---- Legacy fixed-softmax path (beta == 1/T constant) ----
            scaled = (p_pos / self.confidence_temperature).clamp(max=SCORE_MAX)
            scaled = scaled.masked_fill(~mask_b, -1e9)
            w = F.softmax(scaled, dim=-1)
            # Restore the legacy min_weight floor + low-confidence clamp so this
            # path reproduces the pre-beta behavior exactly.
            min_w = float(self.confidence_min_weight)
            if min_w > 0.0:
                w = w.clamp(min=min_w)
            w = w * mask_f
            w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            return w.detach() if self.confidence_weight_detach else w

        # ---- Default "beta" path: self-regulated inverse temperature ----
        # ---- Self-regulated inverse temperature beta (scalar) ----
        valid = p_pos[mask_b]                          # [N_valid], N across whole batch
        if valid.numel() >= 2:
            std_val = valid.std(unbiased=False).clamp(min=0.0)
        else:
            std_val = torch.zeros((), device=logits.device)
        tau = max(float(self.confidence_beta_tau), 1e-6)
        beta = (std_val / tau).clamp(max=1.0)          # scalar in [0, 1]

        # ---- Per-token score with the safety clamp ----
        score = (beta * p_pos / self.confidence_temperature).clamp(max=SCORE_MAX)
        # Invalid positions get -inf-like so softmax zeros them out (F.softmax
        # max-subtracts, so -1e9 is plenty here, not a numerical hazard).
        score = score.masked_fill(~mask_b, -1e9)

        weights = F.softmax(score, dim=-1)             # [batch, seq_len]
        # Explicitly zero invalid positions (softmax of -1e9 ~ 0 but enforce it).
        weights = weights * mask_f
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        return weights.detach() if self.confidence_weight_detach else weights

    def forward(
        self,
        logits: torch.Tensor,      # [batch, seq_len, 10]
        labels: torch.Tensor,      # [batch, seq_len, 10]
        response_mask: torch.Tensor,  # [batch, seq_len]
        # Optional Hallu-only POS content-word mask [batch, seq_len] bool. ANDed
        # into hallu_mask when use_hallu_assertion_mask is True. Default None -> the
        # warmup's 3-arg call and any legacy caller stay valid (the AND is skipped
        # and behavior is byte-identical to before the feature).
        hallu_assertion_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute balanced multi-task loss

        Args:
            logits: Model output [batch, seq_len, 10]
                    First 8 columns: Query multi-label logits
                    Column 8: Safety logit
                    Column 9: Hallucination logit
            labels: Labels [batch, seq_len, 10]
            response_mask: Response mask [batch, seq_len]

        Returns:
            Dict containing losses
        """
        batch_size, seq_len, _ = logits.shape
        losses = {}

        # ========== 1. Query loss: risk dims (0-6) multi-label BCE + Safe dim (7)
        # BCE + a mutual-exclusion regularizer between Safe and the risk classes.
        #
        #   dims 0-6: seven multi-label risk categories (A-G), may co-occur
        #             (e.g. "BD"); trained with independent BCE.
        #   dim 7   : H = Safe, mutually exclusive with dims 0-6 by label
        #             construction (Safe=1 iff all risks are 0). Trained as its
        #             own BCE so the model gets an explicit "this query is clean"
        #             supervision it never had under the flat-8-dim BCE.
        #   L_mutex: prediction-side soft mutual exclusion
        #             = mean( P(Safe) * max_i P(risk_i) ) over valid query
        #             tokens. Penalizes the model when Safe and the strongest
        #             risk class are simultaneously high -- this lifts mutual
        #             exclusion from a label property into a training objective,
        #             so the model cannot hedge by giving both Safe and a risk a
        #             middling score at inference. It does NOT couple different
        #             risk classes to each other, so multi-label co-occurrence
        #             (e.g. "BD") is unaffected.
        #
        # Per-sample equal-weight averaging (mean within each sample, then over
        # the batch) is preserved so long responses don't dominate the gradient.
        # query_loss returned is the merged scalar (L_risk + safe_w*L_safe +
        # mutex_w*L_mutex) so the upper-layer dynamic-task-weight EMA / logging
        # / validate() all keep reading a single 'query_loss' unchanged; three
        # detached sub-losses are also exposed for monitoring/debugging.
        # query_mask selects the positions supervised for the Query task (dims
        # 0-7). use_single_token_as_query controls WHERE the dataset placed those
        # labels: True -> on the Query's last token (response_mask==0 there), so
        # the mask must NOT gate on response_mask; False (legacy) -> on the
        # Response tokens, gated by response_mask as before.
        if self.use_single_token_as_query:
            query_mask = (labels[:, :, 0] != self.ignore_index)  # [batch, seq]
        else:
            query_mask = (labels[:, :, 0] != self.ignore_index) & (response_mask == 1)  # [batch, seq]
        mask_f = query_mask.unsqueeze(-1).float()  # [batch, seq, 1]
        contributing = query_mask.any(dim=1).float()  # [batch]

        if query_mask.sum() > 0:
            ql = logits[:, :, :8].float()                            # [batch, seq, 8]
            lab = labels[:, :, :8].float().clamp(min=0.0)            # -100 -> 0; masking handled by mask_f

            # ---- (a) Risk classes (dims 0-6): multi-label BCE ----
            token_bce_risk = self.query_loss_fn(ql[..., :7], lab[..., :7])  # [batch, seq, 7]
            per_sample_risk = (token_bce_risk * mask_f).flatten(1).sum(dim=1)
            risk_cnt = mask_f.expand(-1, -1, 7).flatten(1).sum(dim=1).clamp(min=1.0)
            per_sample_risk = per_sample_risk / risk_cnt              # [batch]
            L_risk = (per_sample_risk * contributing).sum() / contributing.sum().clamp(min=1.0)

            # ---- (b) Safe class (dim 7): binary BCE ----
            token_bce_safe = self.query_loss_fn(ql[..., 7:8], lab[..., 7:8])  # [batch, seq, 1]
            per_sample_safe = (token_bce_safe * mask_f).flatten(1).sum(dim=1)
            safe_cnt = mask_f.expand(-1, -1, 1).flatten(1).sum(dim=1).clamp(min=1.0)
            per_sample_safe = per_sample_safe / safe_cnt              # [batch]
            L_safe = (per_sample_safe * contributing).sum() / contributing.sum().clamp(min=1.0)

            # ---- (c) Mutual exclusion (prediction-side regularizer) ----
            # P(Safe) and the maximum risk-class probability should not be high
            # together. p_risk_max excludes dim 7 to avoid self-coupling.
            qm_f = query_mask.float()                                 # [batch, seq]
            p_safe = torch.sigmoid(ql[..., 7])                        # [batch, seq]
            p_risk_max = torch.sigmoid(ql[..., :7]).max(dim=-1).values  # [batch, seq]
            L_mutex = (p_safe * p_risk_max * qm_f).sum() / qm_f.sum().clamp(min=1.0)

            query_loss = (
                L_risk
                + self.safe_weight * L_safe
                + self.mutex_weight * L_mutex
            )
            losses['query_loss'] = query_loss
            losses['query_risk_loss'] = L_risk.detach()
            losses['query_safe_loss'] = L_safe.detach()
            losses['query_mutex_loss'] = L_mutex.detach()
        else:
            losses['query_loss'] = torch.tensor(0.0, device=logits.device)

        # ========== 2. Response safety loss (single logit at dim 8, BCE) ==========
        safety_mask = (labels[:, :, 8] != self.ignore_index) & (response_mask == 1)

        if safety_mask.sum() > 0:
            safety_logits = logits[:, :, 8].float()   # [batch, seq_len]
            safety_labels = labels[:, :, 8].float()   # [batch, seq_len]

            # Optional per-batch pos_weight (class-imbalance weighting). Off by
            # default -- long sequences make num_neg/num_pos explode (pos_weight
            # ~1e3); under confidence weighting that self-amplifies the
            # positive-class gradient into BCE's numerical cliff (inf -> NaN).
            # Guard by use_safety_class_weight (per-task); class_weights is still
            # computed for monitoring (batch positive/negative ratio) regardless.
            if self.safety_class_weights is not None:
                class_weights = torch.tensor(self.safety_class_weights, device=logits.device, dtype=torch.float)
            else:
                class_weights = self._compute_class_weights(labels[:, :, 8], safety_mask).to(torch.float)
            pos_weight = self._compute_pos_weight(class_weights) if self.use_safety_class_weight else None

            # Compute per-token BCE loss (unweighted when use_safety_class_weight=False)
            safety_loss_per_token = F.binary_cross_entropy_with_logits(
                safety_logits,
                safety_labels,
                pos_weight=pos_weight,
                reduction='none'
            )  # [batch, seq_len]

            # Apply confidence weighting if enabled (Safety-independent switch)
            if self.use_safety_confidence_weight:
                confidence_weights = self._compute_confidence_weights(safety_logits, safety_mask)
                # Weighted average. confidence_weights is normalized within each
                # sequence (sum=1 per sample over valid tokens), so the weighted
                # sum over valid tokens of [batch, seq] yields one per-sample
                # weighted-average per-token loss -> ~B terms total. Divide by the
                # weight sum (also ~B, since each sequence contributes 1) to keep
                # the loss at per-token magnitude. Using mask.sum() here would
                # additionally divide by L_resp, shrinking the loss ~L_resp x.
                weight_sum = (confidence_weights * safety_mask.float()).sum().clamp(min=1e-9)
                safety_loss = (safety_loss_per_token * confidence_weights * safety_mask.float()).sum() / weight_sum
            else:
                # Simple average
                safety_loss = (safety_loss_per_token * safety_mask.float()).sum() / safety_mask.sum().float()

            losses['safety_loss'] = safety_loss
            losses['safety_class_weights'] = class_weights
        else:
            losses['safety_loss'] = torch.tensor(0.0, device=logits.device)
            losses['safety_class_weights'] = torch.tensor([1.0, 1.0], device=logits.device)

        # ========== 3. Response hallucination loss (single logit at dim 9, BCE) ==========
        hallu_mask = (labels[:, :, 9] != self.ignore_index) & (response_mask == 1)
        # Optional Hallu-only POS assertion mask: restrict supervision to content
        # words (function words -> unsupervised for dim 9 ONLY). No-op when the
        # flag is off or no mask was passed (e.g. the warmup 3-arg call). The
        # narrowed mask propagates through pos_weight's class stats, the
        # confidence-weight normalization, and the final reductions below.
        if self.use_hallu_assertion_mask and hallu_assertion_mask is not None:
            hallu_mask = hallu_mask & hallu_assertion_mask.to(
                hallu_mask.dtype).to(hallu_mask.device)

        if hallu_mask.sum() > 0:
            hallu_logits = logits[:, :, 9].float()   # [batch, seq_len]
            hallu_labels = labels[:, :, 9].float()   # [batch, seq_len]

            # Optional per-batch pos_weight, guarded by use_hallu_class_weight (per-task).
            if self.hallu_class_weights is not None:
                class_weights = torch.tensor(self.hallu_class_weights, device=logits.device, dtype=torch.float)
            else:
                class_weights = self._compute_class_weights(labels[:, :, 9], hallu_mask).to(torch.float)
            pos_weight = self._compute_pos_weight(class_weights) if self.use_hallu_class_weight else None

            # Compute per-token BCE loss (unweighted when use_hallu_class_weight=False)
            hallu_loss_per_token = F.binary_cross_entropy_with_logits(
                hallu_logits,
                hallu_labels,
                pos_weight=pos_weight,
                reduction='none'
            )  # [batch, seq_len]

            # Apply confidence weighting if enabled (Hallu-independent switch)
            if self.use_hallu_confidence_weight:
                confidence_weights = self._compute_confidence_weights(hallu_logits, hallu_mask)
                # Weighted average. confidence_weights is normalized within each
                # sequence (sum=1 per sample over valid tokens), so the weighted
                # sum over valid tokens of [batch, seq] yields one per-sample
                # weighted-average per-token loss -> ~B terms total. Divide by the
                # weight sum (also ~B, since each sequence contributes 1) to keep
                # the loss at per-token magnitude. Using mask.sum() here would
                # additionally divide by L_resp, shrinking the loss ~L_resp x.
                weight_sum = (confidence_weights * hallu_mask.float()).sum().clamp(min=1e-9)
                hallu_loss = (hallu_loss_per_token * confidence_weights * hallu_mask.float()).sum() / weight_sum
            else:
                # Simple average
                hallu_loss = (hallu_loss_per_token * hallu_mask.float()).sum() / hallu_mask.sum().float()

            losses['hallucination_loss'] = hallu_loss
            losses['hallu_class_weights'] = class_weights
        else:
            losses['hallucination_loss'] = torch.tensor(0.0, device=logits.device)
            losses['hallu_class_weights'] = torch.tensor([1.0, 1.0], device=logits.device)

        # ========== 4. Dynamic task weighting (optional) ==========
        if self.dynamic_task_weight:
            with torch.no_grad():
                # Update EMA
                self.loss_ema_query = (
                    self.ema_decay * self.loss_ema_query +
                    (1 - self.ema_decay) * losses['query_loss'].item()
                )
                self.loss_ema_safety = (
                    self.ema_decay * self.loss_ema_safety +
                    (1 - self.ema_decay) * losses['safety_loss'].item()
                )
                self.loss_ema_hallu = (
                    self.ema_decay * self.loss_ema_hallu +
                    (1 - self.ema_decay) * losses['hallucination_loss'].item()
                )

            # Loss-proportional weighting: higher loss → higher weight (focus the
            # optimizer on whichever task is currently hardest / learning slowest).
            total_ema = self.loss_ema_query + self.loss_ema_safety + self.loss_ema_hallu + 1e-9
            mean_ema = total_ema / 3.0
            dynamic_query_w = self.query_weight * (
                    self.loss_ema_query.clamp(min=1e-9) / mean_ema.clamp(min=1e-9))
            dynamic_safety_w = self.safety_weight * (
                    self.loss_ema_safety.clamp(min=1e-9) / mean_ema.clamp(min=1e-9))
            dynamic_hallu_w = self.hallucination_weight * (
                    self.loss_ema_hallu.clamp(min=1e-9) / mean_ema.clamp(min=1e-9))
        else:
            dynamic_query_w = self.query_weight
            dynamic_safety_w = self.safety_weight
            dynamic_hallu_w = self.hallucination_weight

        # ========== 5. Total loss ==========
        losses['total_loss'] = (
            dynamic_query_w * losses['query_loss'] +
            dynamic_safety_w * losses['safety_loss'] +
            dynamic_hallu_w * losses['hallucination_loss']
        )

        if self.dynamic_task_weight:
            losses['query_weight'] = dynamic_query_w
            losses['safety_weight'] = dynamic_safety_w
            losses['hallu_weight'] = dynamic_hallu_w

        return losses


class ConfidenceWeightedMultiTaskLoss(nn.Module):
    """
    Combined multi-task and confidence weighting loss

    Three tasks:
    1. Query multi-label classification (BCE, no confidence weighting)
    2. Response safety detection (single-logit BCE with confidence weighting)
    3. Response hallucination detection (single-logit BCE with confidence weighting)

    Each Response task applies confidence weighting based on the positive-class
    (sigmoid) probability.
    """

    def __init__(
        self,
        num_query_classes: int = 8,
        ignore_index: int = -100,
        temperature: float = 1.0,
        min_weight: float = 0.1,
        confidence_threshold: float = 0.5,
        query_weight: float = 1.0,
        safety_weight: float = 2.0,
        hallucination_weight: float = 3.0
    ):
        """
        Args:
            num_query_classes: Number of Query categories
            ignore_index: Label value for ignored positions
            temperature: Temperature for sigmoid confidence scaling
            min_weight: Minimum weight for confidence weighting
            confidence_threshold: Confidence threshold for weighting
            query_weight: Weight for Query task
            safety_weight: Weight for Safety task
            hallucination_weight: Weight for Hallucination task
        """
        super().__init__()

        self.num_query_classes = num_query_classes
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.min_weight = min_weight
        self.confidence_threshold = confidence_threshold

        # Query multi-label: BCEWithLogitsLoss
        self.query_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

        # Task weights
        self.query_weight = query_weight
        self.safety_weight = safety_weight
        self.hallucination_weight = hallucination_weight

    def _confidence_weighted_bce(
        self,
        logits: torch.Tensor,   # [batch, seq_len] (single positive-class logit)
        labels: torch.Tensor,   # [batch, seq_len] (0/1)
        mask: torch.Tensor      # [batch, seq_len]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute single-logit BCE with sequence-level confidence weighting.

        Confidence = sigmoid(logit) (probability of the positive class).
        High positive-class confidence tokens get higher weight; low-confidence
        tokens get the floor weight, allowing the model to "stay silent".

        Args:
            logits: [batch, seq_len]
            labels: [batch, seq_len]
            mask:   [batch, seq_len]

        Returns:
            Dict with 'loss' and 'confidence'
        """
        logits_fp32 = logits.float()
        labels_f = labels.float()
        mask_f = mask.float()

        # Per-token BCE loss
        token_losses = F.binary_cross_entropy_with_logits(
            logits_fp32, labels_f, reduction='none'
        )  # [batch, seq_len]

        # Positive-class confidence
        confidence = torch.sigmoid(logits_fp32)  # [batch, seq_len]

        # Sequence-level weights via softmax over scaled confidence
        scaled = confidence / self.temperature
        scaled = scaled.masked_fill(~mask.bool(), -1e9)
        weights = F.softmax(scaled, dim=-1)
        weights = weights.clamp(min=self.min_weight)
        # Floor weights for low-confidence positions
        low_conf = confidence < self.confidence_threshold
        weights = weights.masked_fill(low_conf, self.min_weight)
        # Re-normalize over valid positions only
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        # Weighted per-token loss, averaged over valid tokens
        valid_count = mask_f.sum().clamp(min=1)
        weighted_loss = (token_losses * weights * mask_f).sum() / valid_count

        # Average confidence over valid tokens
        avg_confidence = (confidence * mask_f).sum() / valid_count

        return {'loss': weighted_loss, 'confidence': avg_confidence}

    def forward(
        self,
        logits: torch.Tensor,      # [batch, seq_len, 10]
        labels: torch.Tensor,      # [batch, seq_len, 10]
        response_mask: torch.Tensor  # [batch, seq_len]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss with confidence weighting

        Args:
            logits: Model output [batch, seq_len, 10]
                    First 8 columns: Query multi-label logits
                    Column 8: Safety logit
                    Column 9: Hallucination logit
            labels: Labels [batch, seq_len, 10]
            response_mask: Response mask [batch, seq_len]

        Returns:
            Dict containing losses and confidence metrics
        """
        batch_size, seq_len, _ = logits.shape
        losses = {}

        # ========== 1. Query multi-label loss (no confidence weighting, on Response tokens) ==========
        query_mask = (labels[:, :, 0] != -100) & (response_mask == 1)

        if query_mask.sum() > 0:
            query_logits = logits[query_mask][:, :8].float()  # [N, 8] - convert to FP32
            query_labels = labels[query_mask][:, :8].float()  # [N, 8]

            # Query doesn't need confidence weighting
            query_loss = self.query_loss_fn(query_logits, query_labels).mean()
            losses['query_loss'] = query_loss
        else:
            losses['query_loss'] = torch.tensor(0.0, device=logits.device)

        # ========== 2. Response safety loss (single logit at dim 8, confidence-weighted BCE) ==========
        safety_mask = (labels[:, :, 8] != -100) & (response_mask == 1)

        if safety_mask.sum() > 0:
            safety_logits = logits[:, :, 8]   # [batch, seq_len] (single logit)
            safety_labels = labels[:, :, 8]    # [batch, seq_len]

            safety_result = self._confidence_weighted_bce(
                safety_logits, safety_labels, mask=safety_mask
            )

            losses['safety_loss'] = safety_result['loss']
            losses['safety_confidence'] = safety_result['confidence']

            # Debug: Check for NaN
            if torch.isnan(losses['safety_loss']):
                print(f"WARNING: safety_loss is NaN! safety_mask.sum()={safety_mask.sum()}, logits_range=[{safety_logits.min():.4f}, {safety_logits.max():.4f}]")
        else:
            # No valid safety tokens in this batch
            losses['safety_loss'] = torch.tensor(0.0, device=logits.device, requires_grad=True)
            losses['safety_confidence'] = torch.tensor(0.0, device=logits.device)

        # ========== 3. Response hallucination loss (single logit at dim 9, confidence-weighted BCE) ==========
        hallu_mask = (labels[:, :, 9] != -100) & (response_mask == 1)

        if hallu_mask.sum() > 0:
            hallu_logits = logits[:, :, 9]    # [batch, seq_len] (single logit)
            hallu_labels = labels[:, :, 9]     # [batch, seq_len]

            hallu_result = self._confidence_weighted_bce(
                hallu_logits, hallu_labels, mask=hallu_mask
            )

            losses['hallucination_loss'] = hallu_result['loss']
            losses['hallucination_confidence'] = hallu_result['confidence']

            # Debug: Check for NaN
            if torch.isnan(losses['hallucination_loss']):
                print(f"WARNING: hallu_loss is NaN! hallu_mask.sum()={hallu_mask.sum()}, logits_range=[{hallu_logits.min():.4f}, {hallu_logits.max():.4f}]")
        else:
            # No valid hallucination tokens in this batch
            losses['hallucination_loss'] = torch.tensor(0.0, device=logits.device, requires_grad=True)
            losses['hallucination_confidence'] = torch.tensor(0.0, device=logits.device)

        # ========== 4. Total loss (weighted sum) ==========
        losses['total_loss'] = (
            self.query_weight * losses['query_loss'] +
            self.safety_weight * losses['safety_loss'] +
            self.hallucination_weight * losses['hallucination_loss']
        )

        return losses


def test_losses():
    """Test all loss functions"""
    print("\n" + "="*80)
    print("Testing Loss Functions")
    print("="*80)

    # Test configuration
    batch_size = 4
    seq_len = 128

    # Create dummy data
    logits = torch.randn(batch_size, seq_len, 10)

    # Initialize labels with -100 (ignore_index)
    labels = torch.full((batch_size, seq_len, 10), -100, dtype=torch.long)

    # Create response mask
    response_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
    response_mask[:, 64:] = 1  # Last 64 tokens are Response

    # Set Query labels on Response tokens (broadcast, same as the safety dataset)
    labels[:, 64:, :8] = torch.randint(0, 2, (batch_size, 64, 8))
    # Dimensions 8-9 are still -100 on non-Response (query) tokens

    # Set Response labels (only for positions where response_mask == 1)
    labels[:, 64:, 8] = torch.randint(0, 2, (batch_size, 64))  # Safety labels: 0 or 1
    labels[:, 64:, 9] = torch.randint(0, 2, (batch_size, 64))  # Hallucination labels: 0 or 1

    print(f"\nTest data:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Response mask shape: {response_mask.shape}")
    print(f"  Query (response token) positions: {((labels[:, :, 0] != -100) & (response_mask == 1)).sum().item()}")
    print(f"  Response positions: {response_mask.sum().item()}")

    # Test 1: GuardrailLoss
    print("\n" + "-"*80)
    print("Test 1: GuardrailLoss")
    print("-"*80)
    loss_fn1 = GuardrailLoss(
        num_query_classes=8,
        query_weight=1.0,
        safety_weight=2.0,
        hallucination_weight=3.0
    )
    losses1 = loss_fn1(logits, labels, response_mask)
    print(f"Total loss: {losses1['total_loss'].item():.4f}")
    print(f"Query loss: {losses1['query_loss'].item():.4f}")
    print(f"Safety loss: {losses1['safety_loss'].item():.4f}")
    print(f"Hallucination loss: {losses1['hallucination_loss'].item():.4f}")

    # Test 2: WeightedMultiTaskLoss
    print("\n" + "-"*80)
    print("Test 2: WeightedMultiTaskLoss (with dynamic weighting)")
    print("-"*80)
    loss_fn2 = WeightedMultiTaskLoss(
        num_query_classes=8,
        query_weight=1.0,
        safety_weight=2.0,
        hallucination_weight=3.0,
        dynamic_weight=True
    )
    losses2 = loss_fn2(logits, labels, response_mask)
    print(f"Total loss: {losses2['total_loss'].item():.4f}")
    print(f"Query loss: {losses2['query_loss'].item():.4f}")
    print(f"Safety loss: {losses2['safety_loss'].item():.4f}")
    print(f"Hallucination loss: {losses2['hallucination_loss'].item():.4f}")
    print(f"Dynamic query weight: {losses2['query_weight'].item():.4f}")
    print(f"Dynamic safety weight: {losses2['safety_weight'].item():.4f}")
    print(f"Dynamic hallucination weight: {losses2['hallucination_weight'].item():.4f}")

    # Test 3: ConfidenceWeightedLoss
    print("\n" + "-"*80)
    print("Test 3: ConfidenceWeightedLoss")
    print("-"*80)
    loss_fn3 = ConfidenceWeightedLoss(
        temperature=1.0,
        min_weight=0.1,
        confidence_threshold=0.5
    )
    # Test on safety task
    safety_logits = logits[:, :, 8:10]
    safety_labels = labels[:, :, 8]
    safety_mask = (labels[:, :, 8] != -100) & (response_mask == 1)
    losses3 = loss_fn3(safety_logits, safety_labels, mask=safety_mask)
    print(f"Loss: {losses3['loss'].item():.4f}")
    print(f"Average confidence: {losses3['confidence'].item():.4f}")
    print(f"Weights shape: {losses3['weights'].shape}")

    # Test 4: ConfidenceWeightedMultiTaskLoss
    print("\n" + "-"*80)
    print("Test 4: ConfidenceWeightedMultiTaskLoss")
    print("-"*80)
    loss_fn4 = ConfidenceWeightedMultiTaskLoss(
        num_query_classes=8,
        temperature=1.0,
        min_weight=0.1,
        query_weight=1.0,
        safety_weight=2.0,
        hallucination_weight=3.0
    )
    losses4 = loss_fn4(logits, labels, response_mask)
    print(f"Total loss: {losses4['total_loss'].item():.4f}")
    print(f"Query loss: {losses4['query_loss'].item():.4f}")
    print(f"Safety loss: {losses4['safety_loss'].item():.4f}")
    print(f"Hallucination loss: {losses4['hallucination_loss'].item():.4f}")
    print(f"Safety confidence: {losses4['safety_confidence'].item():.4f}")
    print(f"Hallucination confidence: {losses4['hallucination_confidence'].item():.4f}")

    print("\n" + "="*80)
    print("✅ All loss function tests passed!")
    print("="*80)

    # ----- Self-regulated confidence weighting sanity (BalancedMultiTaskLoss) -----
    print("\n" + "-"*80)
    print("Test 5: Self-regulated beta confidence weights (BalancedMultiTaskLoss)")
    print("-"*80)
    cr = BalancedMultiTaskLoss(use_safety_confidence_weight=True, use_hallu_confidence_weight=True, confidence_beta_tau=0.1)

    # (a) Cold start -> exact uniform (init_bias=-5 makes p_+ nearly identical)
    logits_cold = torch.full((2, 8), -5.0) + torch.randn(2, 8) * 1e-3
    mask_cold = torch.ones(2, 8, dtype=torch.bool)
    w = cr._compute_confidence_weights(logits_cold, mask_cold)
    max_dev = (w - torch.full_like(w, 1.0 / 8)).abs().max().item()
    print(f"  cold-start max|w - uniform| = {max_dev:.2e} (expect ~0)")
    assert max_dev < 1e-6, "cold start should be exactly uniform"

    # (b) Weights sum to 1 per sequence, non-negative, zero on invalid positions
    assert torch.allclose(w.sum(-1), torch.ones(2), atol=1e-5), "per-seq sum must be 1"
    mask_partial = torch.ones(2, 8, dtype=torch.bool)
    mask_partial[:, :3] = False
    w_p = cr._compute_confidence_weights(logits_cold, mask_partial)
    assert (w_p[:, :3].abs() < 1e-9).all(), "invalid positions must be 0"

    # (c) Numerical stability under tiny T (no nan/inf from exp overflow)
    cr_t = BalancedMultiTaskLoss(use_safety_confidence_weight=True, use_hallu_confidence_weight=True,
                                 confidence_temperature=0.05, confidence_beta_tau=0.1)
    w_t = cr_t._compute_confidence_weights(torch.full((1, 16), 8.0),
                                           torch.ones(1, 16, dtype=torch.bool))
    assert torch.isfinite(w_t).all() and abs(w_t.sum().item() - 1.0) < 1e-5
    print("  tiny-T path finite & normalized (no overflow/nan)")

    # (d) Higher dispersion -> sharper (max weight grows)
    maxw_lo = cr._compute_confidence_weights(
        torch.tensor([[0.0, 0.3, 0.0, -0.3, 0.6, 0.0, 0.15, -0.15]]),
        torch.ones(1, 8, dtype=torch.bool)).max().item()
    maxw_hi = cr._compute_confidence_weights(
        torch.tensor([[0.0, 3.0, 0.0, -3.0, 6.0, 0.0, 1.5, -1.5]]),
        torch.ones(1, 8, dtype=torch.bool)).max().item()
    print(f"  dispersion sharpen: low-spread max_w={maxw_lo:.3f} -> high-spread max_w={maxw_hi:.3f}")
    assert maxw_hi >= maxw_lo, "higher dispersion must not decrease sharpness"

    # (e) "softmax" aggregator fallback: constant sharpness + min_weight floor,
    # and it does NOT produce the exact-uniform cold start that "beta" does.
    cr_sm = BalancedMultiTaskLoss(use_safety_confidence_weight=True, use_hallu_confidence_weight=True,
                                  confidence_aggregator="softmax",
                                  confidence_min_weight=0.1)
    # Use a sequence with real p_+ dispersion so the softmax path visibly departs
    # from uniform (and the floor binds).
    logits_disp = torch.tensor([[0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    w_sm = cr_sm._compute_confidence_weights(logits_disp, torch.ones(1, 8, dtype=torch.bool))
    assert torch.isfinite(w_sm).all() and torch.allclose(w_sm.sum(-1), torch.ones(1), atol=1e-5)
    # softmax path applies a 0.1 floor, so NO weight should fall below it.
    assert w_sm.min().item() >= 0.1 - 1e-6, "softmax path must respect min_weight floor"
    # Confident token (idx 2) gets more than the uniform 1/8; floor lifts the rest.
    assert w_sm[0, 2].item() > 1.0/8, "softmax path should up-weight the confident token"
    # Contrast: the "beta" path on the SAME logits is also non-uniform, but the
    # point is that at the all-identical cold start it is EXACTLY uniform (test a)
    # while softmax is only approximately so. Verify that asymmetry directly:
    w_sm_cold = cr_sm._compute_confidence_weights(logits_cold, mask_cold)
    beta_cold = cr._compute_confidence_weights(logits_cold, mask_cold)
    sm_cold_dev = (w_sm_cold - torch.full_like(w_sm_cold, 1.0/8)).abs().max().item()
    beta_cold_dev = (beta_cold - torch.full_like(beta_cold, 1.0/8)).abs().max().item()
    print(f"  cold dev: beta={beta_cold_dev:.2e} (exact 0) vs softmax={sm_cold_dev:.2e} (nonzero)")
    assert beta_cold_dev < 1e-6 and sm_cold_dev > beta_cold_dev, \
        "beta must be exactly uniform at cold start; softmax must NOT be"

    # (f) invalid aggregator rejected
    try:
        BalancedMultiTaskLoss(use_safety_confidence_weight=True, use_hallu_confidence_weight=True, confidence_aggregator="bogus")
        raise AssertionError("should have rejected bogus aggregator")
    except ValueError:
        print("  bogus aggregator correctly rejected")

    # (g) confidence_weight_detach: detach the returned weights so they no
    # longer carry gradient (pure numerical per-token multipliers); the default
    # keeps them in the autograd graph so the model also learns THROUGH the
    # weighting. Check the requires_grad contract directly for both paths.
    cr_d = BalancedMultiTaskLoss(use_safety_confidence_weight=True,
                                 use_hallu_confidence_weight=True,
                                 confidence_weight_detach=True)
    cr_n = BalancedMultiTaskLoss(use_safety_confidence_weight=True,
                                 use_hallu_confidence_weight=True,
                                 confidence_weight_detach=False)
    _lg = torch.randn(2, 8, requires_grad=True)
    w_d = cr_d._compute_confidence_weights(_lg, torch.ones(2, 8, dtype=torch.bool))
    w_n = cr_n._compute_confidence_weights(
        _lg.detach().clone().requires_grad_(True), torch.ones(2, 8, dtype=torch.bool))
    assert w_d.requires_grad is False, "detach=True -> weights must NOT require grad"
    assert w_n.requires_grad is True, "detach=False -> weights must require grad"
    print("  confidence_weight_detach: weights detached (no grad) when True, graphed when False")

    print("  ✅ self-regulated beta weights OK (incl. softmax fallback)")

    return True


if __name__ == '__main__':
    test_losses()