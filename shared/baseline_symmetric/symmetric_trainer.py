"""
The symmetric-encoding baseline trainer -- the ONE difference from the main method.

The main EmbeddingTrainer (src/train/train_embedding.py) is asymmetric: in compute_loss the query
goes through `encode_texts` (adapter ON, gradients ON) while positives/negatives go through
`encode_reference_texts`, which disables the adapter and runs under no_grad -- so the fact
embeddings are the FROZEN base model's and never move.

This baseline overrides exactly that method so positives/negatives are encoded the SAME way as the
query: adapter ON, gradients ON. That is the naive InfoNCE setup where the LoRA moves the whole
space (query + facts). Nothing else changes -- the contrastive loss, the group mask, the
0.25 alignment term, the LoRA config, and every hyperparameter are inherited unchanged.

Because compute_loss calls `encode_reference_texts` for the main positive/negative embeddings,
overriding this one method flips the entire training regime from asymmetric to symmetric with no
duplication of the loss code (so the two pipelines cannot silently diverge).
"""

from __future__ import annotations

import torch

from src.train.train_embedding import EmbeddingTrainer


class SymmetricEmbeddingTrainer(EmbeddingTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The anchor terms regularize the adapter output toward the FROZEN base ("reference").
        # Here `encode_reference_texts` is no longer the frozen base, so those references become
        # meaningless -- refuse rather than silently train something un-interpretable. They are
        # 0.0 in the shipped config, so this never fires for the standard comparison.
        bad = {
            "positive_anchor_weight": self.positive_anchor_weight,
            "negative_anchor_weight": self.negative_anchor_weight,
            "raw_query_negative_anchor_weight": self.raw_query_negative_anchor_weight,
        }
        active = {name: weight for name, weight in bad.items() if weight and weight > 0}
        if active:
            raise ValueError(
                "SymmetricEmbeddingTrainer does not support the anchor-to-frozen-base terms "
                f"(they assume a frozen reference, which this baseline removes): {active}. "
                "Set them to 0 for the symmetric baseline."
            )

    def encode_reference_texts(self, model: torch.nn.Module, texts: list[str]) -> torch.Tensor:
        """SYMMETRIC baseline: encode positives/negatives WITH the adapter and WITH gradients,
        identical to the query path.

        The asymmetric parent instead does `with torch.no_grad(), model.disable_adapter():` here,
        freezing the fact space. Removing that is the entire baseline: now the LoRA also moves the
        positive and negative embeddings, so a gradient flows into the adapter from all three legs
        of the triplet -- the naive InfoNCE regime we are comparing against.

        (No eval/no_grad toggling: during compute_loss the model is in train mode and we WANT
        dropout + gradients on this path, exactly as for the query.)
        """
        return self.encode_texts(model, texts)
