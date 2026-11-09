from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainerCallback

from src.data.load_train_data import NORMALIZED_FIELDS, emits_normalized_rows
from src.train.utils import add_instructions


class TripletTextDataset(Dataset):
    def __init__(self, rows: Any, cfg: DictConfig) -> None:
        self.rows = rows
        # A NORMALIZING loader already emitted {query, positive, negative}; there
        # train.data.*_field named the SOURCE field it read (e.g. "gold_session_fact"), which is
        # not a key on the emitted row. Only a PASS-THROUGH loader's *_field names an emitted key.
        if emits_normalized_rows(cfg):
            self.query_field = NORMALIZED_FIELDS["query"]
            self.positive_field = NORMALIZED_FIELDS["positive"]
            self.negative_field = NORMALIZED_FIELDS["negative"]
        else:
            self.query_field = str(cfg.train.data.query_field)
            self.positive_field = str(cfg.train.data.positive_field)
            self.negative_field = str(cfg.train.data.negative_field)
        self.group_field = str(cfg.train.data.get("group_field", "group_id"))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        row = self.rows[index]
        group_id = row.get(self.group_field, index)
        return {
            "query": _require_text(row, self.query_field, index),
            "positive": _require_text(row, self.positive_field, index),
            "negative": _require_text(row, self.negative_field, index),
            "group_id": str(group_id),
        }


class TripletCollator:
    def __call__(self, rows: list[dict[str, str]]) -> dict[str, list[str]]:
        return {
            "query": [row["query"] for row in rows],
            "positive": [row["positive"] for row in rows],
            "negative": [row["negative"] for row in rows],
            "group_id": [row["group_id"] for row in rows],
        }


class EmbeddingTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        tokenizer: AutoTokenizer,
        max_length: int,
        temperature: float,
        query_instruction: str,
        positive_alignment_weight: float = 0.0,
        negative_anchor_weight: float = 0.0,
        positive_anchor_weight: float = 0.0,
        raw_query_negative_anchor_weight: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.temperature = temperature
        self.query_instruction = query_instruction
        self.positive_alignment_weight = positive_alignment_weight
        self.negative_anchor_weight = negative_anchor_weight
        self.positive_anchor_weight = positive_anchor_weight
        self.raw_query_negative_anchor_weight = raw_query_negative_anchor_weight

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, list[str]],
        return_outputs: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        query_embeddings = self.encode_texts(
            model,
            add_instructions(inputs["query"], self.query_instruction),
        )
        positive_embeddings = self.encode_reference_texts(model, inputs["positive"])
        negative_embeddings = self.encode_reference_texts(model, inputs["negative"])
        reference_negative_embeddings = None

        candidates = torch.cat([positive_embeddings, negative_embeddings], dim=0)
        scores = query_embeddings @ candidates.T
        logits = scores / self.temperature
        batch_size = query_embeddings.shape[0]
        positive_mask = build_positive_candidate_mask(
            inputs.get("group_id", []),
            batch_size=batch_size,
            device=logits.device,
        )
        contrastive_loss = multi_positive_contrastive_loss(logits, positive_mask)
        paired_positive_similarity = scores.diag().mean()
        positive_alignment_loss = 1 - paired_positive_similarity

        positive_anchor_loss = scores.new_zeros(())
        positive_anchor_similarity = None
        if self.positive_anchor_weight > 0:
            reference_positive_embeddings = self.encode_reference_texts(model, inputs["positive"])
            positive_anchor_similarity = (
                positive_embeddings * reference_positive_embeddings
            ).sum(dim=1).mean()
            positive_anchor_loss = 1 - positive_anchor_similarity

        negative_anchor_loss = scores.new_zeros(())
        negative_anchor_similarity = None
        if self.negative_anchor_weight > 0:
            reference_negative_embeddings = self.encode_reference_texts(model, inputs["negative"])
            negative_anchor_similarity = (
                negative_embeddings * reference_negative_embeddings
            ).sum(dim=1).mean()
            negative_anchor_loss = 1 - negative_anchor_similarity

        raw_query_negative_anchor_loss = scores.new_zeros(())
        raw_query_negative_current_similarity = None
        raw_query_negative_reference_similarity = None
        raw_query_negative_similarity_delta = None
        if self.raw_query_negative_anchor_weight > 0:
            raw_query_embeddings = self.encode_reference_texts(model, inputs["query"])
            reference_raw_query_embeddings = self.encode_reference_texts(model, inputs["query"])
            if reference_negative_embeddings is None:
                reference_negative_embeddings = self.encode_reference_texts(model, inputs["negative"])

            current_pair_similarity = (raw_query_embeddings * negative_embeddings).sum(dim=1)
            reference_pair_similarity = (
                reference_raw_query_embeddings * reference_negative_embeddings
            ).sum(dim=1)
            pair_similarity_delta = current_pair_similarity - reference_pair_similarity
            raw_query_negative_anchor_loss = pair_similarity_delta.pow(2).mean()
            raw_query_negative_current_similarity = current_pair_similarity.mean()
            raw_query_negative_reference_similarity = reference_pair_similarity.mean()
            raw_query_negative_similarity_delta = pair_similarity_delta.mean()

        loss = (
            contrastive_loss
            + self.positive_alignment_weight * positive_alignment_loss
            + self.positive_anchor_weight * positive_anchor_loss
            + self.negative_anchor_weight * negative_anchor_loss
            + self.raw_query_negative_anchor_weight * raw_query_negative_anchor_loss
        )

        self._log_score_diagnostics(
            scores,
            batch_size,
            positive_mask=positive_mask,
            contrastive_loss=contrastive_loss,
            positive_alignment_loss=positive_alignment_loss,
            positive_alignment_weight=self.positive_alignment_weight,
            positive_anchor_loss=positive_anchor_loss,
            positive_anchor_similarity=positive_anchor_similarity,
            negative_anchor_loss=negative_anchor_loss,
            negative_anchor_similarity=negative_anchor_similarity,
            raw_query_negative_anchor_loss=raw_query_negative_anchor_loss,
            raw_query_negative_current_similarity=raw_query_negative_current_similarity,
            raw_query_negative_reference_similarity=raw_query_negative_reference_similarity,
            raw_query_negative_similarity_delta=raw_query_negative_similarity_delta,
        )

        if return_outputs:
            return loss, {"logits": logits}
        return loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, list[str]],
        prediction_loss_only: bool = True,
        ignore_keys: Any = None,
        **_: Any,
    ) -> tuple[torch.Tensor, None, None]:
        """Validation step. Inputs are TEXT dicts (query/positive/negative), not input_ids,
        so the default prediction_step (which calls model(**inputs)) fails. Reuse our
        tokenizing contrastive loss and report it as eval_loss (no logits/labels)."""
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

    def _log_score_diagnostics(
        self,
        scores: torch.Tensor,
        batch_size: int,
        *,
        positive_mask: torch.Tensor,
        contrastive_loss: torch.Tensor,
        positive_alignment_loss: torch.Tensor,
        positive_alignment_weight: float,
        positive_anchor_loss: torch.Tensor,
        positive_anchor_similarity: torch.Tensor | None,
        negative_anchor_loss: torch.Tensor,
        negative_anchor_similarity: torch.Tensor | None,
        raw_query_negative_anchor_loss: torch.Tensor,
        raw_query_negative_current_similarity: torch.Tensor | None,
        raw_query_negative_reference_similarity: torch.Tensor | None,
        raw_query_negative_similarity_delta: torch.Tensor | None,
    ) -> None:
        if not self.model.training:
            return
        if self.state.global_step % max(1, self.args.logging_steps) != 0:
            return

        with torch.no_grad():
            positive_scores = scores.diag()
            explicit_negative_scores = scores[:, batch_size:].diag()
            best_positive_scores = scores.masked_fill(~positive_mask, -1e9).max(dim=1).values
            negative_mask = ~positive_mask
            hardest_negative_scores = scores.masked_fill(~negative_mask, -1e9).max(dim=1).values
            metrics = {
                "train_instruction_query_to_positive_sim": (
                    positive_scores.mean().detach().float().cpu().item()
                ),
                "train_instruction_query_to_best_positive_sim": (
                    best_positive_scores.mean().detach().float().cpu().item()
                ),
                "train_instruction_query_to_paired_negative_sim": (
                    explicit_negative_scores.mean().detach().float().cpu().item()
                ),
                "train_instruction_query_to_hardest_batch_negative_sim": (
                    hardest_negative_scores.mean().detach().float().cpu().item()
                ),
                "train_positive_over_paired_negative_margin": (
                    positive_scores - explicit_negative_scores
                ).mean().detach().float().cpu().item(),
                "train_positive_over_hardest_negative_margin": (
                    positive_scores - hardest_negative_scores
                ).mean().detach().float().cpu().item(),
                "contrastive_loss": contrastive_loss.detach().float().cpu().item(),
                "positive_alignment_loss": (
                    positive_alignment_loss.detach().float().cpu().item()
                ),
                "positive_alignment_weight": float(positive_alignment_weight),
                "positive_anchor_loss": positive_anchor_loss.detach().float().cpu().item(),
                "negative_anchor_loss": negative_anchor_loss.detach().float().cpu().item(),
                "raw_query_negative_anchor_loss": (
                    raw_query_negative_anchor_loss.detach().float().cpu().item()
                ),
            }
            if positive_anchor_similarity is not None:
                metrics["positive_anchor_similarity"] = (
                    positive_anchor_similarity.detach().float().cpu().item()
                )
            if negative_anchor_similarity is not None:
                metrics["negative_anchor_similarity"] = (
                    negative_anchor_similarity.detach().float().cpu().item()
                )
            if raw_query_negative_current_similarity is not None:
                metrics["raw_query_to_negative_current_sim"] = (
                    raw_query_negative_current_similarity.detach().float().cpu().item()
                )
            if raw_query_negative_reference_similarity is not None:
                metrics["raw_query_to_negative_base_sim"] = (
                    raw_query_negative_reference_similarity.detach().float().cpu().item()
                )
            if raw_query_negative_similarity_delta is not None:
                metrics["raw_query_to_negative_sim_delta"] = (
                    raw_query_negative_similarity_delta.detach().float().cpu().item()
                )
            self.log(metrics)

    def encode_texts(self, model: torch.nn.Module, texts: list[str]) -> torch.Tensor:
        device = next(model.parameters()).device
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        outputs = model(**encoded)
        embeddings = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def encode_reference_texts(self, model: torch.nn.Module, texts: list[str]) -> torch.Tensor:
        disable_adapter = getattr(model, "disable_adapter", None)
        adapter_context = disable_adapter() if callable(disable_adapter) else nullcontext()
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad(), adapter_context:
                return self.encode_texts(model, texts).detach()
        finally:
            if was_training:
                model.train()


class InstructionSimilarityCallback(TrainerCallback):
    def __init__(
        self,
        *,
        dataset: Dataset,
        collator: TripletCollator,
        batch_size: int,
        max_examples: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.collator = collator
        self.batch_size = batch_size
        self.max_examples = max_examples
        self.trainer: EmbeddingTrainer | None = None

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.trainer is None:
            return control

        model = kwargs.get("model", self.trainer.model)
        was_training = model.training
        model.eval()
        metrics = self.compute_metrics(model)
        metrics["epoch"] = state.epoch
        self.trainer.log(metrics)
        if was_training:
            model.train()
        return control

    def compute_metrics(self, model: torch.nn.Module) -> dict[str, float]:
        total = 0
        sums = {
            "eval_raw_query_positive_similarity": 0.0,
            "eval_raw_query_negative_similarity": 0.0,
            "eval_instruction_query_positive_similarity": 0.0,
            "eval_instruction_query_negative_similarity": 0.0,
        }
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )

        with torch.no_grad():
            for batch in dataloader:
                if self.max_examples is not None and total >= self.max_examples:
                    break
                if self.max_examples is not None:
                    remaining = self.max_examples - total
                    batch = {key: values[:remaining] for key, values in batch.items()}

                count = len(batch["query"])
                raw_query_embeddings = self.trainer.encode_reference_texts(
                    model, batch["query"]
                )
                instruction_query_embeddings = self.trainer.encode_texts(
                    model,
                    add_instructions(batch["query"], self.trainer.query_instruction),
                )
                positive_embeddings = self.trainer.encode_reference_texts(
                    model, batch["positive"]
                )
                negative_embeddings = self.trainer.encode_reference_texts(
                    model, batch["negative"]
                )

                sums["eval_raw_query_positive_similarity"] += (
                    raw_query_embeddings * positive_embeddings
                ).sum(dim=1).sum().detach().float().cpu().item()
                sums["eval_raw_query_negative_similarity"] += (
                    raw_query_embeddings * negative_embeddings
                ).sum(dim=1).sum().detach().float().cpu().item()
                sums["eval_instruction_query_positive_similarity"] += (
                    instruction_query_embeddings * positive_embeddings
                ).sum(dim=1).sum().detach().float().cpu().item()
                sums["eval_instruction_query_negative_similarity"] += (
                    instruction_query_embeddings * negative_embeddings
                ).sum(dim=1).sum().detach().float().cpu().item()
                total += count

        metrics = {key: value / max(1, total) for key, value in sums.items()}
        metrics["eval_instruction_positive_minus_negative"] = (
            metrics["eval_instruction_query_positive_similarity"]
            - metrics["eval_instruction_query_negative_similarity"]
        )
        metrics["eval_raw_negative_minus_positive"] = (
            metrics["eval_raw_query_negative_similarity"]
            - metrics["eval_raw_query_positive_similarity"]
        )
        return metrics


def build_model_and_tokenizer(cfg: DictConfig) -> tuple[torch.nn.Module, AutoTokenizer]:
    backend = str(cfg.train.model.get("backend", "qwen3"))
    if backend == "gritlm":
        raise NotImplementedError(
            "GritLM model loading needs the gritlm package/API wired explicitly. "
            "Use train.model.backend=qwen3 for AutoModel-based training."
        )
    if backend != "qwen3":
        raise ValueError(f"Unsupported train.model.backend: {backend}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(cfg.train.model.name),
        padding_side="left",
        trust_remote_code=bool(cfg.train.model.trust_remote_code),
    )
    model = AutoModel.from_pretrained(
        str(cfg.train.model.name),
        trust_remote_code=bool(cfg.train.model.trust_remote_code),
    )
    if bool(cfg.train.model.get("lora", {}).get("enabled", False)):
        model = apply_lora(model, cfg)
    return model, tokenizer


def apply_lora(model: torch.nn.Module, cfg: DictConfig) -> torch.nn.Module:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError("Install peft to train with train.model.lora.enabled=true.") from exc

    lora_cfg = cfg.train.model.lora
    target_modules = lora_cfg.target_modules
    if isinstance(target_modules, str) and "," in target_modules:
        target_modules = [item.strip() for item in target_modules.split(",") if item.strip()]

    peft_config = LoraConfig(
        r=int(lora_cfg.r),
        lora_alpha=int(lora_cfg.alpha),
        lora_dropout=float(lora_cfg.dropout),
        target_modules=target_modules,
        bias=str(lora_cfg.bias),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def build_dataset(rows: Any, cfg: DictConfig) -> TripletTextDataset:
    return TripletTextDataset(rows, cfg)




def build_positive_candidate_mask(
    group_ids: list[str],
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if not group_ids:
        group_ids = [str(index) for index in range(batch_size)]
    if len(group_ids) != batch_size:
        raise ValueError(
            f"Expected {batch_size} group IDs for batch, received {len(group_ids)}."
        )

    group_to_int = {group_id: index for index, group_id in enumerate(dict.fromkeys(group_ids))}
    query_groups = torch.tensor(
        [group_to_int[group_id] for group_id in group_ids],
        dtype=torch.long,
        device=device,
    )
    candidate_groups = torch.cat([query_groups, query_groups], dim=0)
    candidate_is_positive = torch.cat(
        [
            torch.ones(batch_size, dtype=torch.bool, device=device),
            torch.zeros(batch_size, dtype=torch.bool, device=device),
        ],
        dim=0,
    )
    return (
        candidate_is_positive.unsqueeze(0)
        & (query_groups.unsqueeze(1) == candidate_groups.unsqueeze(0))
    )


def multi_positive_contrastive_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    if not torch.all(positive_mask.any(dim=1)):
        raise ValueError("Each query must have at least one positive candidate.")
    masked_positive_logits = logits.masked_fill(~positive_mask, torch.finfo(logits.dtype).min)
    numerator = torch.logsumexp(masked_positive_logits, dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    return -(numerator - denominator).mean()

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def _require_text(row: Any, field: str, index: int) -> str:
    if field not in row:
        raise KeyError(f"Missing field {field!r} in training row {index}.")
    value = row[field]
    if value is None:
        raise ValueError(f"Field {field!r} is None in training row {index}.")
    return str(value)
