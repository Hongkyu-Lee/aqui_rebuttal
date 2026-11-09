from __future__ import annotations

import argparse
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from transformers import (
    EarlyStoppingCallback,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from src.data.load_train_data import load_train_data
from src.utils import atomic_write_json
from src.train.train_embedding import (
    EmbeddingTrainer,
    InstructionSimilarityCallback,
    TripletCollator,
    build_dataset,
    build_model_and_tokenizer,
)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    configure_local_logging(cfg)
    config_paths = save_training_config_files(cfg, args)
    set_seed(int(cfg.train.optimization.seed))

    rows = load_train_data(cfg)
    split = cfg.train.data.get("split", None)
    if split and hasattr(rows, "keys") and split in rows:
        rows = rows[split]

    # Hold out a fixed validation split for best-model selection (lowest eval_loss).
    train_rows, val_rows = split_validation(rows, cfg)
    data_manifest_path = save_training_data_manifest(train_rows, cfg)
    config_paths.append(data_manifest_path)

    model, tokenizer = build_model_and_tokenizer(cfg)
    dataset = build_dataset(train_rows, cfg)
    val_dataset = build_dataset(val_rows, cfg) if val_rows else None
    collator = TripletCollator()
    trainer = EmbeddingTrainer(
        model=model,
        args=build_training_arguments(cfg, has_val=val_dataset is not None),
        train_dataset=dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        max_length=int(cfg.train.model.max_length),
        temperature=float(cfg.train.optimization.temperature),
        query_instruction=str(cfg.train.model.query_instruction),
        positive_alignment_weight=float(
            cfg.train.optimization.get("positive_alignment_weight", 0.0)
        ),
        negative_anchor_weight=float(cfg.train.optimization.get("negative_anchor_weight", 0.0)),
        positive_anchor_weight=float(cfg.train.optimization.get("positive_anchor_weight", 0.0)),
        raw_query_negative_anchor_weight=float(
            cfg.train.optimization.get("raw_query_negative_anchor_weight", 0.0)
        ),
    )
    if bool(cfg.train.evaluation.get("enabled", True)):
        callback = InstructionSimilarityCallback(
            dataset=dataset,
            collator=collator,
            batch_size=int(cfg.train.evaluation.batch_size),
            max_examples=cfg.train.evaluation.get("max_examples", None),
        )
        callback.trainer = trainer
        trainer.add_callback(callback)
    if bool(cfg.train.get("logging", {}).get("wandb", {}).get("enabled", False)):
        trainer.add_callback(WandbConfigFilesCallback(config_paths))
    # Early stopping: with a val split, stop once eval_loss has not improved for `patience`
    # epochs. This lets a single high `epochs` ceiling fit runs that saturate at different
    # times -- overfitters stop early, still-improving runs keep going -- instead of a fixed
    # epoch count that either cuts off (best==last, undertrained) or wastes epochs. Off by
    # default (patience<=0) so existing callers are unchanged; needs eval+load_best_model_at_end,
    # which build_training_arguments already enables when a val split is present.
    patience = int(cfg.train.optimization.get("early_stopping_patience", 0) or 0)
    if val_dataset is not None and patience > 0:
        threshold = float(cfg.train.optimization.get("early_stopping_threshold", 0.0) or 0.0)
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_threshold=threshold,
            )
        )
        print(
            f"[early-stopping] enabled: patience={patience} threshold={threshold} "
            f"(metric=eval_loss, lower is better)",
            flush=True,
        )
    trainer.train(resume_from_checkpoint=cfg.train.optimization.get("resume_from_checkpoint", None))
    # With load_best_model_at_end the in-memory model is the BEST (lowest eval_loss) -> saved here.
    trainer.save_model(str(cfg.train.model.output_dir))
    tokenizer.save_pretrained(str(cfg.train.model.output_dir))
    write_model_selection(trainer, cfg)


def write_model_selection(trainer: object, cfg: DictConfig) -> None:
    """Record where the BEST (lowest eval_loss) and LAST (final epoch) adapters live, so the
    downstream sweep can evaluate and compare both. best = output_dir (save_model wrote it);
    last = the highest-step epoch checkpoint dir (a valid PEFT adapter dir)."""
    import glob
    import re

    out = Path(str(cfg.train.model.output_dir))
    ckpts = glob.glob(str(out / "checkpoint-*"))

    def _step(path: str) -> int:
        m = re.search(r"checkpoint-(\d+)$", path)
        return int(m.group(1)) if m else -1

    last_ckpt = max(ckpts, key=_step) if ckpts else str(out)
    best_metric = getattr(getattr(trainer, "state", None), "best_metric", None)
    best_ckpt = getattr(getattr(trainer, "state", None), "best_model_checkpoint", None)
    # best==last when there was no val-based selection, or the best epoch IS the final one
    best_equals_last = best_metric is None or (
        best_ckpt is not None and Path(str(best_ckpt)).name == Path(str(last_ckpt)).name
    )
    payload = {
        "best_dir": str(out),                       # BEST adapter (saved by save_model)
        "best_checkpoint": str(best_ckpt) if best_ckpt else None,
        "best_eval_loss": best_metric,
        "last_checkpoint": str(last_ckpt),          # LAST-epoch adapter dir
        "best_equals_last": bool(best_equals_last),
        "epochs": float(cfg.train.optimization.epochs),
    }
    atomic_write_json(out / "model_selection.json", payload)
    print(f"[model-select] best_eval_loss={best_metric} best={Path(str(best_ckpt)).name if best_ckpt else '<final>'} "
          f"last={Path(str(last_ckpt)).name} best_equals_last={best_equals_last}", flush=True)


def split_validation(rows: object, cfg: DictConfig) -> tuple[list, list]:
    """Randomly hold out `val_split_ratio` of the training rows as a validation split,
    using a FIXED seed so the same rows are held out on every run of the same dataset
    (and across model sizes). Returns (train_rows, val_rows); val is [] when disabled.

    Splits by GROUP (group_id), not by expanded sample. One source query expands into many
    samples -- 8 gold facts -> 8 samples sharing one group_id -- so a per-sample split would
    put the SAME query on both sides and eval_loss would score a query the model trained on.
    Set train.data.val_split_by_group=false for the old per-sample behaviour.
    """
    import random

    row_list = list(rows) if not isinstance(rows, list) else rows
    ratio = float(cfg.train.data.get("val_split_ratio", 0.0) or 0.0)
    if ratio <= 0.0 or len(row_list) < 10:
        return row_list, []
    seed = int(cfg.train.data.get("val_split_seed", 42))
    by_group = bool(cfg.train.data.get("val_split_by_group", True))

    keys = [str(row.get("group_id", index)) for index, row in enumerate(row_list)]
    if not by_group:
        keys = [str(index) for index in range(len(row_list))]

    groups = sorted(set(keys))
    if len(groups) < 10:
        return row_list, []
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    n_val_groups = max(1, int(len(groups) * ratio))
    val_groups = set(shuffled[:n_val_groups])

    val_rows = [row for row, key in zip(row_list, keys) if key in val_groups]
    train_rows = [row for row, key in zip(row_list, keys) if key not in val_groups]
    print(f"[val-split] {len(train_rows)} train / {len(val_rows)} val samples "
          f"from {len(groups) - len(val_groups)}/{len(val_groups)} train/val groups "
          f"(ratio={ratio}, seed={seed}, by_group={by_group}) -> best model = lowest eval_loss",
          flush=True)
    return train_rows, val_rows


def save_training_data_manifest(rows: object, cfg: DictConfig) -> Path:
    output_dir = Path(str(cfg.train.model.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "train_data_manifest.json"

    row_list = list(rows) if not isinstance(rows, list) else rows
    source_row_indices = sorted(
        {
            int(row["source_row_index"])
            for row in row_list
            if isinstance(row, dict) and row.get("source_row_index") is not None
        }
    )
    question_groups = summarize_question_groups(row_list)
    source_question_keys = summarize_source_question_keys(question_groups)
    payload = {
        "data_config": OmegaConf.to_container(cfg.train.data, resolve=True),
        "num_training_samples": len(row_list),
        "num_source_rows": len(source_row_indices),
        "source_row_indices": source_row_indices,
        "num_source_questions": len(source_question_keys),
        "source_question_keys": source_question_keys,
        "num_question_groups": len(question_groups),
        "question_groups": question_groups,
        "sample_row_preview": row_list[: min(5, len(row_list))],
    }
    atomic_write_json(path, payload)
    return path


def summarize_question_groups(rows: list[object]) -> list[dict[str, object]]:
    groups: dict[str, dict[str, object]] = {}
    for training_row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        group_id = str(row.get("group_id", training_row_index))
        group = groups.setdefault(
            group_id,
            {
                "group_id": group_id,
                "source_row_index": row.get("source_row_index"),
                "qa_index": row.get("qa_index"),
                "source_qa_index": row.get("source_qa_index", row.get("qa_index")),
                "local_qa_index": row.get("local_qa_index"),
                "query_topic": row.get("query_topic"),
                "query": row.get("query"),
                "training_row_indices": [],
                "num_fact_pairs": 0,
            },
        )
        group["training_row_indices"].append(training_row_index)
        group["num_fact_pairs"] = int(group["num_fact_pairs"]) + 1
    return list(groups.values())


def summarize_source_question_keys(question_groups: list[dict[str, object]]) -> list[dict[str, int]]:
    keys = {
        (int(group["source_row_index"]), int(group["qa_index"]))
        for group in question_groups
        if group.get("source_row_index") is not None and group.get("qa_index") is not None
    }
    return [
        {"source_row_index": source_row_index, "qa_index": qa_index}
        for source_row_index, qa_index in sorted(keys)
    ]


def save_training_config_files(cfg: DictConfig, args: argparse.Namespace) -> list[Path]:
    output_dir = Path(str(cfg.train.model.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output_dir / "resolved_train_config.yaml"
    overrides_path = output_dir / "hydra_overrides.txt"

    resolved_config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    overrides_path.write_text(
        "\n".join(
            [
                f"config={Path(args.config).resolve()}",
                *[str(override) for override in args.overrides],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return [resolved_config_path, overrides_path]


class WandbConfigFilesCallback(TrainerCallback):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self._saved = False

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        if self._saved:
            return control
        self._saved = True
        try:
            import wandb
        except ImportError:
            return control
        if wandb.run is None:
            return control
        for path in self.paths:
            if path.exists():
                wandb.save(str(path), base_path=str(path.parent))
        return control


def build_training_arguments(cfg: DictConfig, has_val: bool = False) -> TrainingArguments:
    save_steps = cfg.train.optimization.get("save_every_steps", None)
    logging_cfg = cfg.train.get("logging", {})
    report_to = []
    if bool(logging_cfg.get("tensorboard", False)):
        report_to.append("tensorboard")
    if bool(logging_cfg.get("wandb", {}).get("enabled", False)):
        report_to.append("wandb")
    kwargs = dict(
        output_dir=str(cfg.train.model.output_dir),
        num_train_epochs=float(cfg.train.optimization.epochs),
        per_device_train_batch_size=int(cfg.train.optimization.batch_size),
        gradient_accumulation_steps=int(cfg.train.optimization.gradient_accumulation_steps),
        learning_rate=float(cfg.train.optimization.learning_rate),
        weight_decay=float(cfg.train.optimization.weight_decay),
        warmup_ratio=float(cfg.train.optimization.warmup_ratio),
        max_grad_norm=float(cfg.train.optimization.max_grad_norm),
        logging_strategy="steps",
        logging_steps=int(cfg.train.optimization.log_every),
        logging_dir=str(logging_cfg.get("dir", Path(cfg.train.model.output_dir) / "tensorboard")),
        save_steps=int(save_steps) if save_steps else 500,
        save_strategy="steps" if save_steps else "epoch",
        save_total_limit=int(cfg.train.optimization.save_total_limit)
        if cfg.train.optimization.get("save_total_limit", None)
        else None,
        dataloader_num_workers=int(cfg.train.optimization.num_workers),
        bf16=bool(cfg.train.optimization.get("bf16", False)),
        fp16=bool(cfg.train.optimization.get("fp16", False)),
        gradient_checkpointing=bool(cfg.train.optimization.get("gradient_checkpointing", False)),
        remove_unused_columns=False,
        report_to=report_to,
        run_name=str(logging_cfg.get("run_name", "embedding_model")),
    )
    if has_val:
        # Evaluate on the held-out val split each epoch; keep + reload the checkpoint with
        # the lowest eval_loss as the final saved model. save_strategy must match eval_strategy.
        kwargs.update(
            eval_strategy="epoch",
            save_strategy="epoch",
            per_device_eval_batch_size=int(
                cfg.train.optimization.get("eval_batch_size", cfg.train.optimization.batch_size)
            ),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    return TrainingArguments(**kwargs)


def configure_local_logging(cfg: DictConfig) -> None:
    logging_cfg = cfg.train.get("logging", {})
    wandb_cfg = logging_cfg.get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return

    wandb_dir = Path(str(wandb_cfg.get("dir", Path.cwd() / ".wandb"))).expanduser()
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_MODE"] = str(wandb_cfg.get("mode", "offline"))
    os.environ["WANDB_DIR"] = str(wandb_dir)
    os.environ["WANDB_PROJECT"] = str(wandb_cfg.get("project", "version_03_embedding"))
    os.environ["WANDB_CONSOLE"] = normalize_wandb_console(wandb_cfg.get("console", "off"))
    os.environ["WANDB_SILENT"] = str(wandb_cfg.get("silent", "true"))
    run_name = logging_cfg.get("run_name", None)
    if run_name:
        os.environ["WANDB_NAME"] = str(run_name)


def normalize_wandb_console(value: object) -> str:
    if isinstance(value, bool):
        return "off" if value is False else "auto"
    value_str = str(value).lower()
    if value_str in {"false", "0", "no"}:
        return "off"
    if value_str in {"true", "1", "yes"}:
        return "auto"
    return value_str


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    path = Path(path).resolve()
    with initialize_config_dir(config_dir=str(path.parent), version_base=None):
        return compose(config_name=path.name.removesuffix(path.suffix), overrides=overrides or [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an embedding model with Transformers Trainer.")
    parser.add_argument("--config", default="config/train_embedding.yaml")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


if __name__ == "__main__":
    main()
