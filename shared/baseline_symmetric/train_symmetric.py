"""
Entry point for the SYMMETRIC baseline. A drop-in replacement for
`python -m src.entry_points.train_embedding_model` -- same CLI, same config, same overrides.

    python -m baseline_symmetric.train_symmetric --config config/oppose_gen.yaml <hydra overrides...>

How it stays a fair baseline WITHOUT duplicating the pipeline:
the original entry point's main() constructs the trainer by the name `EmbeddingTrainer` bound in
its own module namespace. We rebind that name to SymmetricEmbeddingTrainer, then call the original
main() unchanged. So the data loading, val split, manifest, early stopping, best-model selection,
saving -- every step -- is the SAME code, executed identically; only the trainer class differs
(and it differs by a single method). There is no copied orchestration that could drift out of sync
with the main pipeline.
"""

from __future__ import annotations

import src.entry_points.train_embedding_model as base_entry

from baseline_symmetric.symmetric_trainer import SymmetricEmbeddingTrainer

# Rebind the trainer used by base_entry.main(). main() references `EmbeddingTrainer` as a module
# global, resolved at call time, so this swap takes effect for the run below and nowhere else --
# the source module on disk is untouched.
base_entry.EmbeddingTrainer = SymmetricEmbeddingTrainer


def main() -> None:
    print("[baseline] SYMMETRIC encoding: query + positive + negative ALL pass through the LoRA "
          "(naive InfoNCE). Facts are NOT frozen. Reusing the main pipeline for everything else.",
          flush=True)
    base_entry.main()


if __name__ == "__main__":
    main()
