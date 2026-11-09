# E2 -- notes for the agent running / fixing this experiment

## Intent (do not change)
The **only** difference between the two arms must be which side the LoRA adapts. Everything else
is identical and must stay identical if you fix bugs:
* base model, LoRA r/alpha/dropout/target_modules,
* data and negatives, loss (incl. `positive_alignment_weight=0.25`),
* validation split (`val_split_seed=42`), early stopping, training seed.

The symmetric arm is `shared/baseline_symmetric/train_symmetric.py`. It rebinds the trainer class
inside the unchanged entry point, so the two pipelines cannot drift apart. Both arms get the same
configs (decision D2: 2-3 configs), and the per-arm pick uses **synthetic validation only**
(`val_mean_hit`). Benchmark numbers are computed after selection and never feed back into it.

At test time the symmetric model must re-encode facts through its adapter
(`apply_adapter_to_memory=true`), with the shared fact cache **off**, because that cache holds
base-model vectors. `eval_retrieval sym ...` in `shared/lib.sh` already does this, with
`prebatch_facts=true` for speed.

## Known risks and what to do
* **OOM in the symmetric arm at batch 128 (most likely problem).** Do **not** change the batch size
  for one arm only: in-batch negatives are part of the objective. Options, in order:
  1. Make sure gradient checkpointing is on (it is in `config/train.yaml`).
  2. Lower `train.model.max_length` from 2048 to 512 for **both** arms. Synthetic texts are short;
     check the token-length distribution first.
  3. If symmetric bs=128 still OOMs, drop the bs=128 configs from `CONFIGS` for both arms and say so
     in the summary.
* **Loader mismatch** (step 1 warns): read E1/AGENT.md. Results are still a valid matched
  comparison, because both arms use the same loader, but they will not reproduce the paper's
  adapter exactly. Mention it in the summary.
* **A trained AQUI adapter for C1 differs from the paper adapter.** That is expected: GPU
  nondeterminism, plus possibly a different loader. Do not substitute the paper adapter, so that
  both arms come from the same pipeline. The paper adapter stays in E1.
* `train_lora` skips runs whose output dir already has `model_selection.json`. Use `FORCE_TRAIN=1`
  to retrain.
* The symmetric ImplexConv eval holds all ~0.86M fact embeddings in RAM (fp32 x 4096 dims, about
  14 GB). If the host runs out of RAM, set `evaluate.prebatch_facts=false` in `eval_retrieval` for
  sym. It is slower but per-row.

## What to report back
`E2_summary.md`, plus:
* the selected config per arm,
* any OOM workaround you applied (it must apply to both arms),
* whether the loader check matched.
