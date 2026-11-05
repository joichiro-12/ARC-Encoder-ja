# Japanese ARC-Encoder on llm-jp-4

This branch trains an ARC-Encoder for Japanese. The encoder is the first 16 layers of `llm-jp-4-8b-base`, and the frozen decoder is `llm-jp-4-8b-instruct`, the LLM frozen in LLM-jp-Live. A second run with 24 layers checks the number of layers.

## What changes from upstream

| Part | Change |
| --- | --- |
| `embed_llm/models/utils/hf_tokenizer.py` | Reads a `tokenizer.json`, with the interface of the other tokenizers |
| `embed_llm/models/utils/loading.py`, `wrapped_models_training.py`, `augmented_model.py` | New backbone families `llmjp4` and `llama_hf`. The new `embedder_type` setting replaces the hard-coded Llama encoder |
| `embed_llm/data/` | The `language` setting chooses the words of the few-shot template (`Document:` or `文書：`) |
| `embed_llm/training/args.py`, `wrapped_models_training.py` | The new `bf16_compute` setting keeps the weights in float32 and computes in bfloat16 |
| `embed_llm/training/distributed.py` | Accepts the GPU UUIDs that ABCI gives |
| `embed_llm/models/transformer_layers.py` | A CPU path for attention, for the tests. xformers has no float32 operator on CPU |
| `embed_llm/generation/eval_ja.py` | Closed-book, open-book and compressed QA on Japanese data, with the retention |
| `scripts/llmjp4/` | Weight conversion, model setup, data preparation and ABCI job scripts |
| `configs/llmjp4/` | Pretraining, fine-tuning and baseline settings |

The default values keep the behavior of upstream.

## Steps on ABCI

```sh
source scripts/llmjp4/env.sh
uv sync --frozen --python 3.11
mkdir -p logs

# On a login node, for the downloads
.venv/bin/python scripts/llmjp4/setup_models.py --download
.venv/bin/python -c "from datasets import load_dataset as d; d('cl-nagoya/auto-wiki-qa'); d('hotchpotch/JQaRA'); d('SkelterLabsInc/JaQuAD', revision='refs/convert/parquet')"

# CPU jobs. The check waits for the data, and for the tokenizer that the first job copies
M=$(qsub -q rt_HC -N 0374_arc_models -l walltime=02:00:00 -v STEP=models scripts/llmjp4/run_cpu.pbs)
P=$(qsub -q rt_HC -N 0374_arc_pre -l walltime=06:00:00 -v STEP=pretrain scripts/llmjp4/run_cpu.pbs)
F=$(qsub -q rt_HC -N 0374_arc_ft -l walltime=04:00:00 -v STEP=finetune scripts/llmjp4/run_cpu.pbs)
E=$(qsub -q rt_HC -N 0374_arc_eval -l walltime=01:00:00 -v STEP=eval scripts/llmjp4/run_cpu.pbs)
qsub -q rt_HC -N 0374_arc_chk -l walltime=01:00:00 -W depend=afterany:$M:$P:$F:$E -v STEP=check scripts/llmjp4/run_cpu.pbs

# The short run: speed and loss on one node, then the evaluation on a few questions
T=$(qsub -q rt_HF -N 0374_arc_prelim -l walltime=03:00:00 -v CONFIG=configs/llmjp4/prelim_arc8_16layers.yaml scripts/llmjp4/train.pbs)
qsub -q rt_HG -N 0374_arc_ev_prelim -l walltime=01:00:00 -W depend=afterany:$T -v RUN=prelim_arc8_16layers,ENCODER=llm-jp-4-8b-base,DATASET=jsquad,N_SAMPLES=200 scripts/llmjp4/eval.pbs

# The baseline and its evaluation
T=$(qsub -q rt_HF -N 0374_arc_base -l walltime=08:00:00 -v CONFIG=configs/llmjp4/finetune_baseline_arc4_frozen.yaml scripts/llmjp4/train.pbs)
qsub -q rt_HG -N 0374_arc_ev_base -l walltime=06:00:00 -W depend=afterany:$T -v RUN=finetune_baseline_arc4_frozen,ENCODER=Llama3.2-3B-hf,DATASET=jsquad scripts/llmjp4/eval.pbs

# Pretraining (one node each)
qsub -q rt_HF -N 0374_arc_pre16 -v CONFIG=configs/llmjp4/pretrain_arc8_16layers.yaml scripts/llmjp4/train.pbs
```

## Speed on ABCI

Seconds per step on one `rt_HF` node (8 H200), over the first steps with the same seed:

| Run | float32 | float32, `checkpoint: false` | `bf16_compute` | `bf16_compute`, `checkpoint: false` |
| --- | --- | --- | --- | --- |
| Pretraining (16 layers, x8) | 7.47 | 5.87 | 1.24 | 0.96 |
| Baseline | 20.55 | not run | 2.63 | out of memory |

The settings here use `bf16_compute`, and `checkpoint: false` for the pretraining only. On the first 60 steps, the loss stays within 0.01 of float32. The attention alone does not explain the gap: one forward and backward call takes 1.5 ms in float32 (`cutlassF`) and 0.8 ms in bfloat16 (`flshattF`), with 3,584 tokens on one GPU.

## Checks

- `uv run --with pytest python -m pytest tests/test_llmjp4.py` runs on CPU, on tiny random models. Set `LLMJP4_TOKENIZER` to a `tokenizer.json` of llm-jp-4 to also check the real tokenizer.
- `scripts/llmjp4/check_conversion.py` compares the converted weights with the Hugging Face model on real text.
