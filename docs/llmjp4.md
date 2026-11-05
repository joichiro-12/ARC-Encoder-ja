# Japanese ARC-Encoder on llm-jp-4

This branch trains an ARC-Encoder for Japanese. The encoder is the first 16 layers of `llm-jp-4-8b-base`, and the frozen decoder is `llm-jp-4-8b-instruct`, the LLM frozen in LLM-jp-Live. A second run with 24 layers checks the number of layers.

## What changes from upstream

| Part | Change |
| --- | --- |
| `embed_llm/models/utils/hf_tokenizer.py` | Reads a `tokenizer.json`, with the interface of the other tokenizers |
| `embed_llm/models/utils/loading.py`, `wrapped_models_training.py`, `augmented_model.py` | New backbone families `llmjp4` and `llama_hf`. The new `embedder_type` setting replaces the hard-coded Llama encoder |
| `embed_llm/data/` | The `language` setting chooses the words of the few-shot template (`Document:` or `文書：`) |
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

# CPU jobs
qsub -q rt_HC -N 0374_arc_models -v STEP="scripts/llmjp4/setup_models.py" scripts/llmjp4/run_cpu.pbs
qsub -q rt_HC -N 0374_arc_pre -v STEP="scripts/llmjp4/prepare_data.py pretrain" scripts/llmjp4/run_cpu.pbs
qsub -q rt_HC -N 0374_arc_ft -v STEP="scripts/llmjp4/prepare_data.py finetune" scripts/llmjp4/run_cpu.pbs
qsub -q rt_HC -N 0374_arc_eval -v STEP="scripts/llmjp4/prepare_data.py eval" scripts/llmjp4/run_cpu.pbs

# Training (one node each)
qsub -q rt_HF -N 0374_arc_pre16 -v CONFIG=configs/llmjp4/pretrain_arc8_16layers.yaml scripts/llmjp4/train.pbs
```

## Checks

- `uv run --with pytest python -m pytest tests/test_llmjp4.py` runs on CPU, on tiny random models. Set `LLMJP4_TOKENIZER` to a `tokenizer.json` of llm-jp-4 to also check the real tokenizer.
- `scripts/llmjp4/check_conversion.py` compares the converted weights with the Hugging Face model on real text.
