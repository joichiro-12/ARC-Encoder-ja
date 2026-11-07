"""Evaluate a Japanese ARC-Encoder on QA with documents, under three conditions:

- closed: the decoder gets no document;
- open: the decoder reads the documents as text;
- compressed: the documents go through the ARC-Encoder.

The retention is (compressed - closed) / (open - closed). Every condition uses the same few-shot examples.

    python -m embed_llm.generation.eval_ja --run_name finetune_arc4_16layers_ftA \
        --llm_path <MODEL_PATH>llm-jp-4-8b-instruct --embedder_path <MODEL_PATH>llm-jp-4-8b-base \
        --data <DATA_PATH>eval/jsquad.jsonl --out_file <RUN>/eval_jsquad.json
"""

import argparse
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

import torch

from embed_llm import TMP_PATH
from embed_llm.data.utils import PROMPT_LABELS, PromptLabels
from embed_llm.models.augmented_model import load_pipeline

CONDITIONS = ["closed", "open", "compressed"]
LENGTH_BINS = [(0, 64), (65, 128), (129, 256), (257, 10**9)]


def normalize(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split()).lower()


def exact_match(prediction: str, answers: list[str]) -> float:
    return float(any(normalize(prediction) == normalize(a) for a in answers))


def char_f1(prediction: str, answers: list[str]) -> float:
    best = 0.0
    pred = Counter(normalize(prediction))
    for answer in answers:
        gold = Counter(normalize(answer))
        same = sum((pred & gold).values())
        if same:
            p, r = same / sum(pred.values()), same / sum(gold.values())
            best = max(best, 2 * p * r / (p + r))
    return best


def build_prompt(condition: str, shots: list[dict], item: dict, labels: PromptLabels) -> tuple[list[str], list[str] | None]:
    """Returns the text pieces and the documents to compress between them."""
    qa = lambda ex, answer: labels.qa_template.format(question=ex["question"]) + answer
    if condition == "closed":
        # No document: drop the line break that follows it in the template
        parts = [qa(ex, ex["answers"][0]).lstrip("\n") + "\n\n" for ex in shots]
        return ["".join(parts) + qa(item, "").lstrip("\n").rstrip()], None
    if condition == "open":
        parts = [labels.doc_prefix + ex["passage"] + qa(ex, ex["answers"][0]) + "\n\n" for ex in shots]
        return ["".join(parts) + labels.doc_prefix + item["passage"] + qa(item, "").rstrip()], None
    pieces = [labels.doc_prefix]
    for ex in shots:
        pieces.append(qa(ex, ex["answers"][0]) + "\n\n" + labels.doc_prefix)
    pieces.append(qa(item, "").rstrip())
    return pieces, [ex["passage"] for ex in shots] + [item["passage"]]


def summarize(rows: list[dict]) -> dict:
    n = max(len(rows), 1)
    return {"n": len(rows), "em": sum(r["em"] for r in rows) / n, "char_f1": sum(r["char_f1"] for r in rows) / n}


def retention(scores: dict, metric: str) -> float | None:
    gap = scores["open"][metric] - scores["closed"][metric]
    return (scores["compressed"][metric] - scores["closed"][metric]) / gap if gap > 0 else None


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--llm_path", required=True)
    parser.add_argument("--embedder_path", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out_file", type=Path, required=True)
    parser.add_argument("--tmp_path", default=TMP_PATH)
    parser.add_argument("--n_shots", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--max_bs", type=int, default=8)
    parser.add_argument("--language", default="ja")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    labels = PROMPT_LABELS[args.language]
    data = [json.loads(line) for line in args.data.open(encoding="utf-8")]
    random.Random(args.seed).shuffle(data)
    shots, items = data[: args.n_shots], data[args.n_shots :]
    items = items[: args.n_samples] if args.n_samples else items

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline, ckpt = load_pipeline(
        run_name=args.run_name,
        tmp_path=args.tmp_path,
        llm_path=args.llm_path,
        embedder_path=args.embedder_path,
        device=device,
        max_bs=args.max_bs,
        ckpt=args.ckpt,
        llm_type="llmjp4",
    )
    tokenizer = pipeline.embed_tokenizer.tokenizer

    rows = []
    for condition in CONDITIONS:
        for start in range(0, len(items), args.max_bs):
            batch = items[start : start + args.max_bs]
            built = [build_prompt(condition, shots, item, labels) for item in batch]
            outputs = pipeline.generate(
                text_to_embed=[b[1] for b in built] if condition == "compressed" else None,
                device=device,
                batch_list_prompts=[b[0] for b in built],
                max_tokens=args.max_tokens,
                temperature=0,
                truncate_line=True,
            )
            for item, output in zip(batch, outputs):
                prediction = output.split("\n")[0].strip()
                rows.append(
                    {
                        "condition": condition,
                        "question": item["question"],
                        "answers": item["answers"],
                        "prediction": prediction,
                        "doc_tokens": len(tokenizer.encode(item["passage"])),
                        "em": exact_match(prediction, item["answers"]),
                        "char_f1": char_f1(prediction, item["answers"]),
                    }
                )

    scores = {c: summarize([r for r in rows if r["condition"] == c]) for c in CONDITIONS}
    result = {
        "run_name": args.run_name,
        "ckpt": ckpt,
        "data": str(args.data),
        "n_shots": args.n_shots,
        "scores": scores,
        "retention_em": retention(scores, "em"),
        "retention_char_f1": retention(scores, "char_f1"),
        "by_doc_length": {
            f"{low}-{high}": {
                c: summarize([r for r in rows if r["condition"] == c and low <= r["doc_tokens"] <= high]) for c in CONDITIONS
            }
            for low, high in LENGTH_BINS
        },
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    with args.out_file.with_suffix(".predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ("scores", "retention_em", "retention_char_f1")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
