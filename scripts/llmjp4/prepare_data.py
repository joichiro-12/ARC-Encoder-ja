"""Build the Japanese data of the ARC-Encoder as plain JSONL files under --out_dir.

    python scripts/llmjp4/prepare_data.py pretrain --out_dir <DATA_PATH>
    python scripts/llmjp4/prepare_data.py finetune --out_dir <DATA_PATH>
    python scripts/llmjp4/prepare_data.py eval --out_dir <DATA_PATH>
    python scripts/llmjp4/prepare_data.py check --out_dir <DATA_PATH>

Pretraining lines hold {"text"}. Fine-tuning lines hold {"passage", "question", "answer"}.
Evaluation lines hold {"passage", "question", "answers"}.
"""

import argparse
import gzip
import json
import os
import random
import re
import sys
from multiprocessing import Pool
from pathlib import Path

EXPERIMENTS = Path("/groups/gcg51557/experiments")
CORPUS = EXPERIMENTS / "0118_dedup_corpusv4_ja/data/all/cleaned"
LLM_JP_EVAL = EXPERIMENTS / "0272_llmjpeval/codes/llm-jp-eval-v2.1.3/local_files/datasets/raw_files"
INSTRUCT4 = EXPERIMENTS / "0290_instruct5_dataset/datasets/instruct4"
FINEWEB = EXPERIMENTS / "0352_fineweb_translate"
KAME = EXPERIMENTS / "0374_japanese_kame"
MOSHIRAG = [KAME / "wiki_gemma4/s02", KAME / "wiki_gemma4_b/s02", KAME / "wiki_gemma4_v2/s02"]

# name: (folder under CORPUS, share of the tokens, share of the documents kept while reading)
PRETRAIN_SOURCES = {
    "ja_fineweb-2": ("ja_fineweb-2", 0.40, 0.1),
    "ja_cc": ("ja_cc/level0", 0.20, 0.1),
    "ja_wiki": ("ja_wiki", 0.20, 1.0),
    "ja_ceek_news": ("ja_ceek_news", 0.15, 0.1),
    "ja_nwjc": ("ja_nwjc", 0.05, 0.1),
}
CHARS_PER_TOKEN = 1.93  # Measured on Japanese references with the llm-jp-4 tokenizer
MIN_DOC_CHARS = 200
TO_ENGLISH = "この文書を英語に翻訳してください。"
SUMMARIZE = "この文書を要約してください。"
BODY_LINE = re.compile(r"moshi \(body\)(?:\([^)]*\))*:(.*)$")  # The body may carry "(subtopic: ...)"


def write_jsonl(path: Path, rows: list[dict], seed: int = 0) -> None:
    # Ranks read every world_size-th line, so similar lines should not stay together
    random.Random(seed).shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{path}: {len(rows)} lines", flush=True)


def sample_pretrain_source(job: tuple) -> dict:
    name, folder, target_tokens, keep, n_holdout, out_dir, seed = job
    rng = random.Random(f"{seed}-{name}")
    files = sorted((CORPUS / folder).rglob("*.jsonl.gz"))
    rng.shuffle(files)
    target_chars = target_tokens * CHARS_PER_TOKEN
    chars = docs = 0
    holdout = []
    out_path = out_dir / "pretrain" / f"{name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for file in files:
            with gzip.open(file, "rt", encoding="utf-8") as f:
                for line in f:
                    if rng.random() >= keep:
                        continue
                    text = json.loads(line)["text"].strip()
                    if len(text) < MIN_DOC_CHARS:
                        continue
                    row = json.dumps({"text": text}, ensure_ascii=False)
                    if len(holdout) < n_holdout:
                        holdout.append(row)
                        continue
                    out.write(row + "\n")
                    chars += len(text)
                    docs += 1
                    if chars >= target_chars:
                        break
            if chars >= target_chars:
                break
    holdout_path = out_dir / "eval" / f"holdout_{name}.jsonl"
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_path.write_text("\n".join(holdout) + "\n", encoding="utf-8")
    return {"name": name, "docs": docs, "chars": chars, "target_chars": int(target_chars), "reached": chars >= target_chars}


def prepare_pretrain(args) -> None:
    jobs = [
        (name, folder, args.total_tokens * share, keep, args.n_holdout, args.out_dir, args.seed)
        for name, (folder, share, keep) in PRETRAIN_SOURCES.items()
    ]
    with Pool(len(jobs)) as pool:
        for result in pool.imap_unordered(sample_pretrain_source, jobs):
            print(result, flush=True)


def moshirag_triples(s02: Path, stems: set[str] | None = None) -> list[dict]:
    """(utterance, reference, body) of each turn that used a reference."""
    rows = []
    for path in sorted(s02.glob("*.json")):
        if stems is not None and path.stem not in stems:
            continue
        record = json.loads(path.read_text(encoding="utf-8")).get("record") or []
        question = reference = None
        for line in record:
            if line.startswith("Human:"):
                question, reference = line[len("Human:") :].strip(), None
            elif line.startswith("Reference:"):
                reference = line[len("Reference:") :].strip()
            elif reference and question and (match := BODY_LINE.match(line)):
                body = match.group(1).strip()
                if body and not reference.startswith("不明"):
                    rows.append({"passage": reference, "question": question, "answer": body})
                reference = None
    return rows


def split_stems(name: str) -> set[str]:
    return set((KAME / "gemma4_split" / name).read_text().split())


def squad_rows(path: Path) -> list[dict]:
    rows = []
    for article in json.loads(path.read_text(encoding="utf-8"))["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"].split("[SEP]", 1)[-1].strip()
            for qa in paragraph["qas"]:
                answers = [a["text"] for a in qa.get("answers", []) if a["text"]]
                if answers:
                    rows.append({"passage": context, "question": qa["question"], "answers": answers})
    return rows


def jaquad_rows(split: str) -> list[dict]:
    from datasets import load_dataset

    rows = []
    # The main revision holds a loading script, which the datasets library no longer runs
    for ex in load_dataset("SkelterLabsInc/JaQuAD", revision="refs/convert/parquet", split=split):
        if ex["answers"]["text"]:
            rows.append({"passage": ex["context"], "question": ex["question"], "answers": list(ex["answers"]["text"])})
    return rows


def first_answer(rows: list[dict]) -> list[dict]:
    return [{"passage": r["passage"], "question": r["question"], "answer": r["answers"][0]} for r in rows]


def extraction_rows() -> list[dict]:
    rows = []
    for version in ("v0.1", "v0.2", "v0.3"):
        with (INSTRUCT4 / f"llmjp_extraction_wiki_ja_{version}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                messages = json.loads(line)["messages"]
                users = [m["content"] for m in messages if m["role"] == "user"]
                replies = [m["content"] for m in messages if m["role"] == "assistant"]
                if "\n\n" not in users[0]:
                    continue
                # The first message is the instruction, then the article; later turns ask about the same article
                instruction, article = users[0].split("\n\n", 1)
                rows.append({"passage": article.strip(), "question": instruction.strip(), "answer": replies[0].strip()})
                for user, reply in zip(users[1:], replies[1:]):
                    rows.append({"passage": article.strip(), "question": user.strip(), "answer": reply.strip()})
    return rows


def wiki_summary_rows(max_rows: int, seed: int) -> list[dict]:
    """The lead of a Wikipedia article is the summary of the rest of the article."""
    rng = random.Random(seed)
    rows = []
    for file in sorted((CORPUS / "ja_wiki").glob("*.jsonl.gz")):
        with gzip.open(file, "rt", encoding="utf-8") as f:
            for line in f:
                lead, _, body = json.loads(line)["text"].strip().partition("\n\n")
                if 30 <= len(lead) <= 400 and len(body) >= 400:
                    rows.append({"passage": body[:6000], "question": SUMMARIZE, "answer": lead.strip()})
    rng.shuffle(rows)
    return rows[:max_rows]


def translation_rows(n_fineweb: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    with (INSTRUCT4 / "jaster_v1.4.1.jsonl").open(encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            if re.match(r"(wikicorpus|alt)-j-to-e", sample["ID"]):
                user = next(m["content"] for m in sample["messages"] if m["role"] == "user")
                reply = next(m["content"] for m in sample["messages"] if m["role"] == "assistant")
                rows.append({"passage": user.split("### 入力:", 1)[-1].strip(), "question": TO_ENGLISH, "answer": reply.strip()})
    # FineWeb: `corpus/original` and `corpus_v2/translated` are aligned line by line.
    # corpus_v2 tells the translations cut by the length limit, through finish_reason.
    translated = sorted((FINEWEB / "corpus_v2/translated/en_fineweb").glob("score_30/*len004096.jsonl.gz"))
    rng.shuffle(translated)
    taken = 0
    for path in translated:
        with gzip.open(path, "rt", encoding="utf-8") as ja_file, gzip.open(
            FINEWEB / "corpus/original/en_fineweb" / path.parent.name / path.name, "rt", encoding="utf-8"
        ) as en_file:
            for ja_line, en_line in zip(ja_file, en_file):
                record = json.loads(ja_line)
                if record["metadata"]["finish_reason"] != "stop" or rng.random() >= 0.02:
                    continue
                ja, en = record["text"].strip(), json.loads(en_line)["text"].strip()
                if 100 <= len(ja) <= 1500:
                    rows.append({"passage": ja, "question": TO_ENGLISH, "answer": en})
                    taken += 1
                    if taken >= n_fineweb:
                        return rows
    return rows


def prepare_finetune(args) -> None:
    from datasets import load_dataset

    out = args.out_dir / "finetune"
    autowiki = load_dataset("cl-nagoya/auto-wiki-qa", split="train")
    write_jsonl(
        out / "autowikiqa.jsonl",
        [{"passage": ex["text"], "question": ex["query"], "answer": ex["answer"]} for ex in autowiki if ex["answer"]],
    )
    write_jsonl(out / "jsquad.jsonl", first_answer(squad_rows(LLM_JP_EVAL / "jsquad_train.json")))
    write_jsonl(out / "jaquad.jsonl", first_answer(jaquad_rows("train")))
    refs = moshirag_triples(MOSHIRAG[0], split_stems("train.txt"))
    for s02 in MOSHIRAG[1:]:
        refs += moshirag_triples(s02)
    write_jsonl(out / "moshirag_refs.jsonl", refs)
    write_jsonl(out / "extraction.jsonl", extraction_rows())
    write_jsonl(out / "wiki_summary.jsonl", wiki_summary_rows(args.max_summaries, args.seed))
    write_jsonl(out / "ja_to_en.jsonl", translation_rows(args.n_fineweb, args.seed))


def jqara_rows(seed: int) -> list[dict]:
    """One passage with the answer and four without it, joined, to imitate the top-5 of a retriever."""
    from datasets import load_dataset

    rng = random.Random(seed)
    questions: dict[str, dict] = {}
    for ex in load_dataset("hotchpotch/JQaRA", split="test"):
        q = questions.setdefault(ex["q_id"], {"question": ex["question"], "answers": ex["answers"], "pos": [], "neg": []})
        q["pos" if int(ex["label"]) == 1 else "neg"].append(ex["text"])
    rows = []
    for q in questions.values():
        if q["pos"] and len(q["neg"]) >= 4:
            passages = rng.sample(q["pos"], 1) + rng.sample(q["neg"], 4)
            rng.shuffle(passages)
            rows.append({"passage": "\n".join(passages), "question": q["question"], "answers": list(q["answers"])})
    return rows


def prepare_eval(args) -> None:
    out = args.out_dir / "eval"
    write_jsonl(out / "jsquad.jsonl", squad_rows(LLM_JP_EVAL / "jsquad_test.json"))
    write_jsonl(out / "jaquad.jsonl", jaquad_rows("validation"))
    write_jsonl(out / "jqara.jsonl", jqara_rows(args.seed))
    held_out = moshirag_triples(MOSHIRAG[0], split_stems("eval_new.txt"))
    write_jsonl(
        out / "moshirag_refs.jsonl",
        [{"passage": r["passage"], "question": r["question"], "answers": [r["answer"]]} for r in held_out],
    )


def check_file(path: Path, required: set[str], tokenizer=None, n_sample: int = 2000) -> dict:
    """Counts the lines, and the lines that miss a field or leave it empty."""
    main_field = "text" if "text" in required else "passage"
    lines = bad = chars = 0
    sample = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            lines += 1
            if required <= row.keys() and all(row[key] for key in required):
                chars += len(row[main_field])
                if len(sample) < n_sample:
                    sample.append(row[main_field])
            else:
                bad += 1
    result = {"file": str(path), "lines": lines, "bad": bad, "gb": round(path.stat().st_size / 1e9, 3), "chars": chars}
    if tokenizer is not None and sample:
        ratio = sum(len(tokenizer.encode(text)) for text in sample) / sum(len(text) for text in sample)
        result["tokens_per_char"] = round(ratio, 4)
        result["tokens"] = int(chars * ratio)
    return result


EXPECTED_FILES = {
    "pretrain": list(PRETRAIN_SOURCES),
    "finetune": ["autowikiqa", "jsquad", "jaquad", "moshirag_refs", "extraction", "wiki_summary", "ja_to_en"],
    "eval": ["jsquad", "jaquad", "jqara", "moshirag_refs"] + [f"holdout_{name}" for name in PRETRAIN_SOURCES],
}
REQUIRED_FIELDS = {
    "pretrain": {"text"},
    "finetune": {"passage", "question", "answer"},
    "eval": {"passage", "question", "answers"},
}


def prepare_check(args) -> None:
    from embed_llm.models.utils.hf_tokenizer import Tokenizer

    tokenizer = Tokenizer(Path(os.environ["ARC_MODEL_PATH"]) / "llm-jp-4-8b-base/tokenizer.json")
    failed = []
    for folder, names in EXPECTED_FILES.items():
        for name in names:
            path = args.out_dir / folder / f"{name}.jsonl"
            required = {"text"} if name.startswith("holdout_") else REQUIRED_FIELDS[folder]
            if not path.exists():
                failed.append(f"{path}: missing")
                continue
            result = check_file(path, required, tokenizer)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result["lines"] == 0 or result["bad"] > result["lines"] * 0.001:
                failed.append(f"{path}: {result['bad']} bad lines of {result['lines']}")
    print("\n".join(failed) if failed else "all files passed")
    sys.exit(1 if failed else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["pretrain", "finetune", "eval", "check"])
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_tokens", type=float, default=5e9)
    parser.add_argument("--n_holdout", type=int, default=2000)
    parser.add_argument("--max_summaries", type=int, default=200000)
    parser.add_argument("--n_fineweb", type=int, default=100000)
    args = parser.parse_args()
    {"pretrain": prepare_pretrain, "finetune": prepare_finetune, "eval": prepare_eval, "check": prepare_check}[args.step](args)


if __name__ == "__main__":
    main()
