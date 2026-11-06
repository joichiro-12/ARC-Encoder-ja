"""Encode the references of the MoshiRAG data with a trained ARC-Encoder, for the additive LLM-jp-Live model.

Each reference goes through the encoder and the MLP, without <Cont>, as in the official MoshiRAG.

    python scripts/llmjp4/encode_moshirag_refs.py --run_name finetune_arc4_16layers_ftB --out_dir <DIR>

<DIR>/refs.npy holds the vectors [N, 4096] in float16. <DIR>/map.json gives for each conversation the frames
of its events ("rf") and their offsets in refs.npy ("off", one more than the events).
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

KAME = Path("/groups/gcg51557/experiments/0374_japanese_kame")
PARQUETS = sorted((KAME / "gemma4_dataset").glob("moshirag_jp_gemma4-*-of-050.parquet")) + [
    KAME / "gemma4_eval_dataset/moshirag_jp_gemma4_eval-001-of-001.parquet"
]
RET_DIR = KAME / "gemma4_post/ret"


def match_events(ret_frames, events: list[dict]) -> list[dict | None]:
    """Pairs the events of the parquet with the generated events, in order (sllm.mrag_prep.match_events)."""
    out, j = [], 0
    for frame in ret_frames:
        while j < len(events) and not (-3 <= int(frame) - int(events[j]["ret_frame"]) <= 9):
            j += 1
        if j >= len(events):
            out.append(None)
            continue
        out.append(events[j])
        j += 1
    return out


def collect_references(parquets, ret_dir: Path) -> tuple[list[dict], dict]:
    """One item per conversation: its stem, the frames of its events and their reference texts."""
    items, counts = [], {"conversations": 0, "events": 0, "no_vector": 0, "unmatched": 0}
    seen = set()
    for path in parquets:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["dialogue_id", "ret_frames", "ref_offsets"]):
            for row in batch.to_pylist():
                stem = row["dialogue_id"].split("/")[-1]
                if stem in seen:
                    continue
                seen.add(stem)
                frames = np.frombuffer(row["ret_frames"], np.int32)
                offsets = np.frombuffer(row["ref_offsets"], np.int32)
                ret_path = ret_dir / f"{stem}.json"
                events = json.loads(ret_path.read_text(encoding="utf-8"))["ret_events"] if ret_path.exists() else []
                texts = []
                for k, event in enumerate(match_events(frames, events)):
                    # An event without a vector stays without one, so the delays drawn in the evaluation stay the same
                    if offsets[k + 1] <= offsets[k]:
                        counts["no_vector"] += 1
                        texts.append("")
                    elif event is None:
                        counts["unmatched"] += 1
                        texts.append("")
                    else:
                        texts.append(str(event.get("reference", "")))
                items.append({"stem": stem, "rf": [int(f) for f in frames], "texts": texts})
                counts["conversations"] += 1
                counts["events"] += len(texts)
    return items, counts


@torch.no_grad()
def encode_texts(pipeline, texts: list[str], device: str, max_batch_tokens: int = 8192) -> list[np.ndarray]:
    """The vectors of each text after the MLP, in the order of the texts. An empty text gets no vector."""
    tokenizer = pipeline.embed_tokenizer.tokenizer
    tokens = [tokenizer.encode(text, bos=False, eos=False) if text else [] for text in texts]
    order = sorted((i for i, ids in enumerate(tokens) if ids), key=lambda i: len(tokens[i]))
    bridge = pipeline.model.bridge_module
    bridge = bridge[0] if isinstance(bridge, torch.nn.ModuleList) else bridge
    vectors: list[np.ndarray | None] = [None] * len(texts)
    start = 0
    while start < len(order):
        end, n_tokens = start, 0
        while end < len(order) and (end == start or n_tokens + len(tokens[order[end]]) <= max_batch_tokens):
            n_tokens += len(tokens[order[end]])
            end += 1
        batch = order[start:end]
        x = torch.tensor([t for i in batch for t in tokens[i]], device=device)
        comp, sizes = pipeline.model.embedder.forward_embedder(input_ids=x, seqlens=[len(tokens[i]) for i in batch], llm_number=0)
        out = bridge(comp).float().cpu().numpy().astype(np.float16)
        position = 0
        for i, size in zip(batch, sizes):
            vectors[i] = out[position : position + size]
            position += size
        start = end
    dim = next(v.shape[1] for v in vectors if v is not None) if order else 0
    return [v if v is not None else np.zeros((0, dim), np.float16) for v in vectors]


def main():
    from embed_llm.models.augmented_model import load_pipeline

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--ckpt", type=int, default=None)
    parser.add_argument("--llm_path", default=None)
    parser.add_argument("--embedder_path", default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Stop after this number of conversations, for a check")
    args = parser.parse_args()

    from embed_llm import MODEL_PATH

    items, counts = collect_references(PARQUETS, RET_DIR)
    if args.limit:
        items = items[: args.limit]
    print(json.dumps(counts), flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline, ckpt = load_pipeline(
        run_name=args.run_name,
        llm_path=args.llm_path or f"{MODEL_PATH}llm-jp-4-8b-instruct",
        embedder_path=args.embedder_path or f"{MODEL_PATH}llm-jp-4-8b-base",
        device=device,
        max_bs=8,
        ckpt=args.ckpt,
        llm_type="llmjp4",
    )
    print(f"checkpoint {ckpt}", flush=True)

    texts = [text for item in items for text in item["texts"]]
    vectors = encode_texts(pipeline, texts, device)
    lengths = np.array([len(v) for v in vectors if len(v)])
    print(f"references {len(lengths)}, vectors per reference: mean {lengths.mean():.2f}, median {np.median(lengths):.0f}, max {lengths.max()}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    refs = np.concatenate(vectors)
    np.save(args.out_dir / "refs.npy", refs)
    table, position, k = {}, 0, 0
    for item in items:
        offsets = [position]
        for _ in item["texts"]:
            position += len(vectors[k])
            offsets.append(position)
            k += 1
        table[item["stem"]] = {"rf": item["rf"], "off": offsets}
    (args.out_dir / "map.json").write_text(json.dumps(table, separators=(",", ":")) + "\n", encoding="utf-8")
    sample = refs[np.random.default_rng(0).choice(len(refs), min(100000, len(refs)), replace=False)].astype(np.float32)
    rms = np.sqrt((sample**2).mean(-1))
    print(f"vectors {refs.shape}, rms of a vector: mean {rms.mean():.5f}, p10 {np.quantile(rms, 0.1):.5f}, p90 {np.quantile(rms, 0.9):.5f}", flush=True)
    assert position == len(refs) and all(math.isfinite(float(v)) for v in rms[:1000])


if __name__ == "__main__":
    main()
