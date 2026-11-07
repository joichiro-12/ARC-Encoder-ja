"""Compare the logits of an llm-jp-4 checkpoint before and after convert_hf_to_arc.py.

    python scripts/llmjp4/check_conversion.py --hf_dir <snapshot> --arc_dir <converted> --device cuda
"""

import argparse
import json
from pathlib import Path

import safetensors.torch
import torch
from transformers import AutoModelForCausalLM

from embed_llm.models.args import ModelArgs
from embed_llm.models.enhanced_transformer import Transformer
from embed_llm.models.utils.hf_tokenizer import Tokenizer

TEXTS = [
    "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。",
    "東京は日本の首都であり、人口は約1400万人である。\n次の問いに答えなさい。",
    "The quick brown fox jumps over the lazy dog. 2026年11月5日",
]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf_dir", type=Path, required=True)
    parser.add_argument("--arc_dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=1e-3)
    args = parser.parse_args()

    tokenizer = Tokenizer(args.arc_dir / "tokenizer.json")
    tokens = [tokenizer.encode(text, bos=True) for text in TEXTS]

    hf_model = AutoModelForCausalLM.from_pretrained(args.hf_dir, torch_dtype=torch.float32).eval().to(args.device)
    hf_logits = [hf_model(input_ids=torch.tensor([t], device=args.device), use_cache=False).logits[0].cpu() for t in tokens]
    del hf_model

    params = json.loads((args.arc_dir / "params.json").read_text())
    with torch.device("meta"):
        model = Transformer(ModelArgs(**params))
    state_dict = safetensors.torch.load_file(args.arc_dir / "consolidated.safetensors")
    model.load_state_dict({k: v.float() for k, v in state_dict.items()}, assign=True, strict=True)
    model = model.eval().to(args.device)

    worst = 0.0
    for t, reference in zip(tokens, hf_logits):
        logits = model(torch.tensor(t, device=args.device), seqlens=[len(t)]).cpu()
        diff = (logits - reference).abs().max().item()
        same_top1 = (logits.argmax(-1) == reference.argmax(-1)).float().mean().item()
        print(f"{len(t)} tokens: max |diff| {diff:.2e}, same top-1 {same_top1:.3f}")
        worst = max(worst, diff)
    print(f"worst max |diff| {worst:.2e} (threshold {args.threshold:.0e}): {'OK' if worst <= args.threshold else 'NG'}")
    raise SystemExit(0 if worst <= args.threshold else 1)


if __name__ == "__main__":
    main()
