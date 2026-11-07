"""Convert an llm-jp-4 checkpoint from the Hugging Face format to the format of this repo.

The output folder holds consolidated.safetensors, params.json and tokenizer.json:

    python scripts/llmjp4/convert_hf_to_arc.py --hf_dir <snapshot> --out_dir <MODEL_PATH>/llm-jp-4-8b-base
"""

import argparse
import json
import shutil
from pathlib import Path

import safetensors.torch
import torch

LAYER_KEYS = {
    "input_layernorm": "attention_norm",
    "post_attention_layernorm": "ffn_norm",
    "self_attn.q_proj": "attention.wq",
    "self_attn.k_proj": "attention.wk",
    "self_attn.v_proj": "attention.wv",
    "self_attn.o_proj": "attention.wo",
    "mlp.gate_proj": "feed_forward.w1",
    "mlp.down_proj": "feed_forward.w2",
    "mlp.up_proj": "feed_forward.w3",
}


def unpermute(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """HF keeps the real parts of RoPE before the imaginary parts in each head; this repo alternates them."""
    dim1, dim2 = w.shape
    return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def convert_key(key: str) -> str:
    if key == "lm_head.weight":
        return "output.weight"
    if key == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    if key == "model.norm.weight":
        return "norm.weight"
    parts = key.split(".")
    assert parts[:2] == ["model", "layers"] and parts[-1] == "weight", key
    return f"layers.{parts[2]}.{LAYER_KEYS['.'.join(parts[3:-1])]}.weight"


def convert_state_dict(hf_state_dict: dict[str, torch.Tensor], n_heads: int, n_kv_heads: int) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in hf_state_dict.items():
        if key.endswith("self_attn.q_proj.weight"):
            value = unpermute(value, n_heads)
        elif key.endswith("self_attn.k_proj.weight"):
            value = unpermute(value, n_kv_heads)
        out[convert_key(key)] = value
    return out


def params_from_config(config: dict) -> dict:
    return {
        "dim": config["hidden_size"],
        "n_layers": config["num_hidden_layers"],
        "head_dim": config["head_dim"],
        "hidden_dim": config["intermediate_size"],
        "n_heads": config["num_attention_heads"],
        "n_kv_heads": config["num_key_value_heads"],
        "norm_eps": config["rms_norm_eps"],
        "vocab_size": config["vocab_size"],
        "rope_theta": config["rope_theta"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads((args.hf_dir / "config.json").read_text())
    params = params_from_config(config)

    hf_state_dict = {}
    for path in sorted(args.hf_dir.glob("*.safetensors")):
        hf_state_dict.update(safetensors.torch.load_file(path))
    state_dict = convert_state_dict(hf_state_dict, params["n_heads"], params["n_kv_heads"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(state_dict, args.out_dir / "consolidated.safetensors")
    (args.out_dir / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    shutil.copy(args.hf_dir / "tokenizer.json", args.out_dir / "tokenizer.json")
    print(f"{len(state_dict)} tensors written to {args.out_dir}: {params}")


if __name__ == "__main__":
    main()
