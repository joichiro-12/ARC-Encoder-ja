"""Put the models of the Japanese ARC-Encoder under MODEL_PATH and TMP_PATH.

Downloading needs the network, so run this on a login node with --download first, then without it on a CPU node.

    python scripts/llmjp4/setup_models.py --download
    python scripts/llmjp4/setup_models.py
"""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import safetensors.torch
from huggingface_hub import snapshot_download

from embed_llm import MODEL_PATH, TMP_PATH

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("convert_hf_to_arc", HERE / "convert_hf_to_arc.py")
convert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert)

LLMJP4 = {"llm-jp-4-8b-base": "llm-jp/llm-jp-4-8b-base", "llm-jp-4-8b-instruct": "llm-jp/llm-jp-4-8b-instruct"}
LLAMA_3B_PARAMS = {
    "dim": 3072,
    "n_layers": 28,
    "head_dim": 128,
    "hidden_dim": 8192,
    "n_heads": 24,
    "n_kv_heads": 8,
    "norm_eps": 1e-5,
    "vocab_size": 128256,
    "rope_theta": 500000.0,
}


def convert_llmjp4(name: str, repo: str, offline: bool) -> None:
    out_dir = Path(MODEL_PATH) / name
    if (out_dir / "consolidated.safetensors").exists():
        print(f"{out_dir} exists")
        return
    snapshot = Path(snapshot_download(repo, local_files_only=offline, allow_patterns=["*.safetensors", "*.json"]))
    config = json.loads((snapshot / "config.json").read_text())
    params = convert.params_from_config(config)
    state_dict = {}
    for path in sorted(snapshot.glob("*.safetensors")):
        state_dict.update(safetensors.torch.load_file(path))
    state_dict = convert.convert_state_dict(state_dict, params["n_heads"], params["n_kv_heads"])
    out_dir.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(state_dict, out_dir / "consolidated.safetensors")
    (out_dir / "params.json").write_text(json.dumps(params, indent=2) + "\n")
    shutil.copy(snapshot / "tokenizer.json", out_dir / "tokenizer.json")
    print(f"{out_dir}: {len(state_dict)} tensors")


def setup_released_arc4(offline: bool) -> None:
    """The baseline: the released ARC4 weights, split like a checkpoint of this repo."""
    weights = Path(snapshot_download("kyutai/ARC4_Encoder_Llama", local_files_only=offline)) / "model.safetensors"
    tokenizer = Path(snapshot_download("unsloth/Llama-3.2-3B-Instruct", local_files_only=offline, allow_patterns=["tokenizer.json"]))
    backbone = Path(MODEL_PATH) / "Llama3.2-3B-hf"
    backbone.mkdir(parents=True, exist_ok=True)
    (backbone / "params.json").write_text(json.dumps(LLAMA_3B_PARAMS, indent=2) + "\n")
    shutil.copy(tokenizer / "tokenizer.json", backbone / "tokenizer.json")

    ckpt = Path(TMP_PATH) / "ARC4_Encoder_Llama/checkpoints/checkpoint_080000"
    state_dict = safetensors.torch.load_file(weights)
    for part in ("embedder", "bridge_module"):
        folder = ckpt / part
        folder.mkdir(parents=True, exist_ok=True)
        tensors = {k[len(part) + 1 :]: v for k, v in state_dict.items() if k.startswith(part + ".")}
        safetensors.torch.save_file(tensors, folder / "consolidated.safetensors")
        print(f"{folder}: {len(tensors)} tensors")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="Only fill the Hugging Face cache")
    args = parser.parse_args()
    if args.download:
        for repo in LLMJP4.values():
            snapshot_download(repo, allow_patterns=["*.safetensors", "*.json"])
        snapshot_download("kyutai/ARC4_Encoder_Llama")
        snapshot_download("unsloth/Llama-3.2-3B-Instruct", allow_patterns=["tokenizer.json"])
        return
    for name, repo in LLMJP4.items():
        convert_llmjp4(name, repo, offline=True)
    setup_released_arc4(offline=True)


if __name__ == "__main__":
    main()
