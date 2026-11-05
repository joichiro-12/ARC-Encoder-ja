from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer
from tokenizers.normalizers import Replace

# Backbone families read from a tokenizer.json, with their (bos, eos) tokens
HF_FAMILIES = {
    "llmjp4": ("<|startoftext|>", "<|return|>"),
    "llama_hf": ("<|begin_of_text|>", "<|end_of_text|>"),
}


class Tokenizer:
    """A tokenizer.json with the interface of the other tokenizers of this repo."""

    def __init__(self, tokenizer_path: str | Path, family: str = "llmjp4"):
        self.model = HFTokenizer.from_file(str(tokenizer_path))
        self.n_words = self.model.get_vocab_size(with_added_tokens=True)
        bos_token, eos_token = HF_FAMILIES[family]
        self.bos_id = self.model.token_to_id(bos_token)
        self.eos_id = self.model.token_to_id(eos_token)
        assert None not in (self.bos_id, self.eos_id), tokenizer_path
        # The normalizer of llm-jp-4 puts "▁" at the start of every text. Only the start of a sequence
        # has it in the normal use of the model, so the pieces of a prompt go without it.
        self.piece_model = self.model
        if family == "llmjp4":
            self.piece_model = HFTokenizer.from_file(str(tokenizer_path))
            self.piece_model.normalizer = Replace(" ", "▁")

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> list[int]:
        ids = (self.model if bos else self.piece_model).encode(s, add_special_tokens=False).ids
        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.model.decode(list(ids), skip_special_tokens=True)
