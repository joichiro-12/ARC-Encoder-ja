"""CPU tests of the llm-jp-4 support, on tiny random models. Run with: python -m pytest tests/test_llmjp4.py"""

import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from embed_llm.data.dataset import sequence_iterator
from embed_llm.data.tokenize import Tokenizer as TokenizerWrapper
from embed_llm.data.tokenize import get_sample
from embed_llm.data.utils import PROMPT_LABELS
from embed_llm.generation.eval_ja import build_prompt, char_f1, exact_match, retention
from embed_llm.models.args import BridgeArgs, EmbedderArgs, ModelArgs, PoolingArgs
from embed_llm.models.args import PipelineArgs
from embed_llm.models.augmented_model import EmbedAugModel, EmbedAugPipeline
from embed_llm.models.enhanced_transformer import Transformer
from embed_llm.models.utils.hf_tokenizer import Tokenizer
from embed_llm.models.wrapped_models_training import fsdp_precision
from embed_llm.training.distributed import visible_devices

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("convert_hf_to_arc", ROOT / "scripts/llmjp4/convert_hf_to_arc.py")
convert_hf_to_arc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert_hf_to_arc)
spec = importlib.util.spec_from_file_location("encode_moshirag_refs", ROOT / "scripts/llmjp4/encode_moshirag_refs.py")
encode_moshirag_refs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(encode_moshirag_refs)
spec = importlib.util.spec_from_file_location("prepare_data", ROOT / "scripts/llmjp4/prepare_data.py")
prepare_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_data)

TINY = {
    "hidden_size": 64,
    "num_hidden_layers": 3,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "intermediate_size": 128,
    "vocab_size": 200,
    "rms_norm_eps": 1e-6,
    "rope_theta": 500000.0,
}


def tiny_args() -> ModelArgs:
    return ModelArgs(**convert_hf_to_arc.params_from_config(TINY))


def tiny_embedder(n_truncated: int = 1, rate: int = -4) -> EmbedderArgs:
    return EmbedderArgs(
        n_truncated_layers=n_truncated,
        pooling_module=PoolingArgs(pool_type="mean_pooled_queries", where="before"),
        compress_rates=[rate],
        trained_layers=TINY["num_hidden_layers"] + 1,
        causal_embedder=False,
        rec_tok=True,
        cont_tok=True,
    )


def test_conversion_matches_hf_logits():
    torch.manual_seed(0)
    hf_model = LlamaForCausalLM(
        LlamaConfig(
            hidden_size=TINY["hidden_size"],
            num_hidden_layers=TINY["num_hidden_layers"],
            num_attention_heads=TINY["num_attention_heads"],
            num_key_value_heads=TINY["num_key_value_heads"],
            head_dim=TINY["head_dim"],
            intermediate_size=TINY["intermediate_size"],
            vocab_size=TINY["vocab_size"],
            rms_norm_eps=TINY["rms_norm_eps"],
            rope_theta=TINY["rope_theta"],
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=False,
            attn_implementation="eager",
        )
    ).eval()
    state_dict = convert_hf_to_arc.convert_state_dict(
        hf_model.state_dict(), TINY["num_attention_heads"], TINY["num_key_value_heads"]
    )
    model = Transformer(tiny_args()).eval()
    model.load_state_dict(state_dict, strict=True)

    tokens = torch.randint(0, TINY["vocab_size"], (24,))
    with torch.no_grad():
        reference = hf_model(input_ids=tokens[None], use_cache=False).logits[0]
        logits = model(tokens, seqlens=[len(tokens)])
    assert (logits - reference).abs().max() < 1e-4


@pytest.mark.parametrize("n", [3, 4, 8, 9, 10, 52, 256])
def test_number_of_compressed_vectors(n):
    torch.manual_seed(0)
    encoder = Transformer(tiny_args(), embedder_args=tiny_embedder()).eval()
    with torch.no_grad():
        h, seqlens = encoder.forward_embedder(torch.randint(0, TINY["vocab_size"], (n,)), seqlens=[n])
    assert seqlens == [math.ceil(n / 4)]
    assert h.shape == (math.ceil(n / 4), TINY["hidden_size"])
    assert encoder.n_layers == TINY["num_hidden_layers"] - 1


def test_loss_updates_the_encoder_and_not_the_decoder():
    torch.manual_seed(0)
    decoder = Transformer(tiny_args())
    for p in decoder.parameters():
        p.requires_grad = False
    model = EmbedAugModel(
        bridge_args=BridgeArgs(bridge_type="mlp", in_dim=64, hidden_dim=32, out_dim=64),
        llms=[decoder],
        embedder=Transformer(tiny_args(), embedder_args=tiny_embedder()),
    )
    n, m = 16, 10
    output = model(
        x=torch.randint(0, 200, (m,)),
        seqlens=[m],
        embeddings=torch.randint(0, 200, (n,)),
        embed_seqlens=[[n]],
        insert_comp_repr=[[0]],
        batch_type="continuation",
    )
    assert output.shape == (m, 200)
    output.sum().backward()
    assert model.bridge_module.layer1.weight.grad.abs().sum() > 0
    assert model.embedder.tok_embeddings.weight.grad.abs().sum() > 0
    assert model.embedder.cont_tok[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in decoder.parameters())


class CharTokenizer:
    """One token per character. Enough to read back what the iterators build."""

    bos_id = 1
    eos_id = 2

    def encode(self, s, bos=False, eos=False):
        return ([1] if bos else []) + [ord(c) for c in s] + ([2] if eos else [])

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i > 2)


@pytest.mark.parametrize("language", ["en", "ja"])
def test_few_shot_template_uses_the_labels(language):
    labels = PROMPT_LABELS[language]
    tok = TokenizerWrapper(tokenizer=CharTokenizer(), model_name="llmjp4")
    samples = [
        get_sample(
            {"question": f"問い{i}", "answer": f"答{i}", "passage": f"文書の中身{i}"},
            tok,
            tok,
            instruct=True,
            labels=labels,
        )
        for i in range(4)
    ]
    sequences = list(
        sequence_iterator(
            iter(samples),
            seq_len=256,
            llm_tokenizer=tok,
            embed_tokenizer=tok,
            is_finite=False,
            adapt_seq_len=True,
            few_shot=2,
            interleave=True,
            labels=labels,
        )
    )
    last = sequences[-1]
    text = CharTokenizer().decode(last.x)
    assert text.startswith(labels.doc_prefix)
    assert labels.question_prefix + "問い1" in text
    assert labels.question_prefix + "問い3" in text
    assert (labels.answer + labels.sep) in text
    assert "中身" not in text  # documents are compressed, so they are not in the text stream
    compressed = last.to_embed[0]["text"]
    assert compressed[-1] == "文書の中身3" and len(compressed) == 3
    # The loss only covers the last answer
    target = "".join(chr(t) for t, m in zip(last.y, last.mask) if m and t > 2)
    assert target == "答3"


def test_real_tokenizer_round_trip():
    path = os.environ.get("LLMJP4_TOKENIZER")
    if not path:
        pytest.skip("set LLMJP4_TOKENIZER to the tokenizer.json of llm-jp-4")
    tokenizer = Tokenizer(path)
    assert tokenizer.n_words == 196608 and (tokenizer.bos_id, tokenizer.eos_id) == (1, 2)
    for text in ["吾輩は猫である。\n名前はまだ無い。", "東京 2026年 ABC 1.5%", "改行\n\nの後"]:
        ids = tokenizer.encode(text, bos=True, eos=True)
        assert ids[0] == 1 and ids[-1] == 2
        assert tokenizer.decode(ids) == text
    # A prompt cut into pieces gives the same tokens as the whole prompt, with no space at the cuts
    whole = tokenizer.encode("文書：\n質問：天気は？\n答え：晴れ", bos=True)
    pieces = tokenizer.encode("文書：", bos=True) + tokenizer.encode("\n質問：天気は？\n答え：晴れ", bos=False)
    assert pieces == whole
    assert tokenizer.encode("東京 大阪", bos=False) != tokenizer.encode("東京大阪", bos=False)  # A real space stays


def test_eval_prompts_put_each_document_between_the_pieces():
    labels = PROMPT_LABELS["ja"]
    shots = [{"passage": f"文書{i}", "question": f"問{i}", "answers": [f"答{i}"]} for i in range(2)]
    item = {"passage": "最後の文書", "question": "最後の問", "answers": ["x"]}
    pieces, docs = build_prompt("compressed", shots, item, labels)
    assert docs == ["文書0", "文書1", "最後の文書"] and len(pieces) == len(docs) + 1
    joined = "".join(p + d for p, d in zip(pieces, docs + [""]))
    assert joined == "文書：文書0\n質問：問0\n答え：答0\n\n文書：文書1\n質問：問1\n答え：答1\n\n文書：最後の文書\n質問：最後の問\n答え："
    assert build_prompt("open", shots, item, labels)[0][0] == joined
    closed, no_docs = build_prompt("closed", shots, item, labels)
    assert no_docs is None and closed[0] == "質問：問0\n答え：答0\n\n質問：問1\n答え：答1\n\n質問：最後の問\n答え："


def test_eval_metrics():
    assert exact_match("東京　都", ["東京都"]) == 1.0
    assert exact_match("ＡＢＣ", ["abc"]) == 1.0
    assert exact_match("大阪", ["東京"]) == 0.0
    assert char_f1("東京都", ["東京"]) == pytest.approx(0.8)
    assert char_f1("", ["東京"]) == 0.0
    scores = {"closed": {"em": 0.2}, "open": {"em": 0.6}, "compressed": {"em": 0.5}}
    assert retention(scores, "em") == pytest.approx(0.75)


def test_moshirag_triples_pair_each_reference_with_its_utterance(tmp_path):
    record = [
        "Human: 質問1",
        "(augmented)",
        "moshi (lead): えーと",
        "Reference: 参照1",
        "moshi (body): 本文1",
        "Human: 雑談",
        "(unaugmented)",
        "moshi: そうですね",
        "Human: 質問2",
        "(augmented)",
        "moshi (lead): うーん",
        "Reference: 不明です",
        "moshi (body)(subtopic: x): 本文2",
        "Human: 質問3",
        "(augmented)",
        "moshi (lead): はい",
        "Reference: 参照3",
        "moshi (body)(subtopic: y): 本文3",
    ]
    (tmp_path / "a.json").write_text(json.dumps({"record": record}, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({"record": record[:5]}, ensure_ascii=False), encoding="utf-8")
    assert prepare_data.moshirag_triples(tmp_path, {"a"}) == [
        {"passage": "参照1", "question": "質問1", "answer": "本文1"},
        {"passage": "参照3", "question": "質問3", "answer": "本文3"},
    ]
    assert len(prepare_data.moshirag_triples(tmp_path)) == 3


def test_squad_rows_strip_the_title(tmp_path):
    path = tmp_path / "s.json"
    data = {"data": [{"paragraphs": [{"context": "題 [SEP] 本文", "qas": [{"question": "問", "answers": [{"text": "答"}]}, {"question": "無", "answers": []}]}]}]}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert prepare_data.squad_rows(path) == [{"passage": "本文", "question": "問", "answers": ["答"]}]


class ModuloTokenizer(CharTokenizer):
    """Keeps the ids inside the vocabulary of the tiny models."""

    def encode(self, s, bos=False, eos=False):
        return ([1] if bos else []) + [3 + ord(c) % 190 for c in s] + ([2] if eos else [])


@pytest.mark.parametrize("condition", ["closed", "open", "compressed"])
def test_generation_for_each_condition(condition):
    torch.manual_seed(0)
    bridge = BridgeArgs(bridge_type="mlp", in_dim=64, hidden_dim=32, out_dim=64)
    tok = TokenizerWrapper(tokenizer=ModuloTokenizer(), model_name="llmjp4")
    pipeline = EmbedAugPipeline(
        pipeline_args=PipelineArgs(embedder_params=tiny_embedder(), bridge_module=bridge),
        llm_tokenizer=[tok],
        embed_tokenizer=tok,
        embedding_model=Transformer(tiny_args(), embedder_args=tiny_embedder()),
    )
    args = tiny_args()
    args.max_batch_size = 2
    pipeline.store_model(pipeline.get_model(llms=[Transformer(args)]))
    pipeline.model.eval()

    labels = PROMPT_LABELS["ja"]
    shots = [{"passage": "短い文書です", "question": "何？", "answers": ["文書"]}]
    items = [{"passage": "別の文書の中身です", "question": "中身は？", "answers": ["x"]}, {"passage": "三つ目", "question": "何番？", "answers": ["y"]}]
    built = [build_prompt(condition, shots, item, labels) for item in items]
    outputs = pipeline.generate(
        text_to_embed=[b[1] for b in built] if condition == "compressed" else None,
        device="cpu",
        batch_list_prompts=[b[0] for b in built],
        max_tokens=4,
        temperature=0,
    )
    assert len(outputs) == 2 and all(isinstance(o, str) for o in outputs)


def test_check_file_counts_the_bad_lines(tmp_path):
    path = tmp_path / "a.jsonl"
    rows = [{"passage": "文書", "question": "問", "answer": "答"}, {"passage": "文書", "question": "", "answer": "答"}, {"text": "x"}]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    result = prepare_data.check_file(path, {"passage", "question", "answer"})
    assert (result["lines"], result["bad"], result["chars"]) == (3, 2, 2)


def test_visible_devices_accepts_the_uuids_of_abci(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-9aa12bd0,GPU-138503ea")
    assert visible_devices() == ["GPU-9aa12bd0", "GPU-138503ea"]


@pytest.mark.parametrize(
    "mixed_precision, bf16_compute, param_dtype",
    [(True, False, torch.float32), (False, False, torch.bfloat16), (True, True, torch.bfloat16)],
)
def test_fsdp_computes_in_the_chosen_dtype(mixed_precision, bf16_compute, param_dtype):
    args = SimpleNamespace(mixed_precision=mixed_precision, bf16_compute=bf16_compute)
    assert fsdp_precision(args).param_dtype == param_dtype


def test_encode_texts_keeps_the_order_and_the_compression():
    torch.manual_seed(0)
    bridge = BridgeArgs(bridge_type="mlp", in_dim=64, hidden_dim=32, out_dim=64)
    tok = TokenizerWrapper(tokenizer=ModuloTokenizer(), model_name="llmjp4")
    pipeline = EmbedAugPipeline(
        pipeline_args=PipelineArgs(embedder_params=tiny_embedder(), bridge_module=bridge),
        llm_tokenizer=[tok],
        embed_tokenizer=tok,
        embedding_model=Transformer(tiny_args(), embedder_args=tiny_embedder()),
    )
    pipeline.store_model(pipeline.get_model(llms=[Transformer(tiny_args())]))
    pipeline.model.eval()
    texts = ["とても長い参照の文章です。" * 3, "", "短い", "中くらいの長さの参照"]
    vectors = encode_moshirag_refs.encode_texts(pipeline, texts, "cpu", max_batch_tokens=12)
    assert [len(v) for v in vectors] == [math.ceil(len(t) / 4) for t in texts]
    assert all(v.shape[1] == 64 and v.dtype == np.float16 for v in vectors)
    # Batches of other sizes give the same vectors
    alone = encode_moshirag_refs.encode_texts(pipeline, texts[3:], "cpu")
    assert np.allclose(vectors[3], alone[0], atol=1e-3)


def test_match_events_pairs_the_frames_in_order():
    events = [{"ret_frame": 100, "reference": "a"}, {"ret_frame": 300, "reference": "b"}]
    matched = encode_moshirag_refs.match_events([98, 200, 305], events)
    assert matched == [events[0], None, None]
    assert encode_moshirag_refs.match_events([100, 305], events) == events
