"""Tests for `ddi.prompt` — the shared query-text builder and the shared encoder."""

from __future__ import annotations

import types
import zlib
from pathlib import Path

import pytest
import torch

from ddi.prompt import DOC_INSTRUCTION, build_query_text, build_user_dialog


class _FakeTokenizer:
    """Minimal stand-in for a HF tokenizer's chat-template surface."""

    def apply_chat_template(self, dialog, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        body = "".join(f"<|{t['role']}|>{t['content']}" for t in dialog)
        return body + ("<|assistant|>" if add_generation_prompt else "")


def _record(question="who wrote it", docs=("doc one", "doc two"), meta=None):
    return {
        "id": "r1",
        "info_contexts": list(docs),
        "dialog": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "someone"},
        ],
        "possible_answers": ["someone"],
        "meta": meta or {},
    }


def test_instruction_is_added_when_documents_exist() -> None:
    dialog = build_user_dialog(_record())
    assert len(dialog) == 1
    assert DOC_INSTRUCTION in dialog[0]["content"]
    assert "who wrote it" in dialog[0]["content"]


def test_assistant_turn_is_dropped() -> None:
    dialog = build_user_dialog(_record())
    assert [t["role"] for t in dialog] == ["user"]


def test_no_documents_means_no_instruction() -> None:
    """Nothing to point at — a "use the passages above" line would be a lie."""
    dialog = build_user_dialog(_record(docs=()))
    assert dialog[0]["content"] == "who wrote it"


def test_self_instructed_records_are_left_alone() -> None:
    """synthetic_qa writes its own instruction; double-wrapping would fight it."""
    dialog = build_user_dialog(_record(meta={"self_instructed": True}))
    assert dialog[0]["content"] == "who wrote it"


def test_original_record_is_not_mutated() -> None:
    rec = _record()
    build_user_dialog(rec)
    assert rec["dialog"][0]["content"] == "who wrote it"


def test_env_kill_switch_restores_bare_question(monkeypatch) -> None:
    monkeypatch.setenv("DDI_PROMPT_INSTRUCTION", "0")
    dialog = build_user_dialog(_record())
    assert dialog[0]["content"] == "who wrote it"


def test_build_query_text_uses_the_chat_template(monkeypatch) -> None:
    text = build_query_text(_FakeTokenizer(), _record())
    assert text.startswith("<|user|>")
    assert text.endswith("<|assistant|>")
    assert DOC_INSTRUCTION in text


@pytest.mark.parametrize("role", ["system", "assistant"])
def test_non_user_final_turn_is_untouched(role) -> None:
    rec = _record()
    rec["dialog"] = [{"role": role, "content": "x"}, {"role": "assistant", "content": "y"}]
    dialog = build_user_dialog(rec)
    assert dialog[0]["content"] == "x"


# --- reasoning templates: the Qwen3 <think> trap ---------------------------------------


class _TemplateSpyTokenizer:
    """Minimal stand-in — `apply_chat_template` records what it was handed."""

    def __init__(self, chat_template):
        self.chat_template = chat_template
        self.seen: dict = {}

    def apply_chat_template(self, dialog, **kwargs):
        self.seen = dict(kwargs)
        return "RENDERED"


_QWEN25 = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>{% endfor %}"
_QWEN3 = (
    "{% if enable_thinking is defined and enable_thinking %}<think>\n{% endif %}"
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>{% endfor %}"
)


def test_thinking_disabled_detects_reasoning_templates() -> None:
    from ddi.prompt import thinking_disabled

    assert thinking_disabled(_TemplateSpyTokenizer(_QWEN3)) is True
    assert thinking_disabled(_TemplateSpyTokenizer(_QWEN25)) is False
    assert thinking_disabled(_TemplateSpyTokenizer(None)) is False


def test_reasoning_template_gets_enable_thinking_false() -> None:
    """The whole point: a <think> block eats the 64-token budget before the answer."""
    from ddi.prompt import build_query_text

    tok = _TemplateSpyTokenizer(_QWEN3)
    build_query_text(tok, {"dialog": [{"role": "user", "content": "q"}]})
    assert tok.seen.get("enable_thinking") is False
    assert tok.seen.get("add_generation_prompt") is True


def test_plain_template_is_not_passed_the_flag() -> None:
    """Qwen2.5 and friends must be called exactly as before — no behaviour change."""
    from ddi.prompt import build_query_text

    tok = _TemplateSpyTokenizer(_QWEN25)
    build_query_text(tok, {"dialog": [{"role": "user", "content": "q"}]})
    assert "enable_thinking" not in tok.seen
    assert tok.seen.get("add_generation_prompt") is True


# --- the cross-path encoding contract --------------------------------------------------
#
# `ddi.prompt.encode_segments` is the only place in the repo that turns documents, a query
# and an answer into a token stream. It has to be, because the research claim is a
# comparison between three measurement paths — eval loss/generation (`ddi/run.py`),
# training (`ddi/finetune.py`) and perplexity (`ddi/perplexity.py`) — and a comparison
# between three different tokenisations measures nothing.
#
# That contract used to be stated in prose ("matches ddi/run.py::_quality_cell's
# tokenization exactly") next to five hand-written copies of the same block, and it broke:
# the `_cache_restart_quality` copy drifted off `build_query_text` and silently lost
# `DOC_INSTRUCTION` (em == 0.0000 in every cell) and `enable_thinking=False` (a Qwen3
# template spending all 64 max_new_tokens inside `<think>`). This test is what was missing.
#
# It drives the real call sites, not `encode_segments` directly — a site that stops calling
# the shared encoder is exactly the failure being guarded against, so testing the encoder
# alone would pass right through it.


def _stable_ids(text: str) -> list[int]:
    """Deterministic word-level ids. `hash()` is salted per process; crc32 is not."""
    return [(zlib.crc32(w.encode()) % 4000) + 1 for w in text.split()] or [1]


class _ContractTokenizer:
    """Word-level tokenizer with a chat-template surface that reacts to `enable_thinking`."""

    eos_token_id = 0

    def __init__(self, chat_template: str | None = None) -> None:
        self.chat_template = chat_template

    def __call__(self, text, return_tensors=None, add_special_tokens=None):  # noqa: ARG002
        return types.SimpleNamespace(
            input_ids=torch.tensor([_stable_ids(text)], dtype=torch.long)
        )

    def apply_chat_template(  # noqa: ARG002
        self, dialog, tokenize=False, add_generation_prompt=False, **kw
    ):
        flag = f" think={kw['enable_thinking']}" if "enable_thinking" in kw else ""
        body = " ".join(f"<{m['role']}> {m['content']}" for m in dialog)
        return body + (" <gen>" if add_generation_prompt else "") + flag

    def decode(self, ids, skip_special_tokens=False):  # noqa: ARG002
        return " ".join(str(i) for i in ids)


class _RecordingModel(torch.nn.Module):
    """Records the `input_ids` of every prefill it is handed. Decode steps are ignored."""

    def __init__(self, vocab: int = 4096) -> None:
        super().__init__()
        self.vocab = vocab
        self._p = torch.nn.Parameter(torch.zeros(1))
        self.prefills: list[list[int]] = []
        self.config = types.SimpleNamespace(vocab_size=vocab)

    @property
    def device(self) -> torch.device:
        return self._p.device

    @property
    def dtype(self) -> torch.dtype:
        return self._p.dtype

    def forward(  # noqa: PLR0913 — mirrors the HF signature the code under test calls
        self,
        input_ids,
        attention_mask=None,  # noqa: ARG002
        position_ids=None,  # noqa: ARG002
        past_key_values=None,
        use_cache=False,  # noqa: ARG002
        **kw,  # noqa: ARG002
    ):
        if past_key_values is None:
            self.prefills.append([int(x) for x in input_ids[0].tolist()])
        seq = int(input_ids.shape[-1])
        logits = torch.zeros(1, seq, self.vocab, device=input_ids.device)
        logits[..., 3] = 1.0
        past_len = len(past_key_values) if past_key_values is not None else 0
        return types.SimpleNamespace(
            logits=logits, past_key_values=_LenOnlyCache(past_len + seq), hidden_states=None
        )


class _LenOnlyCache:
    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n


def _contract_record(rid="c0"):
    return {
        "id": rid,
        "info_contexts": ["alpha bravo charlie", "delta echo"],
        "dialog": [
            {"role": "user", "content": "who wrote it"},
            {"role": "assistant", "content": "someone famous"},
        ],
        "possible_answers": ["someone famous"],
        "meta": {},
    }


@pytest.mark.parametrize("chat_template", [None, _QWEN3], ids=["plain", "reasoning"])
def test_every_path_encodes_a_record_identically(
    chat_template, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """eval loss, restart-quality loss, training, perplexity and generation agree
    token-for-token."""
    from ddi import inference
    from ddi import run as ddi_run
    from ddi.finetune import _encode_record
    from ddi.perplexity import compute_perplexity

    tok = _ContractTokenizer(chat_template)
    record = _contract_record()
    monkeypatch.setattr(ddi_run, "load_dataset", lambda *a, **k: [dict(record)])  # noqa: ARG005

    # 1. training path.
    train_ids, train_docs, train_qlen, train_alen = _encode_record(tok, record)
    train_ids = [int(x) for x in train_ids[0].tolist()]
    assert train_alen > 0, "fixture must have answer tokens or the comparison is vacuous"

    # 2. eval loss path (`_quality_cell`), 3. its generation prefill (`inference.encode`).
    quality_model = _RecordingModel()
    ddi_run._quality_cell(
        model=quality_model,
        tokenizer=tok,
        dataset="synthetic",
        mask_variant="full",
        bridge_k=None,
        model_name="fake",
        seed=0,
        max_records=None,
        max_new_tokens=2,
        metrics=["loss"],
        cfg_hash="deadbeef",
        store_path=tmp_path / "quality.jsonl",
    )
    quality_loss_ids, quality_gen_ids = quality_model.prefills

    # 4. restart-quality path — the copy that actually drifted.
    restart_model = _RecordingModel()
    monkeypatch.setattr(inference, "load", lambda *a, **k: (restart_model, tok))  # noqa: ARG005
    ddi_run._cache_restart_quality(
        {
            "model": "fake",
            "datasets": ["synthetic"],
            "masks": ["full"],
            "layouts": ["sequential"],
            "max_new_tokens": 2,
        },
        "deadbeef",
        0,
        tmp_path / "restart.jsonl",
    )
    restart_loss_ids, _restart_gen_ids = restart_model.prefills

    # 5. perplexity path.
    ppl_model = _RecordingModel()
    compute_perplexity(ppl_model, tok, [record], mask_variant="full", layout="sequential")
    (ppl_ids,) = ppl_model.prefills

    assert quality_loss_ids == train_ids
    assert restart_loss_ids == train_ids
    assert ppl_ids == train_ids

    # The generation path has no answer yet, so it must be exactly the answer-free prefix.
    assert quality_gen_ids == train_ids[: len(train_ids) - train_alen]

    # And the block structure the masks key on agrees too — identical ids with different
    # doc boundaries would still misalign every mask.
    batch = inference.encode(
        tok, docs=list(record["info_contexts"]), query=build_query_text(tok, record)
    )
    assert batch.doc_lengths == train_docs
    assert batch.query_len == train_qlen
    assert sum(train_docs) + train_qlen + train_alen == len(train_ids)


def test_reasoning_template_reaches_the_encoded_stream() -> None:
    """The `enable_thinking=False` flag must survive into the token ids, not just the text.

    Pins the second half of the drift: a caller that reached the right `encode_*` entry
    point but built its own query text would still lose the flag, and the failure looks
    like a mask effect (`em` collapses, `loss` does not) rather than a prompt bug.
    """
    from ddi.prompt import encode_record

    plain, _, _, _ = encode_record(_ContractTokenizer(_QWEN25), _contract_record())
    reasoning, _, _, _ = encode_record(_ContractTokenizer(_QWEN3), _contract_record())
    assert plain.shape != reasoning.shape or not torch.equal(plain, reasoning)


def test_encode_segments_handles_a_record_with_no_documents() -> None:
    """No-doc records are a real dataset shape (kilt_nq closed-book rows), not an edge case."""
    from ddi.prompt import encode_segments

    tok = _ContractTokenizer()
    ids, doc_lengths, query_len, answer_len = encode_segments(tok, [], "one two three", "four")
    assert doc_lengths == []
    assert query_len == 3
    assert answer_len == 1
    assert ids.shape == (1, 4)
