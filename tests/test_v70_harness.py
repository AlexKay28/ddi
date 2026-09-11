"""Fail-closed CPU tests for the isolated V70 harness."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from enum import Enum
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import pytest
import torch

# This host has a mismatched optional torchvision wheel. Transformers imports only this
# enum for text-model utilities, so isolate that host defect from the V70 CPU contract.
try:
    import torchvision  # noqa: F401
except (ModuleNotFoundError, RuntimeError):
    for module_name in tuple(sys.modules):
        if module_name == "torchvision" or module_name.startswith("torchvision."):
            del sys.modules[module_name]

    class InterpolationMode(Enum):
        NEAREST = 0
        NEAREST_EXACT = 1
        BOX = 2
        BILINEAR = 3
        HAMMING = 4
        BICUBIC = 5
        LANCZOS = 6

    torchvision = ModuleType("torchvision")
    torchvision.__spec__ = ModuleSpec("torchvision", loader=None)
    transforms = ModuleType("torchvision.transforms")
    transforms.__spec__ = ModuleSpec("torchvision.transforms", loader=None)
    transforms.InterpolationMode = InterpolationMode
    torchvision.transforms = transforms
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.transforms"] = transforms

from transformers import AutoTokenizer, DynamicCache

import scripts.v70_run as v70_run
from ddi.cache import CacheEntry, DocCache
from ddi.canonical_cache import (
    CanonicalCache,
    ExecutionContract,
    model_config_identity,
    runtime_identity,
)
from ddi.link_token_contract import DocumentPayload
from ddi.prompt import encode_record as encode_shared_record
from scripts.v70_harness import (
    EMBEDDING_POLICY,
    EncodedRequest,
    _kv_equal,
    encode_documents_with_grad,
    evaluate_abcd,
    independent_forward,
    install_fixed_link_tokens,
    layout_tensors,
    monolithic_forward,
    paired_bootstrap,
    paired_quality_gate,
    require_adamw8bit,
    teacher_answer_logits,
    training_loss,
)
from scripts.v70_schedule import stable_schedule


class TinyTokenizer:
    def __init__(self, size: int = 32, restored: bool = False):
        self.tokens = {f"base-{index}": index for index in range(size)}
        self.additional_special_tokens = []
        self.eos_token_id = 1
        if restored:
            self.add_special_tokens(
                {"additional_special_tokens": [f"<|ddi_link_{index}|>" for index in range(5)]}
            )

    def __len__(self):
        return len(self.tokens)

    def add_special_tokens(self, values):
        added = 0
        for token in values["additional_special_tokens"]:
            if token not in self.tokens:
                self.tokens[token] = len(self.tokens)
                added += 1
            if token not in self.additional_special_tokens:
                self.additional_special_tokens.append(token)
        return added

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [self.tokens[text]] if text in self.tokens else [2]

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return " ".join(str(item) for item in ids)


class TinyConfig(SimpleNamespace):
    def to_dict(self):
        return dict(vars(self))


class TinyCausalLM(torch.nn.Module):
    """Small real masked-attention decoder with an HF-shaped DynamicCache API."""

    def __init__(self, vocab_size: int = 32, seed: int = 70):
        super().__init__()
        torch.manual_seed(seed)
        self.config = TinyConfig(
            vocab_size=vocab_size,
            hidden_size=8,
            num_hidden_layers=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=128,
            pad_token_id=None,
            model_type="qwen2",
            rope_scaling=None,
            use_sliding_window=False,
            _attn_implementation="eager",
        )
        self.embed = torch.nn.Embedding(vocab_size, 8)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "q": torch.nn.Linear(8, 8, bias=False),
                        "k": torch.nn.Linear(8, 8, bias=False),
                        "v": torch.nn.Linear(8, 8, bias=False),
                        "o": torch.nn.Linear(8, 8, bias=False),
                    }
                )
                for _ in range(2)
            ]
        )
        self.lm_head = torch.nn.Linear(8, vocab_size, bias=False)

    @property
    def device(self):
        return self.embed.weight.device

    @property
    def dtype(self):
        return self.embed.weight.dtype

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def resize_token_embeddings(self, size):
        old_embed = self.embed
        old_head = self.lm_head
        self.embed = torch.nn.Embedding(size, old_embed.embedding_dim)
        self.lm_head = torch.nn.Linear(old_head.in_features, size, bias=False)
        with torch.no_grad():
            self.embed.weight[: old_embed.num_embeddings].copy_(old_embed.weight)
            self.lm_head.weight[: old_head.out_features].copy_(old_head.weight)
        self.config.vocab_size = size
        return self.embed

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values=None,
        use_cache=False,
        cache_position=None,  # noqa: ARG002
    ):
        hidden = self.embed(input_ids) + position_ids[..., None].to(self.dtype) / 100.0
        cache = DynamicCache() if past_key_values is None else past_key_values
        for index, layer in enumerate(self.layers):
            query = layer["q"](hidden)[:, None]
            key = layer["k"](hidden)[:, None]
            value = layer["v"](hidden)[:, None]
            if use_cache:
                key, value = cache.update(key, value, index)
            scores = query @ key.transpose(-1, -2) / key.shape[-1] ** 0.5
            weights = torch.softmax(scores + attention_mask, dim=-1)
            hidden = hidden + layer["o"]((weights @ value).squeeze(1))
        return SimpleNamespace(
            logits=self.lm_head(hidden),
            past_key_values=cache if use_cache else None,
        )


def tiny_model(seed: int = 70):
    return TinyCausalLM(seed=seed)


@pytest.fixture
def setup():
    model = tiny_model()
    tokenizer = TinyTokenizer()
    state = install_fixed_link_tokens(model, tokenizer)
    request = EncodedRequest(
        (DocumentPayload((2, 3)), DocumentPayload((4, 5, 6))),
        (7, 8),
        (9, 10),
    )
    return model, tokenizer, state, request


def test_k5_reaches_actual_model_call_and_shifts_geometry(setup, monkeypatch):
    model, _, state, request = setup
    calls = []
    original = model.forward

    def record(*args, **kwargs):
        calls.append(
            {
                key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                for key, value in kwargs.items()
            }
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "forward", record)
    zero = monolithic_forward(model, state.contract, request, k=0)
    five = monolithic_forward(model, state.contract, request, k=5)
    assert calls[1]["input_ids"].shape[1] - calls[0]["input_ids"].shape[1] == 10
    assert (
        sum(token in state.contract.link_token_ids for token in calls[1]["input_ids"][0].tolist())
        == 10
    )
    assert calls[1]["attention_mask"].shape == (1, 1, 19, 19)
    assert calls[1]["position_ids"].tolist() == [list(five.layout.positions)]
    assert zero.answer_logits.shape[1] == five.answer_logits.shape[1] == 2
    assert zero.logits.shape[1] == 9 and five.logits.shape[1] == 19


def test_actual_treatment_changes_logits_and_k0_is_repeatable(setup):
    model, _, state, request = setup
    model.eval()
    torch.manual_seed(9)
    first = monolithic_forward(model, state.contract, request, k=0).answer_logits
    torch.manual_seed(9)
    repeat = monolithic_forward(model, state.contract, request, k=0).answer_logits
    torch.manual_seed(9)
    treated = monolithic_forward(model, state.contract, request, k=5).answer_logits
    assert torch.equal(first, repeat)
    assert not torch.equal(first, treated)


def test_independent_document_encoding_has_no_cross_document_leakage(setup):
    model, _, _, request = setup
    model.eval()
    original = encode_documents_with_grad(model, request.documents)
    changed = encode_documents_with_grad(
        model, (request.documents[0], DocumentPayload((11, 12, 13)))
    )
    assert all(
        torch.equal(a, b)
        for a, b in zip(
            original[0].k_per_layer + original[0].v_per_layer,
            changed[0].k_per_layer + changed[0].v_per_layer,
            strict=True,
        )
    )
    assert not torch.equal(original[1].k_per_layer[0], changed[1].k_per_layer[0])


def test_independent_path_preserves_autograd_and_decoder_gradient(setup):
    model, _, state, request = setup
    model.train()
    model.get_input_embeddings().weight.requires_grad_(False)
    model.get_output_embeddings().weight.requires_grad_(False)
    result = independent_forward(model, state.contract, request, k=5)
    loss = result.answer_logits.float().square().mean()
    assert loss.grad_fn is not None
    loss.backward()
    gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "layers" in name and parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() for gradient in gradients)
    assert model.get_input_embeddings().weight.grad is None


def test_teacher_target_is_common_and_training_loss_is_connected(setup):
    student, _, state, request = setup
    teacher = tiny_model()
    install_fixed_link_tokens(teacher, TinyTokenizer())
    teacher.eval()
    student.train()
    first = teacher_answer_logits(teacher, request)
    second = teacher_answer_logits(teacher, request)
    assert torch.equal(first, second)
    zero = training_loss(teacher, student, state.contract, request, k=0, encoding="monolithic")
    five = training_loss(teacher, student, state.contract, request, k=5, encoding="independent")
    assert zero.grad_fn is not None and five.grad_fn is not None
    assert torch.isfinite(zero) and torch.isfinite(five)


def test_frozen_optimizer_has_no_adamw_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    parameter = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(RuntimeError, match="AdamW8bit"):
        require_adamw8bit([parameter], lr=1e-3)


def _canonical_contract(model):
    return ExecutionContract(
        weights_identity="tiny-seed-70",
        adapters_identity="none",
        tokenizer_identity="tiny-v70",
        model_config_identity=model_config_identity(model),
        runtime_identity=runtime_identity(model),
        backend="eager",
        dtype=str(model.dtype),
    )


def test_persisted_c_reopens_hits_only_and_matches_b(setup, tmp_path):
    model, _, _, request = setup
    model.eval()
    contract = _canonical_contract(model)
    tensors = [torch.tensor([doc.token_ids]) for doc in request.documents]
    fresh = CanonicalCache(model, contract)
    b_entries = fresh.documents(tensors)
    first_store = DocCache(tmp_path)
    CanonicalCache(model, contract, first_store).documents(tensors)
    size = first_store.size()

    reopened_store = DocCache(tmp_path)
    reopened = CanonicalCache(model, contract, reopened_store)
    reopened.encode_document = lambda _: (_ for _ in ()).throw(AssertionError("encoded"))
    c_entries = reopened.documents(tensors)
    assert reopened_store is not first_store
    assert reopened_store.size() == size == 2
    assert _kv_equal(b_entries, c_entries)


def test_kv_equal_normalizes_detached_cpu_representation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = torch.arange(6, dtype=torch.float32, device=device).reshape(1, 1, 2, 3)
    source.requires_grad_(True)
    persisted = source.detach().cpu().clone()
    left = [CacheEntry("left", [source], [source + 1], 2, "restart")]
    right = [CacheEntry("right", [persisted], [persisted + 1], 2, "restart")]
    assert _kv_equal(left, right)
    assert source.requires_grad and source.grad_fn is None
    assert not _kv_equal(
        left,
        [CacheEntry("dtype", [persisted.double()], [(persisted + 1).double()], 2, "restart")],
    )
    assert not _kv_equal(
        left,
        [CacheEntry("shape", [persisted.reshape(1, 1, 3, 2)], [persisted + 1], 2, "restart")],
    )


def test_abcd_evaluation_stores_per_record_outputs_and_real_gates(setup, tmp_path):
    model, tokenizer, state, request = setup
    model.eval()
    generation_request = EncodedRequest(request.documents, request.query_token_ids)
    result = evaluate_abcd(
        model,
        tokenizer,
        state,
        [("record-1", generation_request, ["2"])],
        _canonical_contract(model),
        tmp_path,
        k=5,
        max_new_tokens=2,
    )
    row = result["rows"][0]
    assert set(row["arms"]) == {"A", "B", "C", "D"}
    assert row["arms"]["B"]["generated_ids"] == row["arms"]["C"]["generated_ids"]
    assert all(
        {"generated_ids", "prediction", "em", "f1", "presence"} <= set(arm)
        for arm in row["arms"].values()
    )
    assert result["gates"]["C-B"]["pass"]
    assert result["gates"]["C-B"]["metrics"]["em"]["resamples"] == 10_000
    assert result["cache_validation"]["reopened_store"]
    assert result["cache_validation"]["hit_only"]
    assert result["cache_validation"]["entry_count_before"] == 2
    assert result["cache_validation"]["entry_count_after"] == 2


def test_schedule_is_stable_across_pythonhashseed_subprocesses():
    code = (
        "import json; from scripts.v70_schedule import stable_schedule; "
        "print(json.dumps(stable_schedule('record-7', 3, 47).__dict__, sort_keys=True))"
    )
    outputs = []
    for hash_seed in ("1", "999"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        outputs.append(subprocess.check_output([sys.executable, "-c", code], env=env, text=True))
    assert outputs[0] == outputs[1]
    assert stable_schedule("record-7", 3, 47) == stable_schedule("record-7", 3, 47)


def test_paired_bootstrap_and_gate_are_paired_and_fail_closed():
    constant = paired_bootstrap([0.1, 0.1], resamples=100)
    assert constant["ci95"] == pytest.approx([0.1, 0.1])
    rows = [
        {
            "arms": {
                "C": {"em": 1.0, "f1": 0.8},
                "B": {"em": 1.0, "f1": 0.8},
                "D": {"em": 1.0, "f1": 1.0},
            }
        }
        for _ in range(4)
    ]
    assert paired_quality_gate(rows, candidate="C", reference="B")["pass"]
    gate = paired_quality_gate(rows, candidate="C", reference="D")
    assert not gate["pass"]
    assert gate["metrics"]["f1"]["ci95"][0] < -0.02
    with pytest.raises(ValueError):
        paired_bootstrap([])


def test_fixed_marker_checkpoint_restore_keeps_rows_exact():
    model = tiny_model()
    tokenizer = TinyTokenizer()
    state = install_fixed_link_tokens(model, tokenizer)
    rows = model.get_input_embeddings().weight[list(state.contract.link_token_ids)].detach().clone()
    state_dict = deepcopy(model.state_dict())
    restored = TinyCausalLM(vocab_size=37)
    restored.config = TinyConfig(**model.config.to_dict())
    restored.load_state_dict(state_dict)
    restored_state = install_fixed_link_tokens(restored, TinyTokenizer(restored=True))
    assert restored_state.embedding_policy == EMBEDDING_POLICY
    assert restored_state.contract.link_token_ids == state.contract.link_token_ids
    assert torch.equal(
        restored.get_input_embeddings().weight[list(restored_state.contract.link_token_ids)], rows
    )
    assert not restored.get_input_embeddings().weight.requires_grad


def test_layout_tensor_is_real_4d_additive_mask(setup):
    model, _, state, request = setup
    result = monolithic_forward(model, state.contract, request, k=5)
    _, mask, _ = layout_tensors(result.layout, model.dtype, model.device)
    assert mask.ndim == 4
    assert mask[0, 0, 0, 0] == 0
    assert torch.isneginf(mask[0, 0, 0, 1])


def test_runner_record_encoding_preserves_exact_segment_boundaries():
    path = "results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L_restart"
    if not os.path.isdir(path):
        pytest.skip("real local tokenizer checkpoint unavailable")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    record = {
        "id": "boundary-record",
        "info_contexts": ["First document has no trailing space", "Second starts directly."],
        "dialog": [
            {"role": "user", "content": "Which document starts directly?"},
            {"role": "assistant", "content": "Second"},
        ],
        "possible_answers": ["Second"],
    }
    request = v70_run.encode_real_record(tokenizer, record)
    full_ids, doc_lengths, query_len, answer_len = encode_shared_record(tokenizer, record)
    flattened = [
        *(token for document in request.documents for token in document.token_ids),
        *request.query_token_ids,
        *request.answer_token_ids,
    ]
    assert flattened == full_ids[0].tolist()
    assert [len(document.token_ids) for document in request.documents] == doc_lengths
    assert len(request.query_token_ids) == query_len
    assert len(request.answer_token_ids) == answer_len


def test_runner_checkpoint_is_arm_specific_and_carries_v70_metadata(tmp_path):
    model = tiny_model()
    tokenizer = TinyTokenizer()
    model.save_pretrained = lambda path: (path / "model.test").write_text("model")
    tokenizer.save_pretrained = lambda path: (path / "tokenizer.test").write_text("tokenizer")
    output = tmp_path / "checkpoint_k5"
    v70_run.save_v70_checkpoint(model, tokenizer, output, arm="k5", step=3, seed=47)
    metadata = __import__("json").loads((output / v70_run.V70_METADATA).read_text())
    assert metadata["arm"] == "k5" and metadata["k"] == 5 and metadata["step"] == 3
    assert metadata["fixed_marker_conditioning"] is True
    assert metadata["learned_link_tokens"] is False
    assert metadata["effective_batch_size"] == 4
    assert metadata["warmup_optimizer_steps"] == 20
    with pytest.raises(ValueError, match="arm-specific"):
        v70_run.save_v70_checkpoint(
            model, tokenizer, tmp_path / "shared", arm="k0", step=1, seed=47
        )


def test_runner_evaluation_entrypoint_delegates_to_evaluate_abcd(setup, tmp_path, monkeypatch):
    model, tokenizer, state, request = setup
    tokenizer.name_or_path = str(tmp_path)
    (tmp_path / "tokenizer.json").write_text("{}")
    observed = {}

    def fake_evaluate(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {"schema": "called"}

    monkeypatch.setattr(v70_run, "evaluate_abcd", fake_evaluate)
    result = v70_run.run_evaluation(
        model,
        tokenizer,
        state,
        [("record", EncodedRequest(request.documents, request.query_token_ids), ["gold"])],
        tmp_path / "cache",
        weights_identity="weights",
        k=5,
        max_new_tokens=2,
    )
    assert result == {"schema": "called"}
    assert observed["args"][0] is model
    assert observed["kwargs"] == {"k": 5, "max_new_tokens": 2}


def test_runner_accumulates_four_backwards_per_optimizer_step(tmp_path, monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    class FakeOptimizer:
        def __init__(self):
            self.param_groups = [{"lr": 99.0}]
            self.zero_calls = 0
            self.step_gradients = []

        def zero_grad(self, set_to_none=False):
            self.zero_calls += 1
            parameter.grad = None if set_to_none else torch.zeros_like(parameter)

        def step(self):
            self.step_gradients.append(float(parameter.grad))

    class FakeModel:
        config = SimpleNamespace(vocab_size=37)

        def save_pretrained(self, path):
            (path / "model.test").write_text("model")

    class FakeTokenizer:
        def save_pretrained(self, path):
            (path / "tokenizer.test").write_text("tokenizer")

    optimizer = FakeOptimizer()
    records = [{"id": f"record-{index}"} for index in range(4)]
    monkeypatch.setattr(v70_run, "configure_cache_dirs", lambda: None)
    monkeypatch.setattr(v70_run, "_gpu_idle_snapshot", lambda: {"state": "idle"})
    monkeypatch.setattr(
        v70_run,
        "_prepare_teacher",
        lambda checkpoint: (FakeModel(), FakeTokenizer(), None),
    )
    monkeypatch.setattr(
        v70_run,
        "_prepare_student",
        lambda checkpoint, seed: (
            FakeModel(),
            FakeTokenizer(),
            SimpleNamespace(contract=None),
            optimizer,
            [parameter],
        ),
    )
    monkeypatch.setattr(v70_run, "_load_common_records", lambda seed: records)
    monkeypatch.setattr(v70_run, "encode_real_record", lambda tokenizer, record: record)
    backward_losses = []

    def fake_loss(teacher, student, contract, request, **kwargs):
        value = int(request["id"].split("-")[1]) + 1
        loss = parameter * value
        loss.register_hook(lambda gradient: backward_losses.append(float(gradient)))
        return loss

    monkeypatch.setattr(v70_run, "training_loss", fake_loss)
    monkeypatch.setattr(v70_run, "_assert_finite_weights", lambda *args: None)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *args: None)

    checkpoint = tmp_path / "new-output" / "checkpoint_k0"
    v70_run.train_arm("k0", 1, checkpoint, 47)

    assert backward_losses == pytest.approx([0.25] * 4)
    assert optimizer.step_gradients == pytest.approx([2.5])
    assert optimizer.zero_calls == 2
    loss_lines = (tmp_path / "new-output" / "checkpoint_k0.losses.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in loss_lines]
    assert len(rows) == 1
    assert rows[0]["record_ids"] == [f"record-{index}" for index in range(4)]
    assert rows[0]["presentation_losses"] == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert rows[0]["mean_loss"] == pytest.approx(2.5)
    assert rows[0]["lr"] == pytest.approx(5e-7)
    assert len(rows[0]["schedules"]) == 4


def test_runner_warmup_is_linear_then_constant():
    assert v70_run.learning_rate_for_step(1) == pytest.approx(5e-7)
    assert v70_run.learning_rate_for_step(10) == pytest.approx(5e-6)
    assert v70_run.learning_rate_for_step(20) == pytest.approx(1e-5)
    assert v70_run.learning_rate_for_step(21) == pytest.approx(1e-5)
    with pytest.raises(ValueError):
        v70_run.learning_rate_for_step(0)


def test_k0_k5_training_plans_have_identical_records_and_schedules():
    records = [{"id": f"record-{index}"} for index in range(7)]
    k0 = v70_run.training_plan(records, 5, 47)
    k5 = v70_run.training_plan(records, 5, 47)
    def flatten(plan):
        return [(record["id"], schedule) for step in plan for record, schedule in step]
    assert flatten(k0) == flatten(k5)
    assert [record_id for record_id, _ in flatten(k0)] == [
        f"record-{index % 7}" for index in range(20)
    ]
    assert all(len(step) == 4 for step in k0)


def test_confirmation300_is_frozen_unique_subset_and_disjoint():
    confirmation = json.loads(
        (v70_run.ROOT / "experiments/v70_confirmation300.json").read_text()
    )
    v62 = json.loads((v70_run.ROOT / "experiments/v62_split_manifest.json").read_text())
    v67 = json.loads((v70_run.ROOT / "experiments/v67_fresh_dev_split.json").read_text())
    v58 = json.loads(
        (
            v70_run.ROOT
            / "results/v58_cache_qa_100_20260908T122202Z/manifest.json"
        ).read_text()
    )
    v60 = json.loads(
        (v70_run.ROOT / "experiments/v60_confirmation_manifest.json").read_text()
    )
    ids = confirmation["confirmation"]["record_ids"]
    digest = __import__("hashlib").sha256("\n".join(sorted(ids)).encode()).hexdigest()
    assert len(ids) == len(set(ids)) == 300
    assert ids[:100] == v62["splits"]["confirmation"]["record_ids"]
    assert not set(ids[100:]) & set(v62["splits"]["confirmation"]["record_ids"])
    assert digest == confirmation["confirmation"]["sha256"]
    excluded = {
        *v62["splits"]["train"]["record_ids"],
        *v62["splits"]["development"]["record_ids"],
        *v67["development"]["record_ids"],
        *v58["config"]["record_ids"],
        *v60["config"]["record_ids"],
    }
    assert not set(ids) & excluded


def test_runner_loads_frozen_eval_manifests_with_exact_count_and_order():
    dev = v70_run.load_frozen_eval_ids("dev")
    confirmation100 = v70_run.load_frozen_eval_ids("confirmation100")
    confirmation300 = v70_run.load_frozen_eval_ids("confirmation300")
    frozen_dev = json.loads(
        (v70_run.ROOT / "experiments/v67_fresh_dev_split.json").read_text()
    )["development"]["record_ids"]
    frozen_confirmation = json.loads(
        (v70_run.ROOT / "experiments/v70_confirmation300.json").read_text()
    )["confirmation"]["record_ids"]
    assert dev["record_ids"] == frozen_dev
    assert confirmation100["record_ids"] == frozen_confirmation[:100]
    assert confirmation300["record_ids"] == frozen_confirmation
    assert [dev["count"], confirmation100["count"], confirmation300["count"]] == [100, 100, 300]
    assert dev["data_role"] == "development_selection"


@pytest.mark.parametrize("split", v70_run.PRIMARY_EVAL_SPLITS)
def test_runner_refuses_primary_split_truncation(split):
    with pytest.raises(ValueError, match="--records is forbidden"):
        v70_run.load_frozen_eval_ids(split, records=1)


def test_runner_selects_dataset_records_in_manifest_order_and_rejects_missing(monkeypatch):
    split_info = {"record_ids": ["second", "first"]}
    monkeypatch.setattr(
        v70_run,
        "load_dataset",
        lambda *args: [{"id": "first"}, {"id": "unused"}, {"id": "second"}],
    )
    assert [record["id"] for record in v70_run.load_frozen_eval_records(split_info)] == [
        "second",
        "first",
    ]
    split_info["record_ids"].append("missing")
    with pytest.raises(ValueError, match="missing frozen IDs"):
        v70_run.load_frozen_eval_records(split_info)


def test_runner_evaluation_output_contains_full_provenance(tmp_path):
    checkpoint = tmp_path / "checkpoint_k5"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model":"tiny"}')
    (checkpoint / "tokenizer.json").write_text('{"tokenizer":"tiny"}')
    (checkpoint / "weights.test").write_text("weights")
    checkpoint_hashes = v70_run._checkpoint_hashes(checkpoint)
    split_info = v70_run.load_frozen_eval_ids("dev")
    raw = {"schema": "v70-abcd-v1", "rows": [{"record_id": "raw"}], "gates": {"C-B": {}}}
    output = v70_run.build_evaluation_output(
        raw,
        checkpoint=checkpoint,
        checkpoint_hashes=checkpoint_hashes,
        split_info=split_info,
        k=5,
        max_new_tokens=64,
        elapsed_seconds=1.25,
    )
    assert output["schema"] == "v70-evaluation-v2"
    assert output["rows"] is raw["rows"] and output["gates"] is raw["gates"]
    assert output["split"] == "dev" and output["record_count"] == 100
    assert output["elapsed_seconds"] == 1.25 and output["max_new_tokens"] == 64
    provenance = output["provenance"]
    assert provenance["checkpoint_sha256"] == checkpoint_hashes["sha256"]
    assert provenance["config_sha256"] == checkpoint_hashes["files"]["config.json"]
    assert provenance["tokenizer_sha256"] == checkpoint_hashes["files"]["tokenizer.json"]
    assert provenance["record_ids"] == split_info["record_ids"]
    assert set(provenance["source_sha256"]) == set(v70_run.SOURCE_PATHS)
    assert v70_run.evaluation_cache_dir(
        tmp_path / "out.json", checkpoint, "dev", checkpoint_hashes["sha256"]
    ) != v70_run.evaluation_cache_dir(
        tmp_path / "out.json", checkpoint, "confirmation100", checkpoint_hashes["sha256"]
    )


@pytest.mark.skipif(
    not os.path.isdir("results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L_restart"),
    reason="real local tokenizer checkpoint unavailable",
)
def test_real_tokenizer_link_contract_validity():
    path = "results/checkpoints/Qwen__Qwen2.5-7B-Instruct__ddi_qf__8L_restart"
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    base_size = len(tokenizer)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [f"<|ddi_link_{index}|>" for index in range(5)]}
    )
    ids = [
        tokenizer.encode(token, add_special_tokens=False)
        for token in tokenizer.additional_special_tokens[-5:]
    ]
    assert added == 5
    assert all(len(item) == 1 for item in ids)
    flat = [item[0] for item in ids]
    assert len(set(flat)) == 5
    assert min(flat) >= base_size
