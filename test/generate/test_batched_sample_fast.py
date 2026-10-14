# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock

import pytest
import torch

import keys_values.generate.base as generate


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("temperature", [1.0, 0.6, 2.0])
@pytest.mark.parametrize("kwargs_kind", ["dict", "shared_list", "equal_list"])
def test_batched_sample_fast_path_matches_unfiltered_probabilities(
    dtype, temperature, kwargs_kind
):
    # A nontrivial sequence length verifies we sample from the final position.
    logits = torch.tensor(
        [
            [[1.0, 0.0, 2.0, -1.0], [3.0, 1.0, 2.0, -2.0]],
            [[0.0, 5.0, -2.0, 1.0], [-1.0, 2.0, 0.0, 3.0]],
            [[1.0, -1.0, 0.0, 0.0], [0.0, 0.5, 1.0, -2.0]],
        ],
        dtype=dtype,
    )
    params = dict(temperature=temperature, top_k=None, top_p=1.0)
    if kwargs_kind == "dict":
        kwargs = params
    elif kwargs_kind == "shared_list":
        kwargs = [params] * logits.size(0)
    else:
        kwargs = [params.copy() for _ in range(logits.size(0))]

    seen_probs = []

    def fake_multinomial(probs):
        seen_probs.append(probs)
        return probs.argmax(dim=-1, keepdim=True)

    with mock.patch.object(
        generate, "sample", side_effect=AssertionError("Expected batched sampling")
    ), mock.patch.object(
        generate, "multinomial_num_samples_1", side_effect=fake_multinomial
    ):
        tokens = generate.batched_sample(logits, kwargs)

    expected_probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
    assert len(seen_probs) == 1
    assert seen_probs[0].dtype == dtype
    torch.testing.assert_close(seen_probs[0], expected_probs, rtol=0, atol=0)
    torch.testing.assert_close(tokens, expected_probs.argmax(dim=-1, keepdim=True))
    assert tokens.dtype == torch.int64
    assert tokens.shape == (3, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 1},
        {"top_k": 2, "temperature": 0.8},
        {"top_p": 0.8},
        {"top_p": 0.0},
        {"temperature": 0.0},
        {"temperature": -1.0},
        [{"temperature": 1.0}, {"temperature": 0.7}, {"temperature": 1.0}],
        [{"top_p": 1.0}, {"top_p": 0.7}, {"top_p": 1.0}],
    ],
)
def test_batched_sample_fallback_delegates_to_litgpt(kwargs):
    logits = torch.arange(3 * 2 * 8, dtype=torch.float32).view(3, 2, 8)
    sample_calls = []

    def fake_sample(row_logits, **sample_kwargs):
        sample_calls.append((row_logits.clone(), sample_kwargs))
        return row_logits[0, -1].argmax(dim=-1, keepdim=True)

    with mock.patch.object(
        generate,
        "multinomial_num_samples_1",
        side_effect=AssertionError("Expected fallback"),
    ), mock.patch.object(generate, "sample", side_effect=fake_sample):
        tokens = generate.batched_sample(logits, kwargs)

    assert len(sample_calls) == 3
    expected_kwargs = kwargs if isinstance(kwargs, list) else [kwargs] * 3
    for batch_idx, (row_logits, passed_kwargs) in enumerate(sample_calls):
        torch.testing.assert_close(row_logits, logits[batch_idx].unsqueeze(0))
        assert passed_kwargs == expected_kwargs[batch_idx]
    torch.testing.assert_close(tokens, torch.full((3, 1), 7, dtype=torch.int64))


def test_batched_sample_fallback_for_single_row():
    logits = torch.randn(1, 1, 16)
    with mock.patch.object(
        generate,
        "multinomial_num_samples_1",
        side_effect=AssertionError("Expected fallback"),
    ), mock.patch.object(generate, "sample", return_value=torch.tensor([4])) as patched:
        token = generate.batched_sample(logits, {"temperature": 1.0})
    assert patched.call_count == 1
    torch.testing.assert_close(token, torch.tensor([[4]], dtype=torch.int64))


def test_batched_sample_fallback_for_unrecognized_kwargs():
    # Let LitGPT validate unknown options rather than silently discarding them.
    logits = torch.randn(3, 1, 8)
    with mock.patch.object(generate, "sample", side_effect=TypeError("unknown option")):
        with pytest.raises(TypeError, match="unknown option"):
            generate.batched_sample(logits, {"bad_option": 1})


def test_batched_sample_fast_path_distribution():
    # Fixed logits across rows let us check that batched multinomial draws
    # from the same categorical distribution, without requiring the same RNG
    # stream as a sequence of per-row multinomial calls.
    torch.manual_seed(2026)
    logits = torch.tensor([0.0, 1.0, 2.0, -0.4]).view(1, 1, -1).expand(4096, -1, -1)
    tokens = generate.batched_sample(logits, {"temperature": 0.8})
    frequencies = torch.bincount(tokens.flatten(), minlength=4).float() / 4096
    expected = torch.softmax(logits[0, 0] / 0.8, dim=-1)
    torch.testing.assert_close(frequencies, expected, rtol=0, atol=0.03)


def test_batched_sample_fast_path_uses_litgpt_multinomial():
    logits = torch.randn(4, 1, 16)
    with mock.patch.object(
        generate, "multinomial_num_samples_1", wraps=generate.multinomial_num_samples_1
    ) as helper:
        tokens = generate.batched_sample(logits, {})
    helper.assert_called_once()
    assert tokens.shape == (4, 1)
    assert tokens.dtype == torch.int64


def test_batched_sample_fast_path_with_litgpt_compilation_semantics():
    # LitGPT switches to its compile-friendly exponential sampler when
    # torch._dynamo is compiling. We must delegate to that same helper.
    torch.manual_seed(123)
    logits = torch.randn(4, 1, 12)
    with mock.patch("torch._dynamo.is_compiling", return_value=True) as compiling:
        tokens = generate.batched_sample(logits, {})
    compiling.assert_called()
    assert tokens.shape == (4, 1)
    assert ((tokens >= 0) & (tokens < logits.size(-1))).all()


def test_batched_sample_fallback_validates_top_p_and_batch_size():
    logits = torch.randn(3, 1, 8)
    with pytest.raises(ValueError, match="top_p"):
        generate.batched_sample(logits, {"top_p": 1.1})
    with pytest.raises(AssertionError, match="length"):
        generate.batched_sample(logits, [{"temperature": 1.0}])
