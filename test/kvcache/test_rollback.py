# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from itertools import product
from typing import Any, Dict, List

import pytest
import torch

from keys_values.kvcache.base import KVCacheParams
from keys_values.kvcache.test_utils import (
    available_backends,
    create_kv_cache,
    random_args_cache_forward,
    range_from_args,
)

CACHE_LENGTH = 16

# Prefill, warm-up and window sizes are chosen so that the ring buffer of
# `lastrec` wraps around inside the rollback window in some cases.
SCENARIOS = [
    (prefill, warmup, window)
    for prefill, warmup, window in product(
        (6, CACHE_LENGTH),
        (0, 3, 9),
        ([5], [1, 1, 1], [3, 1, 2], [2, 4]),
    )
]


def _params(dtype: torch.dtype) -> KVCacheParams:
    return KVCacheParams(
        max_batch_size=2,
        n_query_groups=2,
        cache_length=CACHE_LENGTH,
        head_size=8,
        n_head=4,
        dtype=dtype,
    )


def _cache_state(kv_cache) -> Dict[str, Any]:
    current_length = kv_cache.current_length
    k_and_v = kv_cache.get_keys_values()
    return dict(
        input_pos=kv_cache.input_pos,
        current_length=current_length,
        token_positions=kv_cache.token_positions()[0, 0, :].tolist(),
        next_position=getattr(kv_cache, "next_position", None),
        keys=k_and_v.keys()[:, :, :current_length, :].clone(),
        values=k_and_v.values()[:, :, :current_length, :].clone(),
    )


def _run_scenario(
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    prefill: int,
    warmup: int,
    window: List[int],
    num_rollback: int,
    use_rollback: bool,
    seed: int,
) -> Dict[str, Any]:
    """
    If `use_rollback`, inserts all tokens of `window` after a rollback point,
    then removes the last `num_rollback` of them. Otherwise, the same chunks
    are inserted, but truncated to their first `sum(window) - num_rollback`
    tokens. Finally, one more token is inserted, and the cache state and this
    attention output are returned.

    """
    params = _params(dtype)
    torch.random.manual_seed(seed)
    total = prefill + warmup + sum(window) + 1
    data = random_args_cache_forward(params, total, vocab_size=64, device=device)
    kv_cache = create_kv_cache(name, params)
    kv_cache(**range_from_args(data, 0, prefill))
    pos = prefill
    for _ in range(warmup):
        kv_cache(**range_from_args(data, pos, pos + 1))
        pos += 1
    if use_rollback:
        kv_cache.mark_rollback_point()
        for num in window:
            kv_cache(**range_from_args(data, pos, pos + num))
            pos += num
        kv_cache.rollback(num_rollback)
        assert not kv_cache.rollback_point_is_marked
    else:
        num_left = sum(window) - num_rollback
        for num in window:
            num_ins = min(num, num_left)
            if num_ins > 0:
                kv_cache(**range_from_args(data, pos, pos + num_ins))
            num_left -= num_ins
            pos += num
    result = _cache_state(kv_cache)
    result["attn_output"] = kv_cache(**range_from_args(data, total - 1, total))
    return result


def _cases(names: List[str]):
    result = []
    for device, name, (prefill, warmup, window) in product(
        available_backends(), names, SCENARIOS
    ):
        if name.startswith("dense") and prefill + warmup + sum(window) >= CACHE_LENGTH:
            continue  # Dense cache cannot hold that many tokens
        for num_rollback in range(sum(window) + 1):
            result.append((device, name, prefill, warmup, window, num_rollback))
    return result


@pytest.mark.parametrize(
    "device, name, prefill, warmup, window, num_rollback",
    _cases(["dense-default", "lastrec-default"]),
)
def test_rollback_equals_shorter_insert(
    device, name, prefill, warmup, window, num_rollback
):
    seed = 31415927 + num_rollback
    kwargs = dict(
        name=name,
        device=device,
        dtype=torch.float32,
        prefill=prefill,
        warmup=warmup,
        window=window,
        num_rollback=num_rollback,
        seed=seed,
    )
    state_rb = _run_scenario(use_rollback=True, **kwargs)
    state_short = _run_scenario(use_rollback=False, **kwargs)
    for key in ("input_pos", "current_length", "token_positions", "next_position"):
        assert state_rb[key] == state_short[key], (key, state_rb[key], state_short[key])
    # Default buffers: Restored content must be exactly the same
    for key in ("keys", "values", "attn_output"):
        torch.testing.assert_close(state_rb[key], state_short[key], atol=0, rtol=0)


@pytest.mark.parametrize(
    "device, name, prefill, warmup, window, num_rollback",
    _cases(["dense-torch-quantized8", "lastrec-torch-quantized8"]),
)
def test_rollback_equals_shorter_insert_quantized(
    device, name, prefill, warmup, window, num_rollback
):
    seed = 27182818 + num_rollback
    kwargs = dict(
        name=name,
        device=device,
        dtype=torch.bfloat16,
        prefill=prefill,
        warmup=warmup,
        window=window,
        num_rollback=num_rollback,
        seed=seed,
    )
    state_rb = _run_scenario(use_rollback=True, **kwargs)
    state_short = _run_scenario(use_rollback=False, **kwargs)
    for key in ("input_pos", "current_length", "token_positions", "next_position"):
        assert state_rb[key] == state_short[key], (key, state_rb[key], state_short[key])
    # With a single cache, the de-quantized buffer stays associated with it
    # throughout, so restored content is exact. See
    # :func:`test_rollback_quantized_shared_dequant_buffer` for the case of
    # several layers.
    for key in ("keys", "values", "attn_output"):
        torch.testing.assert_close(state_rb[key], state_short[key], atol=0, rtol=0)


def _run_scenario_layers(
    name: str,
    prefill: int,
    warmup: int,
    window: List[int],
    num_rollback: int,
    use_rollback: bool,
    seed: int,
    n_layer: int = 3,
) -> List[Dict[str, Any]]:
    """
    Same as :func:`_run_scenario`, but for the KV caches of several layers,
    which share a de-quantized buffer if `name` uses quantization. Layers are
    called in turn, as in a forward pass of the model.

    """
    from keys_values.config import Config
    from keys_values.kvcache.factory import KVCacheFactory
    from keys_values.model import GPT

    params = _params(torch.bfloat16)
    config = Config(
        n_embd=params.n_head * params.head_size,
        n_head=params.n_head,
        n_query_groups=params.n_query_groups,
        n_layer=n_layer,
        vocab_size=64,
    )
    torch.random.manual_seed(0)
    kv_caches = KVCacheFactory.create(
        gpt_model=GPT(config),
        name=name,
        max_batch_size=params.max_batch_size,
        cache_length=params.cache_length,
        dtype=params.dtype,
    )
    torch.random.manual_seed(seed)
    total = prefill + warmup + sum(window) + 1
    data = [random_args_cache_forward(params, total, vocab_size=64) for _ in kv_caches]

    def forward_all(start: int, end: int) -> List[torch.Tensor]:
        return [
            cache(**range_from_args(d, start, end)) for cache, d in zip(kv_caches, data)
        ]

    forward_all(0, prefill)
    pos = prefill
    for _ in range(warmup):
        forward_all(pos, pos + 1)
        pos += 1
    if use_rollback:
        for cache in kv_caches:
            cache.mark_rollback_point()
        for num in window:
            forward_all(pos, pos + num)
            pos += num
        for cache in kv_caches:
            cache.rollback(num_rollback)
    else:
        num_left = sum(window) - num_rollback
        for num in window:
            num_ins = min(num, num_left)
            if num_ins > 0:
                forward_all(pos, pos + num_ins)
            num_left -= num_ins
            pos += num
    attn_outputs = forward_all(total - 1, total)
    results = []
    for cache, attn_output in zip(kv_caches, attn_outputs):
        state = _cache_state(cache)
        state["attn_output"] = attn_output
        results.append(state)
    return results


@pytest.mark.parametrize(
    "name, prefill, warmup, window, num_rollback, atol",
    [
        (name, prefill, warmup, window, num_rollback, atol)
        for (name, atol), (prefill, warmup, window) in product(
            [("lastrec-torch-quantized8", 0.0625), ("lastrec-torch-quantized4", 0.3)],
            SCENARIOS,
        )
        for num_rollback in range(sum(window) + 1)
    ],
)
def test_rollback_quantized_shared_dequant_buffer(
    name, prefill, warmup, window, num_rollback, atol
):
    """
    If several layers share a de-quantized buffer, cache content is quantized
    and de-quantized whenever the association changes. Restored slots are
    then re-quantized from de-quantized content. This round trip is not exact
    (it is not for writes in general), so we only require the deviation to
    be of the order of the quantization error. Slot bookkeeping must be
    exact.

    """
    kwargs = dict(
        name=name,
        prefill=prefill,
        warmup=warmup,
        window=window,
        num_rollback=num_rollback,
        seed=1000 + num_rollback,
    )
    states_rb = _run_scenario_layers(use_rollback=True, **kwargs)
    states_short = _run_scenario_layers(use_rollback=False, **kwargs)
    for state_rb, state_short in zip(states_rb, states_short):
        for key in ("input_pos", "current_length", "token_positions", "next_position"):
            assert state_rb[key] == state_short[key], (
                key,
                state_rb[key],
                state_short[key],
            )
        for key in ("keys", "values", "attn_output"):
            torch.testing.assert_close(
                state_rb[key].float(), state_short[key].float(), atol=atol, rtol=0
            )


@pytest.mark.parametrize("name", ["dense-default", "lastrec-default"])
def test_rollback_across_calls_and_repeated(name):
    """
    Speculative decoding marks a rollback point, inserts tokens over several
    calls, rolls back, and repeats. The cache must track the sequence exactly.

    """
    torch.random.manual_seed(1234)
    params = _params(torch.float32)
    total = 200 if name.startswith("lastrec") else CACHE_LENGTH
    data = random_args_cache_forward(params, total, vocab_size=64)
    kv_cache = create_kv_cache(name, params)
    reference = create_kv_cache(name, params)
    prefill = 4
    kv_cache(**range_from_args(data, 0, prefill))
    reference(**range_from_args(data, 0, prefill))
    pos = prefill
    gen = torch.Generator().manual_seed(5)
    while True:
        num_draft = int(torch.randint(1, 4, (1,), generator=gen))
        if pos + num_draft + 1 > total:
            break
        kv_cache.mark_rollback_point()
        kv_cache(**range_from_args(data, pos, pos + 1))
        for i in range(num_draft):
            p = pos + 1 + i
            kv_cache(**range_from_args(data, p, p + 1))
        num_keep = int(torch.randint(0, num_draft + 1, (1,), generator=gen))
        kv_cache.rollback(num_draft - num_keep)
        reference(**range_from_args(data, pos, pos + 1 + num_keep))
        pos += 1 + num_keep
        state = _cache_state(kv_cache)
        state_ref = _cache_state(reference)
        for key in ("input_pos", "current_length", "token_positions", "next_position"):
            assert state[key] == state_ref[key], (pos, key, state[key], state_ref[key])
        torch.testing.assert_close(state["keys"], state_ref["keys"], atol=0, rtol=0)
        torch.testing.assert_close(state["values"], state_ref["values"], atol=0, rtol=0)


def test_rollback_errors():
    torch.random.manual_seed(1)
    params = _params(torch.float32)
    data = random_args_cache_forward(params, 12, vocab_size=64)
    kv_cache = create_kv_cache("lastrec-default", params)
    assert kv_cache.supports_rollback
    with pytest.raises(IndexError):
        kv_cache.mark_rollback_point()  # Not prefilled yet
    kv_cache(**range_from_args(data, 0, 6))
    with pytest.raises(IndexError):
        kv_cache.rollback(0)  # No rollback point
    kv_cache.mark_rollback_point()
    kv_cache(**range_from_args(data, 6, 8))
    with pytest.raises(ValueError):
        kv_cache.rollback(3)  # Only 2 tokens since rollback point
    assert kv_cache.rollback_point_is_marked
    kv_cache.rollback(0)  # Just clears the rollback point
    assert kv_cache.input_pos == 8
    assert not kv_cache.rollback_point_is_marked
    # Prefill and reset clear the rollback point
    kv_cache.mark_rollback_point()
    kv_cache(**range_from_args(data, 8, 9))
    kv_cache.reset()
    assert not kv_cache.rollback_point_is_marked
    kv_cache(**range_from_args(data, 0, 6))
    kv_cache.mark_rollback_point()
    kv_cache.deallocate_buffers()
    assert not kv_cache.rollback_point_is_marked
    # Replay logging is not supported together with rollback
    kv_cache = create_kv_cache("lastrec-default", params)
    kv_cache.switch_replay_logging(True)
    kv_cache(**range_from_args(data, 0, 6))
    with pytest.raises(ValueError):
        kv_cache.mark_rollback_point()


@pytest.mark.parametrize(
    "name", ["h2o-default", "h2o-vlen-default", "h2o-orig-default"]
)
def test_rollback_not_supported(name):
    torch.random.manual_seed(1)
    params = _params(torch.float32)
    data = random_args_cache_forward(params, 8, vocab_size=64)
    kv_cache = create_kv_cache(name, params)
    assert not kv_cache.supports_rollback
    kv_cache(**range_from_args(data, 0, 6))
    with pytest.raises(NotImplementedError):
        kv_cache.mark_rollback_point()
