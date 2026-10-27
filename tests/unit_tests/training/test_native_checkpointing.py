# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest import mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import load
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.training.arguments import parse_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.ckpt import native
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    init_basic_mock_args,
    init_checkpointing_mock_args,
    setup_model_and_optimizer,
)
from tests.unit_tests.test_utilities import Utils

pytest.importorskip("nvrx_checkpointing")


def _state_dict(step, extra=False):
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    dp_rank = parallel_state.get_data_parallel_rank()
    weight = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
    model = {
        "weight": ShardedTensor.from_rank_offsets(
            "model.weight", weight + 100 * tp_rank + step, (0, tp_rank, tp_size), replica_id=dp_rank
        )
    }
    if extra:
        model["bias"] = ShardedTensor.from_rank_offsets(
            "model.bias", torch.full((4,), float(step), device="cuda"), replica_id=Utils.rank
        )
    return {
        "model": model,
        "rng_state": ShardedObject(
            "rng_state", list(range(step)), (Utils.world_size,), (Utils.rank,)
        ),
        "iteration": step,
    }


def _load(checkpoint_dir):
    template = _state_dict(0)
    template["model"]["weight"].data = torch.empty(4, 4, device="cuda")
    return load(template, checkpoint_dir)


class TestNativeCheckpointing:
    def setup_method(self, method):
        Utils.initialize_model_parallel(2, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_saves_store_the_values_of_each_save(self, tmp_path_dist_ckpt):
        with TempNamedDir(tmp_path_dist_ckpt / "native_values") as checkpoint_dir:
            for step in (1, 2):
                native.save(_state_dict(step), checkpoint_dir / str(step))()
            for step in (1, 2):
                loaded = _load(checkpoint_dir / str(step))
                expected = _state_dict(step)
                assert torch.equal(loaded["model"]["weight"], expected["model"]["weight"].data)
                assert loaded["rng_state"] == list(range(step))
                assert loaded["iteration"] == step

    def test_a_registered_state_is_saved_without_registering_again(self, tmp_path_dist_ckpt):
        with TempNamedDir(tmp_path_dist_ckpt / "native_registered") as checkpoint_dir:
            native.register(_state_dict(1))
            session_id = native._session_id
            native.save(_state_dict(2), checkpoint_dir)()
            assert native._session_id == session_id
            loaded = _load(checkpoint_dir)
            assert torch.equal(loaded["model"]["weight"], _state_dict(2)["model"]["weight"].data)
            assert loaded["rng_state"] == [0, 1] and loaded["iteration"] == 2

    def test_a_state_with_another_structure_is_registered_anew(self, tmp_path_dist_ckpt):
        with TempNamedDir(tmp_path_dist_ckpt / "native_restructured") as checkpoint_dir:
            native.save(_state_dict(1), checkpoint_dir / "1")()
            session_id = native._session_id
            native.save(_state_dict(2, extra=True), checkpoint_dir / "2")()
            assert native._session_id != session_id
            template = _state_dict(0, extra=True)
            loaded = load(template, checkpoint_dir / "2")
            assert torch.equal(loaded["model"]["bias"], torch.full((4,), 2.0, device="cuda"))

    def test_lists_of_sharded_tensors_and_lists_are_saved(self, tmp_path_dist_ckpt):
        # Like the fp32 params of a mixed-precision optimizer, where SwiGLU builds [w, v] lists.
        def state(value):
            halves = [
                ShardedTensor.from_rank_offsets(
                    "fc1",
                    torch.full((2, 4), value + half, device="cuda"),
                    (0, half, 2),
                    replica_id=Utils.rank,
                )
                for half in range(2)
            ]
            proj = ShardedTensor.from_rank_offsets(
                "proj", torch.full((4,), value, device="cuda"), replica_id=Utils.rank
            )
            return {"optimizer": {0: {"fp32_from_fp16_params": [[proj, halves]]}}}

        with TempNamedDir(tmp_path_dist_ckpt / "native_nested_lists") as checkpoint_dir:
            native.save(state(1.0), checkpoint_dir)()
            loaded = load(state(0.0), checkpoint_dir)["optimizer"][0]["fp32_from_fp16_params"]
            assert torch.equal(loaded[0][0], torch.full((4,), 1.0, device="cuda"))
            assert torch.equal(loaded[0][1][1], torch.full((2, 4), 2.0, device="cuda"))

    def test_quantized_tensors_that_are_not_kept_packed_are_saved_dequantized(
        self, tmp_path_dist_ckpt
    ):
        blockwise = pytest.importorskip("transformer_engine.pytorch.tensor.float8_blockwise_tensor")
        quantizer = blockwise.Float8BlockQuantizer(
            fp8_dtype=blockwise.tex.DType.kFloat8E4M3, rowwise=True, columnwise=False
        )
        generator = torch.Generator(device="cuda").manual_seed(0)
        values = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
        weight = quantizer(values)

        def state(data):
            return {"w": ShardedTensor.from_rank_offsets("w", data, replica_id=Utils.rank)}

        with TempNamedDir(tmp_path_dist_ckpt / "native_blockwise") as checkpoint_dir:
            native.save(state(weight), checkpoint_dir)()
            loaded = load(state(torch.empty_like(weight.dequantize())), checkpoint_dir)["w"]
            assert torch.equal(loaded, weight.dequantize())

    def test_the_state_registered_after_a_load_is_the_state_of_the_next_save(
        self, tmp_path_dist_ckpt, run_config
    ):
        model, optimizer = setup_model_and_optimizer(1, 2, 1)
        args = parse_args(ignore_unknown_args=True)
        args.save_tokenizer_assets = False
        with (
            TempNamedDir(tmp_path_dist_ckpt / "native_after_load", sync=True) as checkpoint_dir,
            mock.patch("megatron.training.checkpointing.get_args", new=lambda: args),
            mock.patch("megatron.training.ckpt.native.get_args", new=lambda: args),
        ):
            init_basic_mock_args(args, 2, 1)
            init_checkpointing_mock_args(args, checkpoint_dir)
            save_checkpoint(1, model, optimizer, None, 0)
            native.spin_down()
            iteration, flops = load_checkpoint(model, optimizer, None, restore_training_state=True)
            native.register_after_load(model, optimizer, None, iteration, flops)
            session_id = native._session_id
            save_checkpoint(2, model, optimizer, None, 0)
            assert session_id is not None and native._session_id == session_id
