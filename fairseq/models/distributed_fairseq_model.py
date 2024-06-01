# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import signal
import threading

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from fairseq.distributed import (
    DistributedTimeoutWrapper,
    LegacyDistributedDataParallel,
    ModuleProxyWrapper,
    TPUDistributedDataParallel,
)


logger = logging.getLogger(__name__)


_GOSSIP_DISABLED = False
try:
    import gossip
except ImportError:
    _GOSSIP_DISABLED = True


def fp32_compress_hook(
    process_group: dist.ProcessGroup,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """
    Compress by casting ``GradBucket`` to ``torch.float32`` divided by process group size.

    This DDP communication hook implements a simple gradient compression
    approach that casts ``GradBucket`` tensor to half-precision floating-point format (``torch.float16``)
    and then divides it by the process group size.
    It allreduces those ``float32`` gradient tensors. Once compressed gradient
    tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``).

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, fp16_compress_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    buffer = (
        cast(Tuple[torch.Tensor, ...], bucket)[0]
        if isinstance(bucket, tuple)
        else bucket.buffer()
    )
    compressed_tensor = buffer.to(torch.float32).div_(world_size)

    def decompress(fut):
        decompressed_tensor = buffer
        # Decompress in place to reduce the peak memory.
        # See: https://github.com/pytorch/pytorch/issues/45968
        value = fut if isinstance(fut, torch.Tensor) else fut.value()[0]
        decompressed_tensor.copy_(value)
        return decompressed_tensor

    if torch._utils.is_compiling():
        grad = dist._functional_collectives.all_reduce(
            compressed_tensor, "sum", group_to_use
        )
        return decompress(grad)
    else:
        fut = dist.all_reduce(
            compressed_tensor, group=group_to_use, async_op=True
        ).get_future()
        return fut.then(decompress)


def DistributedFairseqModel(args, model, process_group, device):
    """
    Wrap a *model* to support distributed data parallel training.

    This is similar to the built-in DistributedDataParallel, but allows
    additional configuration of the DistributedDataParallel class to
    use, and also provides easier access to the wrapped model by
    forwarding requests for missing attributes to the wrapped model.

    Args:
        args (argparse.Namespace): fairseq args
        model (BaseFairseqModel): model to wrap
        process_group: the c10d process group to be used for distributed data
            parallel all-reduction.
        device: device to move model to
    """
    assert isinstance(model, nn.Module)
    if args.tpu:
        wrapped_model = TPUDistributedDataParallel(
            module=model.to(device),
            process_group=process_group,
        )
        # forward missing getattr and state_dict/load_state_dict to orig model
        wrapped_model = ModuleProxyWrapper(wrapped_model)
    elif args.ddp_backend in {"c10d", "pytorch_ddp"}:
        wrapped_model = DistributedDataParallel(
            module=model.to(device),
            device_ids=[args.device_id],
            output_device=args.device_id,
            broadcast_buffers=args.broadcast_buffers,
            bucket_cap_mb=args.bucket_cap_mb,
            process_group=process_group,
            find_unused_parameters=args.find_unused_parameters,
        )
        from fairseq.distributed.utils import get_data_parallel_group
        wrapped_model.register_comm_hook(get_data_parallel_group(), fp32_compress_hook)
        # forward missing getattr and state_dict/load_state_dict to orig model
        wrapped_model = ModuleProxyWrapper(wrapped_model)
    elif args.ddp_backend in {"no_c10d", "legacy_ddp"}:
        wrapped_model = LegacyDistributedDataParallel(
            module=model.to(device),
            buffer_size=2 ** 28,
            process_group=process_group,
        )
        # forward missing getattr and state_dict/load_state_dict to orig model
        wrapped_model = ModuleProxyWrapper(wrapped_model)
    elif args.ddp_backend == "slow_mo":
        if _GOSSIP_DISABLED:
            raise ImportError(
                "Cannot find gossip library. Please install from: "
                "github.com/facebookresearch/stochastic_gradient_push"
            )

        # The values of slowmo_momentum below were obtained by tuning on the
        # En-De 16 dataset by training the transformer_wmt_en_de_large model
        if args.slowmo_momentum is None:
            if args.distributed_world_size <= 16:
                args.slowmo_momentum = 0.0
            elif args.distributed_world_size <= 32:
                args.slowmo_momentum = 0.2
            elif args.distributed_world_size <= 64:
                args.slowmo_momentum = 0.5
            else:
                args.slowmo_momentum = 0.6

        wrapped_model = gossip.GossipDataParallel(
            module=model.to(device),
            device_ids=[args.device_id],
            output_device=args.device_id,
            broadcast_buffers=args.broadcast_buffers,
            nprocs_per_node=args.nprocs_per_node,
            slowmo_momentum=args.slowmo_momentum,
            localsgd=(args.slowmo_algorithm == "LocalSGD"),
            localsgd_frequency=args.localsgd_frequency,
        )
        # forward missing getattr and state_dict/load_state_dict to orig model
        wrapped_model = ModuleProxyWrapper(wrapped_model)
    elif args.ddp_backend == "fully_sharded":
        try:
            from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
        except ImportError:
            raise ImportError(
                "Cannot find FullyShardedDataParallel. "
                "Please install fairscale with: pip install fairscale"
            )
        assert isinstance(model, FSDP), "expected model to already be wrapped in FSDP"
        wrapped_model = model
        if args.memory_efficient_fp16:
            wrapped_model = wrapped_model.half()
        if not args.cpu_offload:
            wrapped_model = wrapped_model.to(device=device)
    else:
        raise ValueError("Unknown --ddp-backend: " + args.ddp_backend)

    # kill hung distributed jobs after a timeout
    if getattr(args, "heartbeat_timeout", -1) > 0:
        wrapped_model = DistributedTimeoutWrapper(
            wrapped_model, timeout=getattr(args, "heartbeat_timeout", -1)
        )

    return wrapped_model
