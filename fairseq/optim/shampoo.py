# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import List

import torch
import torch.distributed as dist
import torch.optim
from fairseq.dataclass import FairseqDataclass
from fairseq.optim import FairseqOptimizer, register_optimizer
from omegaconf import II, DictConfig

from distributed_shampoo.distributed_shampoo import DistributedShampoo
from distributed_shampoo.shampoo_types import AdamGraftingConfig, DDPShampooConfig, CommunicationDType

logger = logging.getLogger(__name__)


@dataclass
class FairseqShampooConfig(FairseqDataclass):
    adam_betas: str = field(
        default="(0.9, 0.999)", metadata={"help": "betas for Adam optimizer"}
    )
    adam_eps: float = field(
        default=1e-8, metadata={"help": "epsilon for Adam optimizer"}
    )
    shampoo_eps: float = field(
        default=1e-14, metadata={"help": "epsilon for Shampoo optimizer"}
    )
    optim_update_freq: int = field(
        default=1, metadata={"help": "update shampoo matrix every N steps"}
    )
    optim_max_preconditioner_dim: int = field(
        default=8192, metadata={"help": "max dimension of preconditioner"}
    )
    shampoo_num_trainers_per_group: int = field(
        default=1, metadata={"help": "number of GPUs per distributed process group for distributed computation/memory"}
    )
    weight_decay: float = field(default=0.0, metadata={"help": "weight decay"})
    # TODO common vars below in parent
    tpu: bool = II("common.tpu")
    bf16: bool = II("common.bf16")
    lr: List[float] = II("optimization.lr")


@register_optimizer("shampoo", dataclass=FairseqShampooConfig)
class FairseqShampoo(FairseqOptimizer):
    def __init__(self, cfg: DictConfig, params):
        super().__init__(cfg)
        self.adam_betas = eval(cfg.adam_betas)
        self._optimizer = DistributedShampoo(
            params,
            lr=cfg.lr[0],
            betas=eval(cfg.adam_betas),
            epsilon=cfg.shampoo_eps,
            weight_decay=cfg.weight_decay,
            precondition_frequency=cfg.optim_update_freq,
            max_preconditioner_dim=cfg.optim_max_preconditioner_dim,
            grafting_config=AdamGraftingConfig(
                beta2=eval(cfg.adam_betas)[1],
                epsilon=cfg.adam_eps,
            ), 
            distributed_config=DDPShampooConfig(
                communication_dtype=CommunicationDType.FP32,
                num_trainers_per_group=cfg.shampoo_num_trainers_per_group,
                communicate_params=False,
            ),
        )

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.cfg.lr[0]
            if isinstance(self.cfg.lr, Collection)
            else self.cfg.lr,
            "betas": eval(self.cfg.adam_betas),
            "eps": self.cfg.adam_eps,
            "weight_decay": self.cfg.weight_decay,
        }

    def average_params(self):
        """average Params is only used during BMUF distributed training."""
        state_dict = self.optimizer.state_dict()
        total_gpus = float(dist.get_world_size())

        for _, value in state_dict["state"].items():
            value["exp_avg"] /= total_gpus
            value["exp_avg_sq"] /= total_gpus
            dist.all_reduce(value["exp_avg"], op=dist.ReduceOp.SUM)
            dist.all_reduce(value["exp_avg_sq"], op=dist.ReduceOp.SUM)
            
    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return True

    def state_dict(self):
        return self._optimizer.distributed_state_dict()

    def load_state_dict(self, state_dict):
        self._optimizer.load_distributed_state_dict(state_dict)
