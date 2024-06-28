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
from distributed_shampoo.shampoo_types import AdamGraftingConfig

logger = logging.getLogger(__name__)


@dataclass
class FairseqShampooConfig(FairseqDataclass):
    adam_betas: str = field(
        default="(0.9, 0.999)", metadata={"help": "betas for Adam optimizer"}
    )
    adam_eps: float = field(
        default=1e-8, metadata={"help": "epsilon for Adam optimizer"}
    )
    shampoo_update_freq: int = field(
        default=16, metadata={"help": "update shampoo matrix every N steps"}
    )
    shampoo_max_preconditioner_dim: int = field(
        default=1024, metadata={"help": "max dimension of preconditioner"}
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
            epsilon=cfg.adam_eps,
            weight_decay=cfg.weight_decay,
            precondition_frequency=cfg.shampoo_update_freq,
            max_preconditioner_dim=cfg.shampoo_max_preconditioner_dim,
            grafting_config=AdamGraftingConfig(
                beta2=eval(cfg.adam_betas)[1],
                epsilon=cfg.adam_eps,
            ),
        )
        # self._optimizer = Shampoo(params, **self.optimizer_config)

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

class Shampoo(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        super(Shampoo, self).__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                order = grad.ndimension()
                original_size = grad.size()
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                    for dim_id, dim in enumerate(grad.size()):
                        state["precond_{}".format(dim_id)] = grad.new(dim, dim).zero_()
                        state["inv_precond_{dim_id}".format(dim_id=dim_id)] = grad.new(dim, dim).zero_()
                else:
                    state["exp_avg"] = state["exp_avg"].to(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(p_data_fp32)
                    for dim_id, dim in enumerate(grad.size()):
                        state["precond_{}".format(dim_id)] = state["precond_{}".format(dim_id)].to(p_data_fp32)
                        state["inv_precond_{dim_id}".format(dim_id=dim_id)] = state["inv_precond_{dim_id}".format(dim_id=dim_id)].to(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(
                        p_data_fp32, alpha=-group["weight_decay"] * group["lr"]
                    )
                    
                grad = exp_avg / denom
                for dim_id, dim in enumerate(grad.size()):
                    precond = state["precond_{}".format(dim_id)]
                    inv_precond = state["inv_precond_{}".format(dim_id)]

                    # mat_{dim_id}(grad)
                    grad = grad.transpose_(0, dim_id).contiguous()
                    transposed_size = grad.size()
                    grad = grad.view(dim, -1)

                    grad_t = grad.t()
                    precond.add_(grad @ grad_t)
                    if state["step"] % group["shampoo_update_freq"] == 0:
                        inv_precond.copy_(_matrix_power(precond, -1 / order))

                    if dim_id == order - 1:
                        # finally
                        grad = grad_t @ inv_precond
                        # grad: (-1, last_dim)
                        grad = grad.view(original_size)
                    else:
                        # if not final
                        grad = inv_precond @ grad
                        # grad (dim, -1)
                        grad = grad.view(transposed_size)

                p_data_fp32.add(grad, -step_size)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss

def _matrix_power(matrix: torch.Tensor, power: float) -> torch.Tensor:
    # use CPU for svd for speed up
    device = matrix.device
    matrix = matrix.cpu()
    u, s, v = torch.svd(matrix)
    return (u @ s.pow_(power).diag() @ v.t()).to(device)
