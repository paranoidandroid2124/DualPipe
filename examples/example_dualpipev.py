from typing import List, Optional, Callable, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from dualpipe import DualPipeV, set_p2p_tensor_shapes, set_p2p_tensor_dtype
from dualpipe.utils import WeightGradStore, run_backward


class LinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = F.linear(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        if weight.grad is None:
            weight.grad = torch.zeros_like(weight)

        def grad_weight_fn():
            weight.grad += grad_output.flatten(0, -2).T @ input.flatten(0, -2)

        if WeightGradStore.enabled:
            WeightGradStore.put(grad_weight_fn)
        else:
            grad_weight_fn()
        grad_input = grad_output @ weight
        return grad_input, None


class MyLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return LinearFunc.apply(input, self.weight)


class PipelineStage(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear1 = MyLinear(hidden_size, hidden_size * 4, bias=False)
        self.linear2 = MyLinear(hidden_size * 4, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        return x

    @classmethod
    def overlapped_forward_backward(
        cls,
        module0: "PipelineStage",
        inputs0: List[torch.Tensor],
        criterion0: Optional[Callable],
        labels0: Optional[List[torch.Tensor]],
        module1: "PipelineStage",
        loss1: Optional[torch.Tensor],
        outputs1: Optional[List[torch.Tensor]],
        output_grads1: Optional[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        You should implement custom forward-backward overlap strategy.
        The code below is just an example.
        """
        outputs0 = module0(*inputs0)
        outputs0 = [outputs0] if isinstance(outputs0, torch.Tensor) else outputs0
        if criterion0 is not None:
            loss0 = criterion0(*outputs0, *labels0)
        else:
            loss0 = None

        if loss1 is not None:
            loss1.backward()
            loss1.detach_()
        else:
            run_backward(outputs1, output_grads1)

        return outputs0, loss0


def criterion(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(output, target).clone()


def ref_step(x, l, model, chunks):
    ys, losses = [], []
    for micro_x, micro_l in zip(x.chunk(chunks), l.chunk(chunks)):
        micro_y = model(micro_x)
        loss = criterion(micro_y, micro_l)
        loss.backward()
        ys.append(micro_y)
        losses.append(loss)
    y = torch.cat(ys, 0)
    loss = torch.stack(losses)
    return loss, y


def cal_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    cos_diff = 1 - 2 * (x * y).sum().item() / (x * x + y * y).sum().item()
    return cos_diff


def main(rank, pp_size):
    is_first_rank = rank == 0
    dist.init_process_group(backend='nccl', init_method="env://", world_size=pp_size, rank=rank)
    torch.cuda.set_device(rank)
    torch.set_default_device(f"cuda:{rank}")
    torch.manual_seed(233)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    num_chunks = 20
    micro_batch_size = 3
    seq_len = 256
    hidden_size = 512
    if is_first_rank:
        print(f"{pp_size=}, {num_chunks=}, {seq_len=}, {hidden_size=}", flush=True)
    set_p2p_tensor_shapes([(micro_batch_size, seq_len, hidden_size)])
    set_p2p_tensor_dtype(torch.float32)

    # Create a model and partition it for each process
    full_modules = nn.Sequential(*[PipelineStage(hidden_size) for _ in range(pp_size * 2)])

    # Full inputs
    x = torch.randn(num_chunks * micro_batch_size, seq_len, hidden_size)
    l = torch.randn(num_chunks * micro_batch_size, seq_len, hidden_size)

    # Reference step
    loss_ref, output_ref = ref_step(x, l, full_modules, num_chunks)

    # DualPipeV
    local_full_modules = nn.Sequential(full_modules[rank], full_modules[pp_size * 2 - 1 - rank])
    local_modules = nn.Sequential(PipelineStage(hidden_size), PipelineStage(hidden_size))
    local_modules[0].load_state_dict(local_full_modules[0].state_dict())
    local_modules[1].load_state_dict(local_full_modules[1].state_dict())
    dualpipev_model = DualPipeV(local_modules)

    # DualPipeV inputs
    if not is_first_rank:
        x = None
        l = None

    # Training step
    loss, outputs = dualpipev_model.step(x, num_chunks=num_chunks, criterion=criterion, labels=(l,), return_outputs=False)

    # Check loss
    if is_first_rank:
        assert torch.equal(loss, loss_ref)
    else:
        assert loss is None
    assert outputs is None

    # Check grads
    for ((n, p), p_ref) in zip(local_modules.named_parameters(), local_full_modules.parameters()):
        assert cal_diff(p.grad, p_ref.grad) < 1e-13
    dualpipev_model.zero_grad()

    # Inference step
    with torch.no_grad():
        loss, outputs = dualpipev_model.step(x, num_chunks=num_chunks, criterion=criterion, labels=(l,), return_outputs=True)

    # Check loss and outputs
    if is_first_rank:
        assert torch.equal(loss, loss_ref)
        assert torch.equal(outputs, output_ref)
    else:
        assert loss is None
        assert outputs is None


def test_dualpipev(ngpus):
    torch.multiprocessing.spawn(main, args=(ngpus, ), nprocs=ngpus, daemon=True)


if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    for ngpus in range(num_gpus, 0, -1):
        test_dualpipev(ngpus)
