from typing import List, Optional, Callable, Tuple
import os
import sys # For flushing print statements

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from dualpipe import DualPipe, set_p2p_tensor_shapes, set_p2p_tensor_dtype
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
            # Ensure gradients are accumulated on the correct device
            if grad_output.device != input.device or grad_output.device != weight.device:
                # This might indicate an issue if devices are not consistent
                # For this example, we assume they should be on the same device as weight
                grad_output_flat = grad_output.flatten(0, -2).to(weight.device)
                input_flat = input.flatten(0, -2).to(weight.device)
                weight.grad += grad_output_flat.T @ input_flat
            else:
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
        # Ensure inputs are on the correct device for module0
        device0 = next(module0.parameters()).device
        inputs0_on_device = [inp.to(device0) for inp in inputs0]

        outputs0 = module0(*inputs0_on_device)
        outputs0 = [outputs0] if isinstance(outputs0, torch.Tensor) else outputs0
        if criterion0 is not None:
            # Ensure labels are on the correct device for criterion0
            labels0_on_device = [lbl.to(device0) for lbl in labels0] if labels0 else []
            loss0 = criterion0(*outputs0, *labels0_on_device)
        else:
            loss0 = None

        if loss1 is not None:
            loss1.backward()
            loss1.detach_()
        elif outputs1 is not None and output_grads1 is not None:
            # Ensure outputs1 and output_grads1 are on the correct device for module1's backward pass
            device1 = next(module1.parameters()).device
            outputs1_on_device = [out.to(device1) for out in outputs1]
            output_grads1_on_device = [grad.to(device1) if grad is not None else None for grad in output_grads1]
            
            # Filter out None gradients before calling run_backward
            valid_outputs = []
            valid_grads = []
            for o, g in zip(outputs1_on_device, output_grads1_on_device):
                if g is not None: # Ensure that the gradient is not None
                    valid_outputs.append(o)
                    valid_grads.append(g)
            
            if valid_outputs: # Proceed only if there are tensors requiring gradients
                 # Convert lists to tuples before passing to run_backward
                 run_backward(tuple(valid_outputs), tuple(valid_grads))


        return outputs0, loss0


def criterion(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Ensure target is on the same device as output for loss calculation
    return F.mse_loss(output, target.to(output.device)).clone()


def ref_step(x, l, model, chunks):
    ys, losses = [], []
    # Ensure model is on the correct device (e.g., the device of x)
    # For a single process reference, this might be simpler.
    # If model is already on a GPU, x and l should also be.
    model_device = next(model.parameters()).device
    x = x.to(model_device)
    l = l.to(model_device)

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
    # Ensure x and y are on the same device for comparison
    y = y.to(x.device)
    # Add a small epsilon to the denominator to prevent division by zero if norms are zero
    eps = 1e-12
    cos_diff = 1 - 2 * (x * y).sum().item() / ((x * x).sum() + (y * y).sum() + eps).item() 
    return cos_diff


def main(rank, pp_size):
    # Set environment variables for torch.distributed
    # These are crucial for the 'env://' initialization method.
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # Changed to 127.0.0.1
    os.environ['MASTER_PORT'] = '29501'      # Using port 29501
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(pp_size)

    # Print environment variables for debugging
    print(f"Rank {rank}: MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}, RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}", flush=True)
    sys.stdout.flush() # Ensure print statements are flushed immediately

    is_first_rank = rank == 0
    is_last_rank = rank == pp_size - 1

    # Initialize the process group.
    # backend='nccl' is standard for NVIDIA GPUs.
    dist.init_process_group(backend='nccl', init_method="env://", world_size=pp_size, rank=rank)
    
    # Set the current CUDA device for this process.
    torch.cuda.set_device(rank)
    
    torch.manual_seed(233 + rank) # Seed for reproducibility, add rank for slight variation if needed by tests
    # This environment variable is for cuBLAS workspace configuration, related to determinism.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


    num_chunks = 20
    micro_batch_size = 3
    seq_len = 256
    hidden_size = 512

    current_device = torch.device(f"cuda:{rank}")

    if is_first_rank:
        print(f"Rank {rank}: Initialization complete. Using MASTER_PORT={os.environ['MASTER_PORT']}", flush=True)
        print(f"Rank {rank}: {pp_size=}, {num_chunks=}, {seq_len=}, {hidden_size=}", flush=True)
        sys.stdout.flush()
    
    # Set tensor shapes and dtype for P2P communication.
    # These must be consistent across all ranks involved in P2P.
    set_p2p_tensor_shapes([(micro_batch_size, seq_len, hidden_size)])
    set_p2p_tensor_dtype(torch.float32)

    # Create a full model (conceptually) and then partition it.
    full_modules_ref = nn.Sequential(*[PipelineStage(hidden_size) for _ in range(pp_size)]).to(current_device)


    # Full inputs for the reference step, created on the current device.
    full_x = torch.randn(num_chunks * micro_batch_size, seq_len, hidden_size, device=current_device)
    full_l = torch.randn(num_chunks * micro_batch_size, seq_len, hidden_size, device=current_device)


    # Reference step:
    loss_ref_val_chunks = [None, None]
    output_ref_val_chunks = [None, None]

    if rank == 0: 
        loss_ref_full, output_ref_full = ref_step(full_x, full_l, full_modules_ref, num_chunks)
        loss_ref_val_chunks = loss_ref_full.chunk(2)
        output_ref_val_chunks = output_ref_full.chunk(2)
    
    # DualPipe setup
    local_stage1 = PipelineStage(hidden_size).to(current_device)
    local_stage2 = PipelineStage(hidden_size).to(current_device)
    
    local_stage1.load_state_dict(full_modules_ref[rank].state_dict())
    local_stage2.load_state_dict(full_modules_ref[pp_size - 1 - rank].state_dict())
    
    local_modules_dp = nn.ModuleList([local_stage1, local_stage2]) 
    dualpipe_model = DualPipe(tuple(local_modules_dp), process_group=dist.group.WORLD, rank_mapping=list(range(pp_size)))


    # DualPipe inputs:
    x_dp, l_dp = None, None
    if is_first_rank:
        x_dp = full_x.chunk(2)[0].to(current_device)
        l_dp = full_l.chunk(2)[1].to(current_device) 
    elif is_last_rank:
        x_dp = full_x.chunk(2)[1].to(current_device)
        l_dp = full_l.chunk(2)[0].to(current_device) 

    # Training step with DualPipe
    loss_dp, outputs_dp = dualpipe_model.step(x_dp, num_chunks=num_chunks, criterion=criterion, labels=(l_dp,) if l_dp is not None else [], return_outputs=False)
    
    ref_loss_for_current_rank = None
    if rank == 0: 
        ref_loss_for_current_rank = loss_ref_val_chunks[1]
    elif rank == pp_size -1: 
        ref_loss_for_current_rank = loss_ref_val_chunks[0]

    if loss_dp is not None:
        if ref_loss_for_current_rank is not None:
             assert torch.allclose(loss_dp, ref_loss_for_current_rank.to(current_device), atol=1e-5), f"Rank {rank} loss mismatch"
        else:
            assert False, f"Rank {rank} computed loss_dp but no reference was assigned."
            
    elif ref_loss_for_current_rank is not None:
        assert False, f"Rank {rank} loss_dp is None but a reference loss was expected."
        
    assert outputs_dp is None 

    # Gradient Check
    for i_module, module_in_dp in enumerate(local_modules_dp):
        for name, param in module_in_dp.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Rank {rank}, Module {i_module}, Param {name} grad is None"

    if rank == 0:
        print("Gradient check: Ensured gradients are not None for trainable parameters.", flush=True)
        sys.stdout.flush()

    dualpipe_model.zero_grad(set_to_none=True) 


    # Inference step
    with torch.no_grad():
        loss_dp_inf, outputs_dp_inf = dualpipe_model.step(x_dp, num_chunks=num_chunks, criterion=criterion, labels=(l_dp,) if l_dp is not None else [], return_outputs=True)

    ref_loss_inf_for_current_rank = None
    ref_output_inf_for_current_rank = None

    if rank == 0: 
        ref_loss_inf_for_current_rank = loss_ref_val_chunks[1]
        ref_output_inf_for_current_rank = output_ref_val_chunks[1]
    elif rank == pp_size -1: 
        ref_loss_inf_for_current_rank = loss_ref_val_chunks[0]
        ref_output_inf_for_current_rank = output_ref_val_chunks[0]

    if loss_dp_inf is not None:
        if ref_loss_inf_for_current_rank is not None:
            assert torch.allclose(loss_dp_inf, ref_loss_inf_for_current_rank.to(current_device), atol=1e-5), f"Rank {rank} inference loss mismatch"
    elif ref_loss_inf_for_current_rank is not None:
         assert False, f"Rank {rank} inference loss_dp_inf is None but a reference loss was expected."

    if outputs_dp_inf is not None:
        if ref_output_inf_for_current_rank is not None:
            assert torch.allclose(outputs_dp_inf, ref_output_inf_for_current_rank.to(current_device), atol=1e-5), f"Rank {rank} inference output mismatch"
    elif ref_output_inf_for_current_rank is not None:
        assert False, f"Rank {rank} inference outputs_dp_inf is None but a reference output was expected."
        
    # Clean up the process group
    dist.destroy_process_group()
    if rank == 0:
        print(f"Rank {rank} finished successfully for pp_size={pp_size}.", flush=True)
        sys.stdout.flush()


def test_dualpipe(ngpus):
    if ngpus < 2 : 
        print(f"Skipping test_dualpipe for ngpus={ngpus}, requires at least 2 GPUs for meaningful P2P.", flush=True)
        sys.stdout.flush()
        return
    
    torch.multiprocessing.spawn(main, args=(ngpus, ), nprocs=ngpus, daemon=True)


if __name__ == "__main__":
    available_gpus = torch.cuda.device_count()
    
    if available_gpus == 0:
        print("No CUDA GPUs found. This example requires GPUs.", flush=True)
    else:
        start_gpus = available_gpus
        if available_gpus % 2 != 0 and available_gpus > 1 : 
            start_gpus = available_gpus -1
        elif available_gpus == 1: 
            start_gpus = 0 

        if start_gpus >=2: 
            for ngpus_to_test in range(start_gpus, 1, -2): 
                print(f"--- Testing DualPipe with {ngpus_to_test} GPUs ---", flush=True)
                sys.stdout.flush()
                try:
                    test_dualpipe(ngpus_to_test)
                    print(f"--- Test with {ngpus_to_test} GPUs PASSED ---", flush=True)
                except Exception as e:
                    print(f"--- Test with {ngpus_to_test} GPUs FAILED: {e} ---", flush=True)
                finally:
                    sys.stdout.flush() # Ensure all output is flushed
        elif available_gpus > 0 : 
             print(f"Skipping DualPipe test as it requires at least 2 GPUs for this example's P2P setup, found {available_gpus}.", flush=True)
        else: 
             print("No GPUs to test with (re-check after available_gpus logic).", flush=True)
    sys.stdout.flush() # Final flush

