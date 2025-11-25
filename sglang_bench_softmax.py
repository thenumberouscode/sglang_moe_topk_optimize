import itertools
import os

import pytest
import torch
import triton
from softmax_extension import topk_softmax


VLLM_AVAILABLE = False
# CI environment detection
IS_DIFF = (
    os.getenv("IS_DIFF", "0").lower() == "1"
)



def sglang_topk_softmax(gating_output, topk, optimize=False):
    num_tokens, num_experts = gating_output.shape

    topk_weights = torch.empty(
        (num_tokens, topk), device=gating_output.device, dtype=torch.float32
    )
    topk_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=gating_output.device
    )
    topk_softmax(
        topk_weights,
        topk_indices,
        gating_output,
        optimize,
        False,
        0,
        None
    )
    
    #  (arg0: torch.Tensor, arg1: torch.Tensor, arg2: torch.Tensor, arg3: bool, arg4: float, arg5: Optional[torch.Tensor]) -> None

    return topk_weights, topk_indices


def calculate_diff(num_tokens, num_experts, topk):
    gating_output = torch.randn(
        (num_tokens, num_experts), device="cuda", dtype=torch.float32
    )
    weights_vllm, indices_vllm = vllm_topk_softmax(gating_output.clone(), topk)
    weights_sglang, indices_sglang = sglang_topk_softmax(gating_output.clone(), topk)

    weights_diff = torch.abs(weights_vllm - weights_sglang).mean().item()
    indices_match = torch.equal(indices_vllm, indices_sglang)

    if not VLLM_AVAILABLE:
        print("⚠️ vLLM not available, skipping comparison")
        return

    if (
        torch.allclose(weights_vllm, weights_sglang, atol=1e-3, rtol=1e-3)
        and indices_match
    ):
        print("✅ VLLM and SGLang topk_softmax implementations match")
    else:
        print(
            f"❌ Implementations differ: Weights diff={weights_diff}, Indices match={indices_match}"
        )

def calculate_diff_torch(num_tokens, num_experts, topk, optimize):
    gating_output = torch.randn(
        (num_tokens, num_experts), device="cuda", dtype=torch.float32
    )
    weights_vllm, indices_vllm = torch.topk(gating_output.clone(), k=topk, dim=1)
    weights_sglang, indices_sglang = sglang_topk_softmax(gating_output.clone(), topk, optimize)

    weights_diff = torch.abs(weights_vllm - weights_sglang).mean().item()
    indices_match = torch.equal(indices_vllm, indices_sglang)


    if (
        torch.allclose(weights_vllm, weights_sglang, atol=1e-3, rtol=1e-3)
        and indices_match
    ):
        print("✅ TORCH and SGLang topk_softmax implementations match")
    else:
        print(
            f"❌ Implementations differ: Weights diff={weights_diff}, Indices match={indices_match}"
        )


# CI environment uses simplified parameters
if IS_DIFF:
    num_tokens_range = [1024]  # Single value for CI
    num_experts_range = [512]  # Single value for CI
    topk_range = [2]  # Single value for CI
else:
    num_tokens_range = [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    num_experts_range = [512]
    topk_range = [1, 2, 4, 8]



def benchmark(num_tokens, num_experts, topk):

    gating_output = torch.randn(
        (num_tokens, num_experts), device="cuda", dtype=torch.float32
    )

    sglang_topk_softmax(gating_output, topk)


if __name__ == "__main__":
    for i in num_tokens_range:
        for j in num_experts_range:
            for k in topk_range:
                if IS_DIFF:
                    calculate_diff_torch(i, j, k, True)
                    calculate_diff_torch(i, j, k, False)
                else:
                    benchmark(i, j, k)
