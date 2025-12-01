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

IS_TRITON =  (
    os.getenv("IS_TRI", "0").lower() == "1"
)

import os
import logging
import sys

def setup_logger():
    """从环境变量设置日志级别"""
    logger = logging.getLogger("moe")
    
    # 从环境变量获取级别，默认为WARNING
    log_level_str = os.getenv("LOG_LEVEL", "WARNING").upper()
    print("log level ", log_level_str)
    
    # 将字符串转换为logging级别
    log_level = getattr(logging, log_level_str, logging.WARNING)
    
    logger.setLevel(log_level)
    
    # 避免重复添加handler
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        # handler级别也从环境变量获取，默认与logger相同
        handler_level_str = os.getenv("MOE_HANDLER_LEVEL", log_level_str).upper()
        handler_level = getattr(logging, handler_level_str, log_level)
        handler.setLevel(handler_level)
        
        # 设置格式
        formatter = logging.Formatter(
            '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# 使用
logger = setup_logger()



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
        logger.info("⚠️ vLLM not available, skipping comparison")
        return

    if (
        torch.allclose(weights_vllm, weights_sglang, atol=1e-3, rtol=1e-3)
        and indices_match
    ):
        logger.info("✅ VLLM and SGLang topk_softmax implementations match")
    else:
        logger.info(
            f"❌ Implementations differ: Weights diff={weights_diff}, Indices match={indices_match}"
        )

def calculate_diff_torch(num_tokens, num_experts, topk, optimize):
    gating_output = torch.randn(
        (num_tokens, num_experts), device="cuda", dtype=torch.float32
    )
    weights_torch, indices_torch = torch.topk(gating_output.clone(), k=topk, dim=1)
    weights_sglang, indices_sglang = sglang_topk_softmax(gating_output.clone(), topk, optimize)

    weights_diff = torch.abs(weights_torch - weights_sglang).mean().item()
    indices_match = torch.equal(indices_torch, indices_sglang)


    if (
        torch.allclose(weights_torch, weights_sglang, atol=1e-3, rtol=1e-3)
        and indices_match
    ):
        logger.info("✅ TORCH and SGLang topk_softmax implementations match")
    else:
        logger.info(
            f"❌ Implementations differ: Weights diff={weights_diff}, Indices match={indices_match}, Optimize {optimize}"
        )
        if optimize:
            i = 0
            logger.debug("indices_torch ", indices_torch)
            logger.debug("indices_sglang ", indices_sglang)
            for th, sgl in zip(indices_torch, indices_sglang):
                if not torch.equal(th, sgl):
                    for tth, ssgl in zip(th, sgl):
                        logger.info("i %d th compare", i)
                        logger.info("torch result: index %s value %s", tth, gating_output[i][tth])
                        logger.info("sglang result: index %s value %s", ssgl, gating_output[i][ssgl])
                    raise Exception("optimize topk wrong ", num_tokens, " ", num_experts, " ", topk)
                i = i + 1

if IS_DIFF:
    num_tokens_range = [1024, 2048, 4096, 8192]  # Single value for CI
    num_experts_range = [512]  # Single value for CI
    topk_range = [1, 2, 4, 8]  # Single value for CI
    # num_experts_range = [32, 64, 128, 256, 12]
else:
    num_tokens_range = [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    num_experts_range = [512]
    # num_experts_range = [32, 64, 128, 256, 12]
    topk_range = [1, 2, 4, 8]

configs = list(itertools.product(num_tokens_range, num_experts_range, topk_range))


def benchmark(num_tokens, num_experts, topk, optimize):

    gating_output = torch.randn(
        (num_tokens, num_experts), device="cuda", dtype=torch.float32
    )

    sglang_topk_softmax(gating_output, topk, optimize)

line_vals = ["sglang"]
line_names = ["SGLang"]
styles = [("blue", "-")]

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens", "num_experts", "topk"],
        x_vals=configs,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="Latency (us)",
        plot_name="topk-softmax-performance",
        args={},
    )
)
def triton_benchmark(num_tokens, num_experts, topk, provider):

    gating_output = torch.randn(
        (num_tokens, num_experts), device="cuda", dtype=torch.float32
    )

    fn = lambda: sglang_topk_softmax(gating_output, topk)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(fn, quantiles=quantiles)

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    for i in num_tokens_range:
        for j in num_experts_range:
            for k in topk_range:
                if IS_DIFF:
                    logger.debug("calculate diff start num_tokens ", i, " num_experts ", j, " topk ", k)
                    calculate_diff_torch(i, j, k, True)
                    logger.debug("calculate diff end")
                elif IS_TRITON:
                    triton_benchmark.run(print_data=True)
                else:
                    logger.debug("benchmark start num_tokens ", i, " num_experts ", j, " topk ", k)
                    benchmark(i, j, k, True)
                    logger.debug("benchmark end")
