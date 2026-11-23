"""Measured ceilings of one RTX PRO 6000 Blackwell: dense GEMM TFLOP/s (BF16, FP8, NVFP4 when
torch exposes it), grouped BF16 GEMM at the MoE shapes, and DRAM bandwidth.

python3 -m inkling2_train.hw_bench --out /work/hw_bench.json
"""

import argparse
import json
import subprocess
import time

import torch


def timeit(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        best.append((time.perf_counter() - t0) / iters)
    return min(best), sorted(best)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/hw_bench.json")
    args = ap.parse_args()
    dev = torch.device("cuda")
    p = torch.cuda.get_device_properties(dev)
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,clocks.max.sm,clocks.max.mem,power.limit,memory.total,pci.bus_id",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    res = {"device": p.name, "sm_count": p.multi_processor_count, "cc": f"{p.major}.{p.minor}",
           "total_mem_gib": p.total_memory / 2**30, "l2_mib": getattr(p, "L2_cache_size", 0) / 2**20,
           "nvidia_smi": smi, "torch": torch.__version__}
    n = 8192
    a = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    best, med = timeit(lambda: a @ b)
    res["bf16_gemm_8192_tflops"] = {"best": 2 * n**3 / best / 1e12, "median": 2 * n**3 / med / 1e12}
    a8, b8 = a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn).t()
    one = torch.ones((), device=dev)
    try:
        best, med = timeit(lambda: torch._scaled_mm(a8, b8, scale_a=one, scale_b=one, out_dtype=torch.bfloat16))
        res["fp8_gemm_8192_tflops"] = {"best": 2 * n**3 / best / 1e12, "median": 2 * n**3 / med / 1e12}
    except Exception as exc:
        res["fp8_gemm_8192_tflops"] = f"unavailable: {exc}"[:200]
    try:  # NVFP4 (e2m1 x2 packed) with e4m3 block-16 scales, swizzled layout
        from torch.nn.functional import ScalingType, SwizzleType  # noqa: F401
        a4 = torch.randint(0, 255, (n, n // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        b4 = torch.randint(0, 255, (n, n // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        sa = torch.ones(n * (n // 16), device=dev).to(torch.float8_e4m3fn)
        sb = torch.ones(n * (n // 16), device=dev).to(torch.float8_e4m3fn)
        fn = lambda: torch.nn.functional.scaled_mm(  # noqa: E731
            a4, b4.t(), sa, ScalingType.BlockWise1x16, sb, ScalingType.BlockWise1x16,
            swizzle_a=SwizzleType.SWIZZLE_32_4_4, swizzle_b=SwizzleType.SWIZZLE_32_4_4, output_dtype=torch.bfloat16)
        best, med = timeit(fn)
        res["nvfp4_gemm_8192_tflops"] = {"best": 2 * n**3 / best / 1e12, "median": 2 * n**3 / med / 1e12}
    except Exception as exc:
        res["nvfp4_gemm_8192_tflops"] = f"unavailable in torch: {exc}"[:200]
    del a, b, a8, b8
    # grouped BF16 GEMM at MoE shapes: 6144 tokens x top-8 rows over 128 experts
    rows, e = 6144 * 8, 128
    x = torch.randn(rows, 2304, device=dev, dtype=torch.bfloat16)
    w = torch.randn(e, 2304, 2304, device=dev, dtype=torch.bfloat16)
    offs = torch.arange(1, e + 1, device=dev, dtype=torch.int32) * (rows // e)
    try:
        best, med = timeit(lambda: torch._grouped_mm(x, w.transpose(-2, -1), offs=offs), iters=10)
        res["bf16_grouped_mm_moe_w13_tflops"] = {"best": 2 * rows * 2304 * 2304 / best / 1e12,
                                                 "median": 2 * rows * 2304 * 2304 / med / 1e12}
    except Exception as exc:
        res["bf16_grouped_mm_moe_w13_tflops"] = f"unavailable: {exc}"[:200]
    del x, w
    # DRAM bandwidth: device-to-device copy of 4 GiB (read + write) and a read-only reduction
    buf = torch.empty(2**30, device=dev, dtype=torch.float32)
    dst = torch.empty_like(buf)
    best, med = timeit(lambda: dst.copy_(buf), iters=10)
    res["dram_copy_gbs"] = {"best": 2 * buf.numel() * 4 / best / 1e9, "median": 2 * buf.numel() * 4 / med / 1e9}
    best, med = timeit(lambda: buf.sum(), iters=10)
    res["dram_read_gbs"] = {"best": buf.numel() * 4 / best / 1e9, "median": buf.numel() * 4 / med / 1e9}
    print(json.dumps(res, indent=1))
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
