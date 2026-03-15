"""
Triton kernels for FP8 quantization and dequantization.

Provides block-wise FP8 (E4M3FN) quantization for activations and weights,
BF16 dequantization, and FP8 block-scaled matrix multiplication.
"""

import torch
import triton
import triton.language as tl
from triton import Config

# ---------------------------------------------------------------------------
# Activation per tensor quantization
# ---------------------------------------------------------------------------

@triton.jit
def _fp8_dqd_kernel(
    x_ptr,           # 输入指针
    out_ptr,         # 输出指针
    scale_ptr,       # scale指针 (per-row)
    n_cols,          # 每行的元素数量
    BLOCK_SIZE: tl.constexpr,
):
    """
    FP8 DQD Kernel 
    """
    row_idx = tl.program_id(0)
    
    scale = tl.load(scale_ptr + row_idx)
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + col_offsets
        mask = cols < n_cols
        
        offset = row_idx * n_cols + cols
        
        x = tl.load(x_ptr + offset, mask=mask, other=0.0)
        
        x_scaled = x * scale
        
        FP8_MAX = 448.0
        x_clamped = tl.minimum(tl.maximum(x_scaled, -FP8_MAX), FP8_MAX)
        
        x_fp8 = x_clamped.to(tl.float8e4nv)
        x_dequant = x_fp8.to(tl.float32)
        

        result = x_dequant / scale

        tl.store(out_ptr + offset, result, mask=mask)
 
 
@triton.jit
def _compute_scale_kernel(
    x_ptr,
    scale_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    计算每行的scale: scale = FP8_MAX / max(|x|)
    """
    row_idx = tl.program_id(0)
    
    max_val = 0.0
    
    col_offsets = tl.arange(0, BLOCK_SIZE)

    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + col_offsets
        mask = cols < n_cols
        offset = row_idx * n_cols + cols
        
        x = tl.load(x_ptr + offset, mask=mask, other=0.0)
        max_val = tl.maximum(max_val, tl.max(tl.abs(x), axis=0))

    FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max
    EPS = 1e-12
    scale = FP8_MAX / tl.maximum(max_val, EPS)
    
    tl.store(scale_ptr + row_idx, scale)
 
 
@triton.jit
def _fp8_dqd_fused_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):

    row_idx = tl.program_id(0)
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    offset = row_idx * n_cols + col_offsets

    x = tl.load(x_ptr + offset, mask=mask, other=0.0)

    abs_x = tl.abs(x)
    max_val = tl.max(abs_x, axis=0)

    FP8_MAX: tl.constexpr = 448.0
    EPS: tl.constexpr = 1e-12
    scale = FP8_MAX / tl.maximum(max_val, EPS)

    x_scaled = x * scale
    
    x_fp8 = x_scaled.to(tl.float8e4nv)
    x_dequant = x_fp8.to(tl.float32)

    result = x_dequant / scale

    tl.store(out_ptr + offset, result, mask=mask)
 
 
def pseudo_quantize_tensor_per_tensor_fp8_triton(x: torch.Tensor) -> torch.Tensor:
    """
    Triton实现的per-row FP8 DQD
    
    Args:
        x: 输入tensor，shape为 (batch, ...) 
           会被reshape为 (batch, -1) 处理
    
    Returns:
        与输入相同shape的反量化tensor
    """
    original_shape = x.shape
    original_dtype = x.dtype

    if x.ndim == 1:
        x_2d = x.unsqueeze(0)
    else:
        x_2d = x.reshape(x.shape[0], -1)
    
    n_rows, n_cols = x_2d.shape
    
    x_2d = x_2d.float().contiguous()

    out = torch.empty_like(x_2d)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    if BLOCK_SIZE <= 65536:  
        _fp8_dqd_fused_kernel[(n_rows,)](
            x_2d, out, n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    else:  
        BLOCK_SIZE = 4096
        scale = torch.empty(n_rows, device=x.device, dtype=torch.float32)
        
        _compute_scale_kernel[(n_rows,)](
            x_2d, scale, n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        _fp8_dqd_kernel[(n_rows,)](
            x_2d, out, scale, n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    out = out.reshape(original_shape).to(original_dtype)
    
    return out


# ---------------------------------------------------------------------------
# Activation quantization
# ---------------------------------------------------------------------------

@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """Block-wise activation quantization to FP8 with per-block scaling."""
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def act_quant(x, block_size=128):
    """
    Quantize activations to FP8 (E4M3FN) with block-wise scaling.

    Args:
        x: Contiguous input tensor. Last dimension must be divisible by block_size.
        block_size: Block size for quantization granularity.

    Returns:
        Tuple of (quantized_tensor, scale_tensor).
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)  # noqa
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


# ---------------------------------------------------------------------------
# Weight cast: BF16/FP16 -> FP8
# ---------------------------------------------------------------------------

@triton.jit
def weight_cast_to_fp8_kernel(x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """2D block-wise weight quantization to FP8."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.0
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid_m * n + pid_n, s)


def weight_cast_to_fp8(x, block_size=128):
    """
    Quantize a 2D weight matrix to FP8 (E4M3FN) with 2D block-wise scaling.

    Args:
        x: Contiguous 2D weight tensor of shape (M, N).
        block_size: Block size for both row and column tiling.

    Returns:
        Tuple of (fp8_weight, scale_matrix).
    """
    assert x.is_contiguous()
    assert x.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    sM = torch.tensor(1.0 * M / block_size).ceil().int()
    sN = torch.tensor(1.0 * N / block_size).ceil().int()
    s = x.new_empty(sM, sN, dtype=torch.float32)
    grid = lambda meta: (  # noqa
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_cast_to_fp8_kernel[grid](x, y, s, M, N, BLOCK_SIZE=block_size)
    return y, s


# ---------------------------------------------------------------------------
# Weight cast: FP8 -> BF16
# ---------------------------------------------------------------------------

@triton.jit
def weight_cast_to_bf16_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """2D block-wise FP8 dequantization using scale factors."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_cast_to_bf16(x, s, block_size=128):
    """
    Dequantize an FP8 weight tensor back to the default float dtype (typically BF16).

    Args:
        x: Contiguous 2D FP8 weight tensor of shape (M, N).
        s: Contiguous 2D scale tensor.
        block_size: Block size matching the one used during quantization.

    Returns:
        Dequantized weight tensor of the same shape as `x`.
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (  # noqa
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_cast_to_bf16_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


# ---------------------------------------------------------------------------
# FP8 block-scaled matrix multiplication
# ---------------------------------------------------------------------------

fp8_gemm_configs = [
    Config(
        {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": 128},
        num_stages=num_stages,
        num_warps=8,
    )
    for block_m in [32]
    for block_n in [128]
    for num_stages in [3, 4, 5, 6]
]


@triton.autotune(configs=fp8_gemm_configs, key=["N", "K"])
@triton.jit
def w8a8_block_fp8_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Block-scaled FP8 GEMM kernel with per-block scaling for both operands."""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def w8a8_block_fp8_matmul(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor
):
    """
    Perform block-scaled FP8 matrix multiplication: C = (A * a_s) @ (B * b_s)^T.

    Args:
        a: First input matrix (contiguous).
        a_s: Per-block scale factors for `a` (contiguous).
        b: Second input matrix (contiguous, transposed layout).
        b_s: Per-block scale factors for `b` (contiguous).

    Returns:
        Result tensor in the default float dtype.
    """
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert (
        a_s.is_contiguous() and b_s.is_contiguous()
    ), "Scaling factor tensors must be contiguous"
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (  # noqa
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    w8a8_block_fp8_matmul_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    return c
