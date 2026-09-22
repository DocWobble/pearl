"""Architecture-neutral canonical integer noising for the SM120 boundary.

This is deliberately the same integer algebra used by Pearl's reference tests.
It is a correctness oracle for replacing the Hopper TMA/WGMMA noising kernels;
it is not an alternate proof format.
"""

from __future__ import annotations

import torch


def canonical_integer_noising(
    A: torch.Tensor,
    B: torch.Tensor,
    EAL: torch.Tensor,
    EBR: torch.Tensor,
    EAR_R_major: torch.Tensor,
    EBL_R_major: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(ApEA, BpEB, AxEBL_i32, EARxBpEB_i32)`` exactly.

    Inputs are contiguous CUDA int8 tensors.  ``EAR_R_major`` and
    ``EBL_R_major`` are K-by-R, matching Pearl's canonical representation.
    ``torch._int_mm`` preserves integer accumulation before the specified
    int8 narrowing of the noised operands.
    """
    tensors = (A, B, EAL, EBR, EAR_R_major, EBL_R_major)
    if not all(t.is_cuda and t.is_contiguous() and t.dtype is torch.int8 for t in tensors):
        raise ValueError("canonical SM120 noising requires contiguous CUDA int8 tensors")

    # These equations are the established reference in tests/test_pearl_gemm.py.
    EA_i32 = torch._int_mm(EAL, EAR_R_major.t().contiguous())
    EB_i32 = torch._int_mm(EBR, EBL_R_major.t().contiguous())
    ApEA = (A.to(torch.int32) + EA_i32).to(torch.int8)
    BpEB = (B.to(torch.int32) + EB_i32).to(torch.int8)
    AxEBL_i32 = torch._int_mm(A, EBL_R_major)
    EARxBpEB_i32 = torch._int_mm(BpEB, EAR_R_major)
    return ApEA, BpEB, AxEBL_i32, EARxBpEB_i32
