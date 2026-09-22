"""GPU proof for the architecture-neutral canonical noising boundary."""

import torch

from pearl_gemm.sm120_reference import canonical_integer_noising


def sparse_k_by_r(k: int, r: int) -> torch.Tensor:
    result = torch.zeros((k, r), dtype=torch.int8, device="cuda")
    rows = torch.arange(k, device="cuda")
    result[rows, rows % r] = 1
    result[rows, (rows * 17 + 3) % r] = -1
    return result.contiguous()


def main() -> None:
    torch.manual_seed(7)
    m, n, k, r = 32, 64, 2048, 128
    A = torch.randint(-64, 64, (m, k), dtype=torch.int8, device="cuda")
    B = torch.randint(-64, 64, (n, k), dtype=torch.int8, device="cuda")
    EAL = torch.randint(-64, 64, (m, r), dtype=torch.int8, device="cuda")
    EBR = torch.randint(-64, 64, (n, r), dtype=torch.int8, device="cuda")
    EAR = sparse_k_by_r(k, r)
    EBL = sparse_k_by_r(k, r)

    ApEA, BpEB, AxEBL, EARxBpEB = canonical_integer_noising(A, B, EAL, EBR, EAR, EBL)

    assert torch.equal(ApEA, (A.to(torch.int32) + torch._int_mm(EAL, EAR.t())).to(torch.int8))
    assert torch.equal(BpEB, (B.to(torch.int32) + torch._int_mm(EBR, EBL.t())).to(torch.int8))
    assert torch.equal(AxEBL, torch._int_mm(A, EBL))
    assert torch.equal(EARxBpEB, torch._int_mm(BpEB, EAR))
    torch.cuda.synchronize()
    print("SM120 canonical integer noising: PASS")


if __name__ == "__main__":
    main()
