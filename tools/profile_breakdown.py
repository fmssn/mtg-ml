"""Group the kernel rows of `bench_update.py --torch-profile-json` into the parts of the training step.

    python tools/profile_breakdown.py prof.json

Prints, per part of the network or the step, the CUDA time per step, its share of the summed kernel
time and the kernel launches per step. The grouping is by kernel name (cuDNN RNN kernels are the GRU,
flash / sdpa kernels the entity attention, and so on); a kernel that matches nothing lands in "other".
"""

import json
import re
import sys

PARTS = [  # (part, regex over the kernel name), first match wins
    ("GRU recurrence (cuDNN per-step gemm + cell kernels)", r"GRU_|elemWiseRNN|rnn|cutlass.*(s1688|64x64)|gemmSN_TN|Kernel2<cutlass_80_tensorop_s1688"),
    ("embedding bags (gather forward, transposed gradient)", r"EmbeddingBag|embedding_bag|index_add|tma_scatter"),
    ("entity attention (flash fwd/bwd)", r"sdpa|flash|compute_dot_do_o|convert_dq"),
    ("optimizer (fused Adam, grad norm)", r"multi_tensor|_fused_adam|fused_adam|foreach"),
    ("memset / memcpy", r"Memset|Memcpy|memcpy"),
    ("dense gemms (MLPs, scorer, projections)", r"nvjet|gemm|cutlass|cublas|sm90|sm80"),
    ("compiled pointwise / losses (Inductor triton)", r"^triton_"),
]


def main(path: str) -> None:
    d = json.load(open(path))
    steps, rows = d["steps"], [r for r in d["rows"] if r["cuda_us"] > 0]
    total = sum(r["cuda_us"] for r in rows)
    out = {}
    for r in rows:
        part = next((p for p, rx in PARTS if re.search(rx, r["name"])), "other")
        t, c = out.get(part, (0.0, 0))
        out[part] = (t + r["cuda_us"], c + r["calls"])
    print(f"{steps} steps, wall {d['wall_s']:.2f}s, summed kernel time {total / 1e6:.2f}s ({total / 1e3 / steps:.2f} ms/step)")
    for part, (t, c) in sorted(out.items(), key=lambda kv: -kv[1][0]):
        print(f"{part:58s} {t / 1e3 / steps:7.2f} ms/step {100 * t / total:5.1f}%  {c / steps:8.1f} launches/step")
    print(f"{'total':58s} {total / 1e3 / steps:7.2f} ms/step        {sum(c for _, c in out.values()) / steps:8.1f} launches/step")


if __name__ == "__main__":
    main(sys.argv[1])
