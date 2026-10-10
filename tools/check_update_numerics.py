"""Do two PPO-update configurations compute the same loss and gradients on the same minibatches?

    python tools/check_update_numerics.py --data upd.pkl --init policy.pt --a precision=bf16 --b precision=bf16,gru_precision=bf16

The update runs one epoch in "padded" mode (the math of the CUDA-graph path, eagerly) with the
optimizer step replaced by a recording of the loss and of the gradient, so the parameters stay
fixed and both configurations see identical minibatches (same generator seed). Per minibatch it
reports the relative difference of the loss and the relative L2 / cosine distance of the gradients;
the summary is the mean and the worst over `--steps` minibatches. Reference scale: the same
comparison of an FP32 and a BF16 update (--a precision=fp32 --b precision=bf16).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def record(net, opt, data, cfg, steps, seed, mode="padded", device="cuda"):
    import torch

    from mtg_ml.rl import ppo

    out = []

    def fake_step(net_, opt_, cfg_, loss, fold_clip=False):
        loss.backward()
        out.append((loss.item(), torch.cat([p.grad.flatten().float() for p in net_.parameters() if p.grad is not None]).cpu()))
        if len(out) >= steps:
            raise StopIteration

    real = ppo._step
    ppo._step = fake_step
    try:
        ppo.ppo_update(net, opt, data, cfg, device=device, gen=torch.Generator().manual_seed(seed), mode=mode)
    except StopIteration:
        pass
    finally:
        ppo._step = real
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--a", required=True, help="PPOConfig overrides of configuration A: k=v,k=v")
    ap.add_argument("--b", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    import torch

    from mtg_ml.rl.model import PolicyNet, load_partial
    from mtg_ml.rl.ppo import PPOConfig, make_optimizer

    data = pickle.load(open(args.data, "rb"))
    ck = torch.load(args.init, map_location="cpu", weights_only=False)

    def cfg_of(spec):
        kw = {}
        for kv in spec.split(","):
            if kv:
                k, v = kv.split("=")
                kw[k] = type(getattr(PPOConfig, k))(v) if not isinstance(getattr(PPOConfig, k), bool) else v == "1"
        return PPOConfig(minibatch=args.minibatch, epochs=1, target_kl=None, **kw)

    runs = []
    for spec in (args.a, args.b):
        net = PolicyNet(**ck["config"]).to(args.device)
        load_partial(net, ck["model"])
        cfg = cfg_of(spec)
        runs.append(record(net, make_optimizer(net.parameters(), cfg.lr, args.device), data, cfg, args.steps, 0, device=args.device))
    rel_loss, rel_grad, cos = [], [], []
    for (la, ga), (lb, gb) in zip(*runs):
        rel_loss.append(abs(la - lb) / max(abs(la), 1e-12))
        rel_grad.append(((ga - gb).norm() / ga.norm()).item())
        cos.append(torch.nn.functional.cosine_similarity(ga, gb, dim=0).item())
    n = len(rel_loss)
    print(f"A [{args.a}] vs B [{args.b}] over {n} minibatches of {args.minibatch}")
    print(f"loss relative difference: mean {sum(rel_loss) / n:.2e} max {max(rel_loss):.2e}")
    print(f"gradient relative L2 difference: mean {sum(rel_grad) / n:.2e} max {max(rel_grad):.2e}; cosine mean {sum(cos) / n:.6f} min {min(cos):.6f}")


if __name__ == "__main__":
    main()
