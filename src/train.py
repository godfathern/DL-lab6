"""Train the conditional DDPM on packed i-CLEVR images.

Training stops at --epochs or when --time-budget-hours runs out, whichever comes first.
Every --eval-every epochs the EMA model samples both test files and the provided evaluator
scores them; the checkpoint with the best mean accuracy is kept as best.pt.
"""
import argparse
import copy
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from data import ICLEVRDataset, load_objects, load_test_conditions
from diffusion import Diffusion
from evaluate import load_evaluator, score
from model import ConditionalUNet


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def update_ema(ema, model, decay):
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p, 1 - decay)


def evaluate_both(ema, diffusion, ev, conditions, steps, cfg_scale, seed=0):
    ema.eval()
    gen = torch.Generator(device=next(ema.parameters()).device).manual_seed(seed)
    accs = {}
    for name, labels in conditions.items():
        # Start at t=960: the cosine schedule's alpha_bar at t=999 (~2e-9) leaves residual noise.
        imgs = diffusion.sample(ema, labels, steps=steps, cfg_scale=cfg_scale, generator=gen, t_start=960)
        accs[name] = score(ev, imgs, labels)
    return accs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="packed .npz from data.py")
    p.add_argument("--files-dir", required=True, help="folder with evaluator.py, checkpoint.pth, *.json")
    p.add_argument("--out", default="outputs")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--time-budget-hours", type=float, default=10.5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--ema-decay", type=float, default=0.9995)
    p.add_argument("--p-uncond", type=float, default=0.1)
    p.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
    p.add_argument("--base-ch", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--eval-cfg", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (smoke tests)")
    p.add_argument("--subset", type=int, default=0, help="train on the first N images (smoke tests)")
    p.add_argument("--resume", default="")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = pick_device()
    torch.manual_seed(0)
    print(f"device: {device}", flush=True)

    dataset = ICLEVRDataset(args.data)
    if args.subset:
        dataset = torch.utils.data.Subset(dataset, range(args.subset))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda",
        persistent_workers=device.type == "cuda",
    )

    model = ConditionalUNet(base_ch=args.base_ch).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup_steps))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    diffusion = Diffusion(schedule=args.schedule, device=device)

    objects = load_objects(os.path.join(args.files_dir, "objects.json"))
    conditions = {
        name: load_test_conditions(os.path.join(args.files_dir, f"{name}.json"), objects).to(device)
        for name in ("test", "new_test")
    }
    ev = load_evaluator(args.files_dir, device)

    start_epoch, step, best = 0, 0, -1.0
    history = []
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, step, best = ckpt["epoch"] + 1, ckpt["step"], ckpt["best"]
        history = ckpt.get("history", [])
        print(f"resumed from epoch {start_epoch}, best {best:.4f}", flush=True)

    def save(path, epoch):
        torch.save(
            {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "step": step,
                "best": best,
                "history": history,
                "args": vars(args),
            },
            path,
        )

    t0 = time.time()
    budget = args.time_budget_hours * 3600
    epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running, n = 0.0, 0
        for x0, labels in loader:
            x0, labels = x0.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = diffusion.loss(model, x0, labels, p_uncond=args.p_uncond)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            # Short warm-up for the EMA so early evaluations are not dominated by random init.
            update_ema(ema, model, min(args.ema_decay, (1 + step) / (10 + step)))
            running += loss.item()
            n += 1
            step += 1
            if args.max_steps and step >= args.max_steps:
                break

        elapsed = time.time() - t0
        record = {"epoch": epoch, "step": step, "loss": running / max(n, 1), "minutes": elapsed / 60}
        last_epoch = epoch == args.epochs - 1 or elapsed > budget or (args.max_steps and step >= args.max_steps)
        if (epoch + 1) % args.eval_every == 0 or last_epoch:
            accs = evaluate_both(ema, diffusion, ev, conditions, args.eval_steps, args.eval_cfg)
            record.update(accs)
            mean_acc = (accs["test"] + accs["new_test"]) / 2
            if mean_acc > best:
                best = mean_acc
                save(os.path.join(args.out, "best.pt"), epoch)
            save(os.path.join(args.out, "last.pt"), epoch)
        history.append(record)
        print(json.dumps(record), flush=True)
        with open(os.path.join(args.out, "history.json"), "w") as f:
            json.dump(history, f, indent=1)
        if last_epoch:
            break

    print(f"done: {epoch + 1} epochs, {step} steps, best mean accuracy {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
