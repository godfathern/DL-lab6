"""Generate the required Lab 6 outputs from a trained checkpoint.

Produces:
  images/test/0..31.png, images/new_test/0..31.png
  grid_test.png, grid_new_test.png       (8 x 4 grids)
  denoising_process.png                  ({blue cube, yellow sphere, purple cylinder})
  results.json                           (evaluator accuracies, plus any sweep results)
"""
import argparse
import json
import os

import torch
from torchvision.utils import make_grid, save_image

from data import encode_labels, load_objects, load_test_conditions
from diffusion import Diffusion
from evaluate import classifier_grad_fn, load_evaluator, score
from model import ConditionalUNet
from train import pick_device

DENOISE_CONDITION = ["blue cube", "yellow sphere", "purple cylinder"]


def to_unit(x):
    """Undo the mean 0.5 / std 0.5 normalization for saving viewable PNGs."""
    return (x.clamp(-1, 1) + 1) / 2


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    base_ch = ckpt.get("args", {}).get("base_ch", 128)
    schedule = ckpt.get("args", {}).get("schedule", "cosine")
    model = ConditionalUNet(base_ch=base_ch).to(device)
    model.load_state_dict(ckpt["ema"])
    return model.eval(), schedule, ckpt


def run(model, diffusion, ev, labels, steps, cfg, cg, seed, cg_fn):
    gen = torch.Generator(device=labels.device).manual_seed(seed)
    return diffusion.sample(
        model, labels, steps=steps, cfg_scale=cfg, classifier_grad=cg_fn if cg > 0 else None, cg_scale=cg, generator=gen
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--files-dir", required=True)
    p.add_argument("--out", default="outputs/final")
    p.add_argument("--steps", type=int, default=250, help="DDIM steps (1000 = full DDPM length)")
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--cg", type=float, default=0.0, help="classifier guidance weight (0 = off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep-cfg", default="", help="comma-separated CFG scales to compare, e.g. 1,1.5,2,3,4")
    p.add_argument("--sweep-cg", default="", help="comma-separated classifier-guidance weights to compare")
    p.add_argument("--auto-best", action="store_true", help="use the best swept setting for the final outputs")
    args = p.parse_args()

    device = pick_device()
    model, schedule, ckpt = load_model(args.ckpt, device)
    diffusion = Diffusion(schedule=schedule, device=device)
    ev = load_evaluator(args.files_dir, device)
    cg_fn = classifier_grad_fn(ev)
    objects = load_objects(os.path.join(args.files_dir, "objects.json"))
    conditions = {
        name: load_test_conditions(os.path.join(args.files_dir, f"{name}.json"), objects).to(device)
        for name in ("test", "new_test")
    }
    os.makedirs(args.out, exist_ok=True)
    results = {"checkpoint_epoch": ckpt.get("epoch"), "steps": args.steps, "seed": args.seed, "sweeps": []}

    def evaluate_setting(cfg, cg):
        accs = {}
        for name, labels in conditions.items():
            accs[name] = score(ev, run(model, diffusion, ev, labels, args.steps, cfg, cg, args.seed, cg_fn), labels)
        row = {"cfg": cfg, "cg": cg, **accs, "mean": (accs["test"] + accs["new_test"]) / 2}
        print(json.dumps(row), flush=True)
        results["sweeps"].append(row)
        return row

    for cfg in [float(s) for s in args.sweep_cfg.split(",") if s]:
        evaluate_setting(cfg, args.cg)
    for cg in [float(s) for s in args.sweep_cg.split(",") if s]:
        evaluate_setting(args.cfg, cg)
    cfg, cg = args.cfg, args.cg
    if args.auto_best and results["sweeps"]:
        best = max(results["sweeps"], key=lambda r: r["mean"])
        cfg, cg = best["cfg"], best["cg"]
    results["final_cfg"], results["final_cg"] = cfg, cg

    for name, labels in conditions.items():
        imgs = run(model, diffusion, ev, labels, args.steps, cfg, cg, args.seed, cg_fn)
        acc = score(ev, imgs, labels)
        results[name] = acc
        img_dir = os.path.join(args.out, "images", name)
        os.makedirs(img_dir, exist_ok=True)
        for i, img in enumerate(to_unit(imgs)):
            save_image(img, os.path.join(img_dir, f"{i}.png"))
        save_image(make_grid(to_unit(imgs), nrow=8, padding=2), os.path.join(args.out, f"grid_{name}.png"))

    # Denoising process for the required condition: 10 frames from pure noise to the result.
    label = encode_labels(DENOISE_CONDITION, objects)[None].to(device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    final, traj = diffusion.sample(
        model, label, steps=args.steps, cfg_scale=cfg, classifier_grad=cg_fn if cg > 0 else None, cg_scale=cg,
        generator=gen, return_trajectory=True,
    )
    idx = torch.linspace(0, len(traj) - 1, 10).round().long().tolist()
    frames = torch.cat([traj[i] for i in idx])
    save_image(make_grid(to_unit(frames), nrow=len(idx), padding=2), os.path.join(args.out, "denoising_process.png"))
    results["denoising_condition_acc"] = score(ev, final, label)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("=" * 50)
    print(f"Evaluator accuracy  test.json:     {results['test']:.4f}")
    print(f"Evaluator accuracy  new_test.json: {results['new_test']:.4f}")
    print(f"(cfg={cfg}, cg={cg}, DDIM steps={args.steps}, seed={args.seed}, epoch={results['checkpoint_epoch']})")
    print("=" * 50, flush=True)


if __name__ == "__main__":
    main()
