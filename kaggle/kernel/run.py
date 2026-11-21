"""Kaggle launcher: train the conditional DDPM, then generate all Lab 6 outputs.

Inputs come from two private datasets attached to the notebook:
  lab6-ddpm-data  packed images (iclevr_64.npz) + evaluator.py, checkpoint.pth, *.json
  lab6-ddpm-code  the src/*.py files
Everything written to /kaggle/working/outputs is downloadable after the run.
"""
import glob
import os
import shutil
import subprocess
import sys

import torch

TIME_BUDGET_HOURS = float(os.environ.get("TIME_BUDGET_HOURS", "10.5"))


def find_dir(filename):
    matches = glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
    if not matches:
        sys.exit(f"could not find {filename} under /kaggle/input")
    return os.path.dirname(matches[0])


if not torch.cuda.is_available():
    sys.exit("No GPU attached. Check the notebook accelerator and the account's phone verification.")
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, flush=True)

files_dir = find_dir("evaluator.py")
data_path = os.path.join(find_dir("iclevr_64.npz"), "iclevr_64.npz")
# Copy the code somewhere writable so Python can cache bytecode.
src_dir = "/kaggle/working/src"
shutil.copytree(find_dir("train.py"), src_dir, dirs_exist_ok=True)
out_dir = "/kaggle/working/outputs"


def run(script, *args):
    cmd = [sys.executable, "-u", script, *args]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=src_dir, check=True)


run(
    "train.py",
    "--data", data_path,
    "--files-dir", files_dir,
    "--out", out_dir,
    "--epochs", "1000",
    "--time-budget-hours", str(TIME_BUDGET_HOURS),
    "--batch-size", "64",
)
run(
    "generate.py",
    "--ckpt", os.path.join(out_dir, "best.pt"),
    "--files-dir", files_dir,
    "--out", os.path.join(out_dir, "final"),
    "--steps", "250",
    "--sweep-cfg", "1,1.5,2,2.5,3,4,5",
    "--auto-best",
)
# Keep the downloadable output small: optimizer state is only needed for resuming.
ckpt = torch.load(os.path.join(out_dir, "best.pt"), map_location="cpu", weights_only=False)
torch.save({k: ckpt[k] for k in ("ema", "epoch", "best", "history", "args")}, os.path.join(out_dir, "best_ema.pt"))
