"""Score saved PNGs with the provided evaluator, for the report screenshots.

    python score_images.py --images ../outputs/final/images --files-dir ../file
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

from data import load_objects, load_test_conditions
from evaluate import load_evaluator, score
from train import pick_device


def load_pngs(folder, n):
    imgs = []
    for i in range(n):
        arr = np.array(Image.open(os.path.join(folder, f"{i}.png")).convert("RGB"), dtype=np.float32)
        imgs.append(torch.from_numpy(arr).permute(2, 0, 1) / 127.5 - 1.0)  # mean 0.5, std 0.5
    return torch.stack(imgs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="../outputs/final/images")
    p.add_argument("--files-dir", default="../file")
    args = p.parse_args()

    ev = load_evaluator(args.files_dir, pick_device())
    objects = load_objects(os.path.join(args.files_dir, "objects.json"))
    print("=" * 50)
    for name in ("test", "new_test"):
        labels = load_test_conditions(os.path.join(args.files_dir, f"{name}.json"), objects)
        imgs = load_pngs(os.path.join(args.images, name), len(labels))
        print(f"Evaluator accuracy  {name + '.json:':<15} {score(ev, imgs, labels):.4f}   ({len(labels)} images)")
    print("=" * 50)


if __name__ == "__main__":
    main()
