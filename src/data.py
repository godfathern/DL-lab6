"""i-CLEVR data handling: label encoding, image packing, and the training dataset."""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 24
IMAGE_SIZE = 64


def load_objects(objects_path):
    with open(objects_path) as f:
        return json.load(f)


def encode_labels(label_names, objects):
    """Turn a list like ["red cube", "blue sphere"] into a 24-dim multi-hot vector."""
    vec = torch.zeros(NUM_CLASSES)
    for name in label_names:
        vec[objects[name]] = 1.0
    return vec


def load_test_conditions(json_path, objects):
    """Return a (N, 24) multi-hot tensor in the same order as the test JSON list."""
    with open(json_path) as f:
        conditions = json.load(f)
    return torch.stack([encode_labels(c, objects) for c in conditions])


def pack_images(image_dir, train_json, objects_json, out_path, size=IMAGE_SIZE):
    """Resize every training image to size x size RGB and save images + labels in one .npz.

    Packing once makes training I/O trivial: the whole set is about 220 MB as uint8.
    """
    from PIL import Image

    objects = load_objects(objects_json)
    with open(train_json) as f:
        train = json.load(f)
    names = sorted(train)
    images = np.empty((len(names), size, size, 3), dtype=np.uint8)
    labels = np.zeros((len(names), NUM_CLASSES), dtype=np.uint8)
    for i, name in enumerate(names):
        with Image.open(os.path.join(image_dir, name)) as img:
            img = img.convert("RGB").resize((size, size), Image.BICUBIC)
            images[i] = np.asarray(img)
        for obj in train[name]:
            labels[i, objects[obj]] = 1
        if (i + 1) % 2000 == 0:
            print(f"packed {i + 1}/{len(names)}", flush=True)
    np.savez(out_path, images=images, labels=labels)
    print(f"saved {len(names)} images to {out_path}")


class ICLEVRDataset(Dataset):
    """Packed i-CLEVR images as float tensors in [-1, 1] with multi-hot labels."""

    def __init__(self, npz_path, hflip=True):
        data = np.load(npz_path)
        self.images = torch.from_numpy(data["images"]).permute(0, 3, 1, 2).contiguous()
        self.labels = torch.from_numpy(data["labels"]).float()
        # Horizontal flips keep every object's color and shape, so the labels stay valid.
        self.hflip = hflip

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].float() / 127.5 - 1.0
        if self.hflip and torch.rand(()) < 0.5:
            img = img.flip(-1)
        return img, self.labels[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack i-CLEVR training images into one .npz file.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--train-json", default="file/train.json")
    parser.add_argument("--objects-json", default="file/objects.json")
    parser.add_argument("--out", default="data/iclevr_64.npz")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pack_images(args.image_dir, args.train_json, args.objects_json, args.out)
