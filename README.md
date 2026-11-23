# DL Lab 6: Conditional DDPM on i-CLEVR

A conditional DDPM that generates 64x64 i-CLEVR scenes from multi-hot object labels,
trained from scratch and scored with the provided ResNet18 evaluator.

| Test file | Evaluator accuracy |
|---|---|
| test.json | 0.9306 |
| new_test.json | 0.9167 |

## Code

| File | Purpose |
|---|---|
| `src/data.py` | Label encoding, packing images into one `.npz`, training dataset |
| `src/model.py` | Conditional U-Net (FiLM conditioning from timestep + label embeddings) |
| `src/diffusion.py` | Cosine schedule, epsilon-prediction loss, DDIM sampling with CFG and classifier guidance |
| `src/evaluate.py` | Loads the provided evaluator unmodified and scores images |
| `src/train.py` | Training loop with EMA, mixed precision, periodic evaluation |
| `src/generate.py` | Writes test images, 8x4 grids, the denoising grid, and accuracies |
| `kaggle/` | Kaggle launcher and private-dataset upload script used for training |

## Reproduce

Install PyTorch, torchvision, Pillow, and NumPy. Put the provided files in `file/` and the
i-CLEVR images in `data/iclevr/`.

1. Pack the training images:

   ```bash
   python src/data.py --image-dir data/iclevr --out data/iclevr_64.npz
   ```

2. Train (about 10.5 hours on one T4):

   ```bash
   cd src && python train.py --data ../data/iclevr_64.npz --files-dir ../file --out ../outputs/train
   ```

3. Generate the submission outputs and print both accuracies. The defaults are the
   final settings: 100 DDIM steps, start timestep 960, CFG scale 4, seed 0.

   ```bash
   cd src && python generate.py --ckpt ../outputs/train/best.pt --files-dir ../file --out ../outputs/final
   ```

4. Score the saved PNGs exactly as submitted:

   ```bash
   cd src && python score_images.py --images ../outputs/final/images --files-dir ../file
   ```
