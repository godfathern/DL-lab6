"""Load the provided evaluator without modifying its model, weights, or scoring code."""
import importlib.util
import os

import torch


def load_evaluator(files_dir, device):
    """Return an evaluation_model instance from the provided evaluator.py.

    The provided class hard-codes the checkpoint path './checkpoint.pth' and calls .cuda(),
    so it is constructed from inside files_dir. Without CUDA (local Mac smoke tests), the
    same architecture and weights are built on another device using the same class, so
    compute_acc and eval are still the provided ones.
    """
    # PyTorch >= 2.6 defaults torch.load to weights_only=True; this env var restores the
    # old default for the evaluator's own torch.load call without editing the file.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    spec = importlib.util.spec_from_file_location("evaluator", os.path.join(files_dir, "evaluator.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cwd = os.getcwd()
    os.chdir(files_dir)
    try:
        if torch.cuda.is_available():
            return module.evaluation_model()
        import torch.nn as nn
        import torchvision.models as models

        ev = module.evaluation_model.__new__(module.evaluation_model)
        checkpoint = torch.load("./checkpoint.pth", map_location="cpu", weights_only=False)
        ev.resnet18 = models.resnet18(weights=None)
        ev.resnet18.fc = nn.Sequential(nn.Linear(512, 24), nn.Sigmoid())
        ev.resnet18.load_state_dict(checkpoint["model"])
        ev.resnet18 = ev.resnet18.to(device).eval()
        ev.classnum = 24
        return ev
    finally:
        os.chdir(cwd)


def evaluator_device(ev):
    return next(ev.resnet18.parameters()).device


def score(ev, images, labels):
    """Accuracy from the provided eval(); images in [-1, 1] (mean 0.5, std 0.5 normalization)."""
    dev = evaluator_device(ev)
    return ev.eval(images.to(dev), labels.to(dev))


def classifier_grad_fn(ev):
    """Gradient of sum_k log p(label_k | x0) over the requested labels, for classifier guidance.
    Uses the frozen evaluator network; its weights are never updated."""
    dev = evaluator_device(ev)

    def grad(x0, labels):
        x = x0.to(dev).requires_grad_(True)
        probs = ev.resnet18(x).clamp(1e-6, 1 - 1e-6)
        logp = (labels.to(dev) * probs.log()).sum()
        (g,) = torch.autograd.grad(logp, x)
        return g.to(x0.device)

    return grad
