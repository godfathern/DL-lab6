"""DDPM forward process, epsilon-prediction loss, and DDIM / DDPM sampling with guidance."""
import math

import torch
import torch.nn.functional as F


def cosine_alpha_bar(T, s=0.008):
    """Cumulative signal fraction alpha_bar_t from Nichol & Dhariwal (2021), t = 1..T."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0)


def linear_alpha_bar(T, beta_start=1e-4, beta_end=0.02):
    """The original DDPM linear beta schedule (Ho et al., 2020)."""
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
    return torch.cumprod(1 - betas, dim=0)


class Diffusion:
    def __init__(self, T=1000, schedule="cosine", device="cpu"):
        self.T = T
        alpha_bar = cosine_alpha_bar(T) if schedule == "cosine" else linear_alpha_bar(T)
        self.alpha_bar = alpha_bar.float().to(device)

    def q_sample(self, x0, t, noise):
        """Forward process: x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps."""
        ab = self.alpha_bar[t][:, None, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def loss(self, model, x0, labels, p_uncond=0.1):
        """Simple DDPM loss on predicted noise. Labels are zeroed with probability p_uncond
        so the same network also learns the unconditional model needed for CFG."""
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        drop = (torch.rand(b, device=x0.device) < p_uncond).float()[:, None]
        labels = labels * (1 - drop)
        pred = model(self.q_sample(x0, t, noise), t, labels)
        return F.mse_loss(pred.float(), noise)

    def guided_eps(self, model, x, t_batch, labels, cfg_scale):
        """Classifier-free guidance: eps_u + w (eps_c - eps_u). cfg_scale=1 is plain conditional."""
        if cfg_scale == 1.0:
            return model(x, t_batch, labels)
        both = model(torch.cat([x, x]), torch.cat([t_batch, t_batch]), torch.cat([labels, torch.zeros_like(labels)]))
        eps_c, eps_u = both.chunk(2)
        return eps_u + cfg_scale * (eps_c - eps_u)

    @torch.no_grad()
    def sample(
        self,
        model,
        labels,
        steps=100,
        cfg_scale=2.0,
        eta=0.0,
        classifier_grad=None,
        cg_scale=0.0,
        generator=None,
        return_trajectory=False,
        clip_x0=True,
        t_start=None,
    ):
        """DDIM sampling (eta=0 deterministic, eta=1 matches DDPM ancestral sampling).

        classifier_grad(x0_hat, labels) returns d log p(labels | x0_hat) / d x0_hat, which is
        added as classifier guidance (Dhariwal & Nichol, 2021) with weight cg_scale.
        Returns final images in [-1, 1] and, optionally, the predicted x_t at each step.
        """
        device = labels.device
        n = labels.shape[0]
        x = torch.randn(n, 3, 64, 64, device=device, generator=generator)
        first = self.T - 1 if t_start is None else t_start
        timesteps = torch.linspace(first, 0, steps, device=device).round().long()
        trajectory = [x.clone()]
        for i, t in enumerate(timesteps):
            t_batch = t.repeat(n)
            ab_t = self.alpha_bar[t]
            ab_prev = self.alpha_bar[timesteps[i + 1]] if i + 1 < steps else torch.tensor(1.0, device=device)
            eps = self.guided_eps(model, x, t_batch, labels, cfg_scale).float()
            x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            if clip_x0:
                x0 = x0.clamp(-1, 1)
            if classifier_grad is not None and cg_scale > 0:
                # Gradient w.r.t. the predicted clean image, mapped to x_t through
                # dx0/dx_t ~ 1/sqrt(alpha_bar_t) (the U-Net Jacobian is ignored).
                with torch.enable_grad():
                    grad = classifier_grad(x0.detach(), labels) / ab_t.sqrt()
                eps = eps - cg_scale * (1 - ab_t).sqrt() * grad
                x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
                if clip_x0:
                    x0 = x0.clamp(-1, 1)
            if clip_x0:
                eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
            sigma = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).clamp(min=0).sqrt()
            noise = torch.randn(x.shape, device=device, generator=generator) if i + 1 < steps else 0
            x = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * eps + sigma * noise
            if return_trajectory:
                trajectory.append(x.clone())
        x = x.clamp(-1, 1)
        return (x, trajectory) if return_trajectory else x
