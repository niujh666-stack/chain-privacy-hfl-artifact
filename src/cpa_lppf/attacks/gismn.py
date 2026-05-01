from __future__ import annotations

import torch

from cpa_lppf.attacks.dlg import DLGAttack
from cpa_lppf.utils.metrics import psnr, ssim_simple


def total_variation(x: torch.Tensor) -> torch.Tensor:
    return (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean() + (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()


class GISMNAttack(DLGAttack):
    """GI-SMN-style stronger gradient inversion baseline with restarts and TV prior."""

    name = "GI-SMN"

    def __init__(self, feature_dim: int = 4096, steps: int = 500, lr: float = 0.08, restarts: int = 2, tv_weight: float = 1e-4):
        super().__init__(feature_dim=feature_dim, steps=steps, lr=lr)
        self.restarts = int(restarts)
        self.tv_weight = float(tv_weight)

    def invert_batch(self, model, images: torch.Tensor, labels: torch.Tensor, device: torch.device):
        model = model.to(device).eval()
        images = images.to(device)
        labels = labels.to(device).long()
        criterion = torch.nn.CrossEntropyLoss()
        logits = model(images)
        loss = criterion(logits, labels)
        true_grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=False)
        true_grads = [g.detach() for g in true_grads]

        best = None
        for restart in range(self.restarts):
            dummy_x = (torch.rand_like(images) * (0.8 + 0.1 * restart)).requires_grad_(True)
            dummy_logits = torch.randn(images.shape[0], model.num_classes, device=device, requires_grad=True)
            opt = torch.optim.Adam([dummy_x, dummy_logits], lr=self.lr)
            for step in range(self.steps):
                opt.zero_grad(set_to_none=True)
                pred = model(dummy_x.clamp(0, 1))
                soft_y = torch.softmax(dummy_logits, dim=1)
                dummy_loss = -(soft_y * torch.log_softmax(pred, dim=1)).sum(dim=1).mean()
                dummy_grads = torch.autograd.grad(dummy_loss, list(model.parameters()), create_graph=True)
                grad_loss = sum(torch.mean((dg - tg) ** 2) for dg, tg in zip(dummy_grads, true_grads))
                reg = self.tv_weight * total_variation(dummy_x)
                obj = grad_loss + reg
                obj.backward()
                opt.step()
                if step % 50 == 0:
                    with torch.no_grad():
                        dummy_x.add_(0.005 * torch.randn_like(dummy_x)).clamp_(0, 1)
            score = float(obj.detach().cpu().item())
            recon = dummy_x.detach().clamp(0, 1).cpu()
            if best is None or score < best[0]:
                best = (score, recon)
        recon = best[1]
        return {"reconstruction": recon, "psnr": psnr(recon, images.cpu()), "ssim": ssim_simple(recon, images.cpu())}
