import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha=-1, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()

        # BCE with logits
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # Probabilities
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)

        # Focal modulation
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
