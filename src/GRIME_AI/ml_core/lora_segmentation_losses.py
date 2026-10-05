#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: John Edward Stranzl, Jr.
# Affiliation(s): University of Nebraska-Lincoln, Blade Vision Systems, LLC
# Contact: jstranzl2@huskers.unl.edu, johnstranzl@gmail.com
# Created: Oct 31, 2025
# License: Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# ============================================================================
# ===                         class BinaryDiceLoss                         ===
# ============================================================================
# ============================================================================
class BinaryDiceLoss(nn.Module):
    # Binary Dice Loss: expects probs [B,1,H,W], targets [B,1,H,W] in {0,1}
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Flatten
        probs_f = probs.reshape(probs.size(0), -1)
        targets_f = targets.reshape(targets.size(0), -1)

        inter = (probs_f * targets_f).sum(dim=1)
        union = probs_f.sum(dim=1) + targets_f.sum(dim=1)
        dice = (2.0 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


# ============================================================================
# ============================================================================
# ===                        class MultiClassDiceLoss                      ===
# ============================================================================
# ============================================================================
class MultiClassDiceLoss(nn.Module):
    # Multi-class Dice Loss: expects probs [B,C,H,W], targets [B,H,W] with class indices
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = probs.size(1)
        total_dice = 0.0

        for c in range(num_classes):
            probs_c = probs[:, c]                  # [B,H,W]
            targets_c = (targets == c).float()     # one-hot mask for class c

            probs_f = probs_c.reshape(probs_c.size(0), -1)
            targets_f = targets_c.reshape(targets_c.size(0), -1)

            inter = (probs_f * targets_f).sum(dim=1)
            union = probs_f.sum(dim=1) + targets_f.sum(dim=1)
            dice_c = (2.0 * inter + self.eps) / (union + self.eps)

            total_dice += dice_c.mean()

        return 1.0 - total_dice / num_classes

# ======================================================================================================================
# ======================================================================================================================
# =====     =====     =====     =====     =====      class TverskyLoss      =====     =====     =====     =====    =====
# ======================================================================================================================
# ======================================================================================================================
class TverskyLoss(nn.Module):
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # Works for binary and multi-class. logits: [B,C,H,W]; targets: [B,H,W]
        # with class indices. Averages the Tversky index over foreground classes
        # (skips class 0 = background), mirroring MultiClassDiceLoss. For C==2
        # this reduces to the original binary behavior on class 1.
        probs = torch.softmax(logits, dim=1)
        num_classes = probs.size(1)
        start = 1 if num_classes > 1 else 0   # skip background when multi-class
        losses = []
        for c in range(start, num_classes):
            p_c = probs[:, c]
            t_c = (targets == c).float()
            TP = (p_c * t_c).sum(dim=(1, 2))
            FP = (p_c * (1 - t_c)).sum(dim=(1, 2))
            FN = ((1 - p_c) * t_c).sum(dim=(1, 2))
            tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
            losses.append(1.0 - tversky.mean())
        if not losses:
            return logits.sum() * 0.0   # no foreground class present
        return torch.stack(losses).mean()


# ======================================================================================================================
# ======================================================================================================================
# =====     =====     =====     =====     =====       class FocalLoss       =====     =====     =====     =====    =====
# ======================================================================================================================
# ======================================================================================================================
class FocalLoss(nn.Module):
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()

# ======================================================================================================================
# ======================================================================================================================
# =====     =====     =====     =====     =====   class LovaszSoftmaxLoss   =====     =====     =====     =====    =====
# ======================================================================================================================
# ======================================================================================================================
class LovaszSoftmaxLoss(nn.Module):
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # logits: [B, C, H, W], targets: [B, H, W]
        probs = F.softmax(logits, dim=1)
        return self.lovasz_softmax_flat(probs, targets)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def lovasz_softmax_flat(self, probs, labels):
        B, C, H, W = probs.shape
        losses = []
        for c in range(1, C):  # skip background
            fg = (labels == c).float()
            if fg.sum() == 0:
                continue
            errors = (fg - probs[:, c]).abs()
            errors_sorted, perm = torch.sort(errors.view(-1), descending=True)
            fg_sorted = fg.view(-1)[perm]
            grad = self.lovasz_grad(fg_sorted)
            losses.append(torch.dot(errors_sorted, grad))
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def lovasz_grad(self, gt_sorted):
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jaccard = 1.0 - intersection / union
        if jaccard.numel() == 0:
            return gt_sorted
        return jaccard
