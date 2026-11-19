#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from peft import LoraConfig, get_peft_model, PeftModel

class GeneralLoRAWrapper:
    """
    General-purpose LoRA adapter for any PyTorch model.
    Configure with target module names depending on the architecture.
    """
    def __init__(self, r=8, alpha=16, dropout=0.1,
                 target_modules=None, bias="none", task_type="VISION",
                 modules_to_save=None):
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.bias = bias
        self.task_type = task_type
        self.target_modules = target_modules or []
        self.modules_to_save = modules_to_save or []
        self._peft_model = None

    def to_dict(self):
        """LoRA config as a plain dict, saved into the checkpoint so inference
        rebuilds the identical LoraConfig instead of hardcoding target_modules.
        Keys match LoraConfig kwargs so it round-trips as LoraConfig(**d)."""
        return {
            "r": self.r,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "bias": self.bias,
            "target_modules": list(self.target_modules),
            "modules_to_save": list(self.modules_to_save),
        }

    def apply(self, model, device=None):
        """
        Wrap an existing base model with LoRA. Returns the wrapped model.
        Does not alter your training loop; only trainable params differ.
        """
        lcfg = LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            bias=self.bias,
            target_modules=self.target_modules,
            modules_to_save=self.modules_to_save,
        )
        if device is None:
            from appcore.ml_core.device_utils import get_device
            device = get_device()
        self._peft_model = get_peft_model(model, lcfg).to(device)
        return self._peft_model

    def configure_optimizer(self, lr, weight_decay=0.0):
        """
        Return an optimizer scoped to LoRA (and modules_to_save) trainable parameters only.
        """
        if self._peft_model is None:
            raise RuntimeError("Call apply() first.")
        params = [p for p in self._peft_model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    def save_adapters(self, output_dir: str):
        """
        Save only LoRA adapter weights (portable, small).
        """
        if self._peft_model is None:
            raise RuntimeError("No LoRA-wrapped model to save.")
        self._peft_model.save_pretrained(output_dir)

    def load_adapters(self, base_model, adapter_dir: str, device=None):
        """
        Load adapters onto a fresh base model. Returns LoRA-wrapped model.
        """
        if device is None:
            from appcore.ml_core.device_utils import get_device
            device = get_device()
        self._peft_model = PeftModel.from_pretrained(base_model, adapter_dir).to(device)
        return self._peft_model
