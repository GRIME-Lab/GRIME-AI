#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# sam3_lora_trainer.py
#
# Drives SAM3 LoRA fine-tuning (Sompote/SAM3_LoRA) from the ML Image Processing
# training tab. It generates the YAML config the repo's SAM3TrainerNative
# expects, then launches train_sam3_lora_native.py as a SUBPROCESS.
#
# Why subprocess and not in-process import:
#   SAM3TrainerNative hardcodes bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz"
#   relative to the current working directory. Running the script with cwd set to
#   the SAM3_LoRA install/clone lets that relative asset path resolve. Importing
#   it in-process would run with the app's CWD and fail to find the vocab file.
#
# Outputs (written by the trainer to output_dir):
#   best_lora_weights.pt, last_lora_weights.pt, val_stats.json
#
# Author: John Edward Stranzl, Jr.
# License: Apache License, Version 2.0

import os
import sys
import subprocess
import tempfile
from pathlib import Path

import yaml


# ----------------------------------------------------------------------
# Locate the installed train_sam3_lora_native module (site-packages after a
# normal `pip install`, so it works in the app, conda, and Docker). No
# environment variables, no clone-folder assumptions.
# ----------------------------------------------------------------------
def resolve_trainer_script(explicit=None):
    # Optional explicit override (a path to the .py or a dir containing it),
    # passed programmatically — not from the environment.
    if explicit:
        if os.path.isfile(explicit):
            return os.path.abspath(explicit)
        cand = os.path.join(explicit, "train_sam3_lora_native.py")
        if os.path.isfile(cand):
            return os.path.abspath(cand)

    # Locate the installed module's file WITHOUT importing it. Importing
    # train_sam3_lora_native would pull in sam3 -> triton in the current
    # process; under the PyCharm debugger that breaks triton's DLL init. We
    # only need the path — the actual import happens in the training subprocess.
    import importlib.util
    spec = importlib.util.find_spec("train_sam3_lora_native")
    if spec and spec.origin:
        return os.path.abspath(spec.origin)
    return None


# Default LoRA target modules SAM3_LoRA fine-tunes.
_DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "fc1", "fc2"]


class SAM3LoRATrainer:
    """Builds the SAM3_LoRA config and runs its trainer as a subprocess.

    Mirrors the call shape of the other GRIME/OpsiLum trainers: construct with
    the parameters the tab collected, then call run().
    """

    def __init__(self, *,
                 data_dir,                 # COCO root with train/ (+ valid/) each holding _annotations.coco.json
                 output_dir,               # where best/last _lora_weights.pt land
                 num_epochs=50,
                 batch_size=2,
                 learning_rate=5e-5,
                 weight_decay=0.01,
                 # LoRA
                 rank=16, alpha=32, dropout=0.1,
                 target_modules=None,
                 apply_to_vision_encoder=False,  # freeze backbone: LoRA on decoder only (far less compute)
                 apply_to_mask_decoder=True,
                 apply_to_detr_encoder=False,
                 apply_to_detr_decoder=False,
                 apply_to_geometry_encoder=False,
                 apply_to_text_encoder=False,   # keep False: preserves prompt discrimination
                 # discrimination / negatives (the gauge-vs-pole fix)
                 num_negatives=1,
                 num_cross_negatives=1,
                 generic_negatives=None,
                 num_workers=0,
                 resolution=1008,  # model RoPE is fixed to 1008; other sizes assert in vitdet
                 source_annotation="instances_default.json",
                 checkpoint_path=None,
                 sam3_lora_home=None):
        self.data_dir = str(data_dir)
        self.output_dir = str(output_dir)
        self.source_annotation = source_annotation
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self._prepared_dir = None
        self.num_epochs = int(num_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.dropout = float(dropout)
        self.target_modules = list(target_modules or _DEFAULT_TARGET_MODULES)
        self.apply_to_vision_encoder = bool(apply_to_vision_encoder)
        self.apply_to_mask_decoder = bool(apply_to_mask_decoder)
        self.apply_to_detr_encoder = bool(apply_to_detr_encoder)
        self.apply_to_detr_decoder = bool(apply_to_detr_decoder)
        self.apply_to_geometry_encoder = bool(apply_to_geometry_encoder)
        self.apply_to_text_encoder = bool(apply_to_text_encoder)
        self.num_negatives = int(num_negatives)
        self.num_cross_negatives = int(num_cross_negatives)
        self.generic_negatives = generic_negatives
        self.num_workers = int(num_workers)
        self.resolution = int(resolution)

        self.script = resolve_trainer_script(sam3_lora_home)
        self._proc = None
        self._config_path = None

    # ------------------------------------------------------------------
    def build_config(self):
        """Return the config dict SAM3TrainerNative consumes (keys verified
        against train_sam3_lora_native.py)."""
        cfg = {
            "lora": {
                "rank": self.rank,
                "alpha": self.alpha,
                "dropout": self.dropout,
                "target_modules": self.target_modules,
                "apply_to_vision_encoder": self.apply_to_vision_encoder,
                "apply_to_text_encoder": self.apply_to_text_encoder,
                "apply_to_geometry_encoder": self.apply_to_geometry_encoder,
                "apply_to_detr_encoder": self.apply_to_detr_encoder,
                "apply_to_detr_decoder": self.apply_to_detr_decoder,
                "apply_to_mask_decoder": self.apply_to_mask_decoder,
            },
            "training": {
                "data_dir": self.data_dir,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "num_negatives": self.num_negatives,
                "num_cross_negatives": self.num_cross_negatives,
                "num_workers": self.num_workers,
                "resolution": self.resolution,
                "data_format": "coco",
            },
            "output": {
                "output_dir": self.output_dir,
            },
        }
        if self.checkpoint_path:
            cfg["model"] = {"checkpoint_path": self.checkpoint_path}
        if self.generic_negatives:
            cfg["training"]["generic_negatives"] = list(self.generic_negatives)
        return cfg

    # ------------------------------------------------------------------
    def write_config(self, path=None):
        cfg = self.build_config()
        if path is None:
            fd, path = tempfile.mkstemp(prefix="sam3_lora_", suffix=".yaml")
            os.close(fd)
        with open(path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        self._config_path = path
        return path

    # ------------------------------------------------------------------
    def prepare_dataset(self, val_fraction=0.2, seed=42):
        """Bridge GRIME's single flat COCO folder to the train/valid layout
        SAM3_LoRA requires.

        GRIME provides one folder with images + instances_default.json holding
        ALL annotated images. SAM3_LoRA wants <data_dir>/train/ and
        <data_dir>/valid/, each with its own _annotations.coco.json and images.
        We split the single annotation set by image into train/valid subsets,
        write a filtered COCO json + link the images into each folder.

        Sets self.data_dir to the prepared directory. Returns it.
        """
        src = self.data_dir

        # Already in SAM3 layout?
        if os.path.exists(os.path.join(src, "train", "_annotations.coco.json")):
            return src

        # Find the source COCO annotation.
        ann_src = None
        for name in (self.source_annotation, "instances_default.json",
                     "_annotations.coco.json"):
            cand = os.path.join(src, name)
            if os.path.exists(cand):
                ann_src = cand
                break
        if ann_src is None:
            raise RuntimeError(
                f"No COCO annotation found in {src} "
                f"(looked for {self.source_annotation}, instances_default.json, "
                f"_annotations.coco.json).")

        import json as _json
        import shutil
        import random as _random

        with open(ann_src, "r") as f:
            coco = _json.load(f)

        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        categories = coco.get("categories", [])

        # Split IMAGE ids into train/valid (deterministic).
        img_ids = [im["id"] for im in images]
        _random.Random(seed).shuffle(img_ids)
        n_val = int(len(img_ids) * val_fraction) if len(img_ids) > 1 else 0
        val_ids = set(img_ids[:n_val])
        train_ids = set(img_ids[n_val:])

        prepared = os.path.join(self.output_dir, "sam3_data")

        def _link_or_copy(fn):
            s = os.path.join(src, fn)
            return s if os.path.exists(s) else None

        def _write_split(split_name, keep_ids):
            split_dir = os.path.join(prepared, split_name)
            os.makedirs(split_dir, exist_ok=True)
            keep_imgs = [im for im in images if im["id"] in keep_ids]
            keep_anns = [an for an in annotations if an["image_id"] in keep_ids]
            sub = {
                "images": keep_imgs,
                "annotations": keep_anns,
                "categories": categories,
            }
            for k in ("info", "licenses"):
                if k in coco:
                    sub[k] = coco[k]
            with open(os.path.join(split_dir, "_annotations.coco.json"), "w") as f:
                _json.dump(sub, f)
            missing = 0
            for im in keep_imgs:
                fn = os.path.basename(im.get("file_name", ""))
                if not fn:
                    continue
                s = _link_or_copy(fn)
                if s is None:
                    missing += 1
                    continue
                d = os.path.join(split_dir, fn)
                if os.path.exists(d):
                    continue
                # Prefer a hard link: same bytes on disk, no duplication, and on
                # Windows it needs no admin/developer mode (same volume). Fall
                # back to symlink, then copy, only if hard-linking fails.
                try:
                    os.link(s, d)
                except (OSError, NotImplementedError, AttributeError):
                    try:
                        os.symlink(s, d)
                    except (OSError, NotImplementedError, AttributeError):
                        shutil.copyfile(s, d)
            if missing:
                print(f"[SAM3] warning: {missing} {split_name} image(s) "
                      f"not found in {src}")
            return len(keep_imgs)

        n_train = _write_split("train", train_ids)
        n_val_written = _write_split("valid", val_ids) if val_ids else 0
        print(f"[SAM3] dataset prepared: {n_train} train, "
              f"{n_val_written} valid images -> {prepared}")

        self._prepared_dir = prepared
        self.data_dir = prepared
        return prepared

    # ------------------------------------------------------------------
    def _preflight(self):
        """Validate everything before launching, so failures are explicit."""
        if self.script is None or not os.path.exists(self.script):
            raise RuntimeError(
                "SAM3 LoRA trainer (train_sam3_lora_native) is not installed in "
                "this environment. Install the sam3-lora package into the "
                "application's Python environment.")
        os.makedirs(self.output_dir, exist_ok=True)

        # Bridge GRIME's flat COCO folder into the required train/ layout.
        self.prepare_dataset()

        train_split = os.path.join(self.data_dir, "train", "_annotations.coco.json")
        if not os.path.exists(train_split):
            raise RuntimeError(
                f"COCO training annotations not found after preparation: "
                f"{train_split}")

    # ------------------------------------------------------------------
    def run(self, on_output=None, python_exe=None):
        """Launch training as a subprocess. Streams each stdout line to
        on_output(line) if given. Blocks until completion; returns the exit code.
        Call request_stop() from another thread to terminate early."""
        self._preflight()
        cfg_path = self.write_config()

        exe = python_exe or sys.executable
        cmd = [exe, self.script, "--config", cfg_path]

        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")

        # The trainer resolves its BPE vocab from the installed sam3 package, so
        # CWD no longer matters. Use the script's own directory as cwd.
        self._proc = subprocess.Popen(
            cmd, cwd=os.path.dirname(self.script), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        try:
            for line in self._proc.stdout:
                if on_output is not None:
                    on_output(line.rstrip("\n"))
                else:
                    print(line, end="")
        finally:
            self._proc.wait()

        rc = self._proc.returncode
        self._proc = None
        return rc

    # ------------------------------------------------------------------
    def request_stop(self):
        """Terminate the training subprocess if running."""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()

    # ------------------------------------------------------------------
    @property
    def best_weights_path(self):
        return os.path.join(self.output_dir, "best_lora_weights.pt")

    @property
    def last_weights_path(self):
        return os.path.join(self.output_dir, "last_lora_weights.pt")
