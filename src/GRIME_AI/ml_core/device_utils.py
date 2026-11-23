#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single place that decides which compute device GRIME AI uses.

Priority: NVIDIA CUDA, then Apple Silicon MPS, then CPU.

Every trainer, inference engine and GUI check calls these helpers instead
of calling torch.cuda directly, so CUDA-only calls are never made on a Mac.
"""

import os

# Ops that MPS has not implemented fall back to the CPU instead of raising.
# Must be set before torch initializes the MPS backend.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch


# ------------------------------------------------------------------------------------------------------------------
def mps_available() -> bool:
    try:
        return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    except Exception:
        return False


# ------------------------------------------------------------------------------------------------------------------
def get_device_type() -> str:
    """'cuda', 'mps' or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"


# ------------------------------------------------------------------------------------------------------------------
def get_device() -> torch.device:
    return torch.device(get_device_type())


# ------------------------------------------------------------------------------------------------------------------
def _type_of(device=None) -> str:
    if device is None:
        return get_device_type()
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, int):
        return "cuda"
    return str(device).split(":")[0]


# ------------------------------------------------------------------------------------------------------------------
def gpu_available() -> bool:
    """True when a CUDA or MPS GPU is usable."""
    return get_device_type() in ("cuda", "mps")


# ------------------------------------------------------------------------------------------------------------------
def gpu_name():
    """Display name of the GPU, or None when there is none."""
    t = get_device_type()
    if t == "cuda":
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "NVIDIA GPU"
    if t == "mps":
        return "Apple Silicon GPU (MPS)"
    return None


# ------------------------------------------------------------------------------------------------------------------
def amp_supported(device=None) -> bool:
    """Mixed precision (autocast + GradScaler) is used on CUDA only."""
    return _type_of(device) == "cuda"


# ------------------------------------------------------------------------------------------------------------------
def pin_memory_supported(device=None) -> bool:
    """Pinned host memory only speeds up CUDA transfers."""
    return _type_of(device) == "cuda"


# ------------------------------------------------------------------------------------------------------------------
def ultralytics_device(device=None):
    """Device value in the form Ultralytics expects (0, 'mps' or 'cpu')."""
    t = _type_of(device)
    return 0 if t == "cuda" else t


# ------------------------------------------------------------------------------------------------------------------
def empty_cache(device=None) -> None:
    t = _type_of(device)
    try:
        if t == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif t == "mps" and mps_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# ------------------------------------------------------------------------------------------------------------------
def synchronize(device=None) -> None:
    t = _type_of(device)
    try:
        if t == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif t == "mps" and mps_available():
            torch.mps.synchronize()
    except Exception:
        pass


# ------------------------------------------------------------------------------------------------------------------
def memory_mb(device=None):
    """(allocated_MB, reserved_MB) for the GPU, or None when not reported."""
    t = _type_of(device)
    try:
        if t == "cuda" and torch.cuda.is_available():
            return (torch.cuda.memory_allocated() / 1024 ** 2,
                    torch.cuda.memory_reserved() / 1024 ** 2)
        if t == "mps" and mps_available():
            return (torch.mps.current_allocated_memory() / 1024 ** 2,
                    torch.mps.driver_allocated_memory() / 1024 ** 2)
    except Exception:
        pass
    return None


# ------------------------------------------------------------------------------------------------------------------
def manual_seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif mps_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


# First Windows build number that is Windows 11 (Microsoft's published value).
WINDOWS_11_FIRST_BUILD = 22000


# ------------------------------------------------------------------------------------------------------------------
def _os_label() -> str:
    import platform
    system = platform.system()
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]} ({platform.machine()})"
    if system == "Windows":
        # Windows 11 reports itself as version 10.0, so platform.release()
        # can say "10". Windows 11 is identified by its build number.
        version = platform.version()            # e.g. "10.0.22631"
        try:
            build = int(version.split(".")[2])
        except (IndexError, ValueError):
            build = 0
        release = "11" if build >= WINDOWS_11_FIRST_BUILD else platform.release()
        try:
            edition = platform.win32_edition() or ""   # e.g. "Professional"
        except Exception:
            edition = ""
        edition = {"Professional": "Pro", "Core": "Home"}.get(edition, edition)
        name = f"Windows {release} {edition}".strip()
        return f"{name} (build {version})"
    return f"{system} {platform.release()} ({platform.machine()})"


# ------------------------------------------------------------------------------------------------------------------
def platform_info(device=None, mixed_precision=None) -> dict:
    """
    The platform a model was trained on, for the training report and model card.
    mixed_precision: text from the trainer (e.g. "bf16", "fp16", "off (fp32)").
    """
    import platform
    t = _type_of(device)
    gpu = None
    if t == "cuda":
        try:
            gpu = torch.cuda.get_device_name(0)
        except Exception:
            gpu = "NVIDIA GPU"
    elif t == "mps":
        gpu = "Apple Silicon GPU (MPS)"
    return {
        "device": {"cuda": "CUDA", "mps": "MPS", "cpu": "CPU"}.get(t, t),
        "gpu": gpu or "none",
        "os": _os_label(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "mixed_precision": mixed_precision or "N/A",
    }


# Display labels for platform_info() keys, in display order.
PLATFORM_LABELS = [
    ("device", "Device"),
    ("gpu", "GPU"),
    ("os", "Operating System"),
    ("python", "Python"),
    ("pytorch", "PyTorch"),
    ("mixed_precision", "Mixed Precision"),
]
