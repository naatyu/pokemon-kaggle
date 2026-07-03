from __future__ import annotations

import torch


def resolve_torch_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return requested


def configure_torch_runtime(device: str) -> None:
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")


def describe_torch_device(device: str) -> str:
    if not device.startswith("cuda"):
        return f"torch_device={device}"
    index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    capability = ".".join(str(part) for part in torch.cuda.get_device_capability(index))
    return f"torch_device={device} cuda_name={name} cuda_capability={capability}"
