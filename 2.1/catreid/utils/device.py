import torch

def get_device(prefer_mps: bool = True) -> torch.device:
    # Default MPS for Apple Silicon
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
