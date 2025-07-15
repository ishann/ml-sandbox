import gc
import torch
def cleanup():
    """Clean up memory for macOS MPS backend."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()