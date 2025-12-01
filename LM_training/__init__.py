import importlib.metadata
from .generation import generate

__all__ = [
    "generate",
]

try:
    __version__ = importlib.metadata.version("LM_training")
except importlib.metadata.PackageNotFoundError:
    # Fallback when running from source without installed package metadata
    __version__ = "0.0.0"
