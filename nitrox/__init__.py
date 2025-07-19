"""
Nitrox: Lightning-fast Python library for slicing and converting audio/video files.

Zero bloat, blazing speed. Wraps ffmpeg with a sleek, Pythonic API.

Optional Dependencies:
- numpy: For audio array processing (to_numpy, from_numpy methods)
  Install with: pip install nitrox[audio]
"""

from .core import Media
from .exceptions import FFmpegError, MediaNotFoundError, NitroxException

# Ultra-minimal alias for power users
X = Media

__version__ = "0.1.0"
__author__ = "Cockatoo Team"
__email__ = "zac@cockatoo.com"

__all__ = [
    "Media",
    "X",  # Ultra-minimal alias
    "NitroxException",
    "FFmpegError",
    "MediaNotFoundError",
]
