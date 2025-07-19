"""Media file information extraction using ffprobe."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from .exceptions import FFmpegError


class MediaInfo:
    """
    Media file information extracted via ffprobe.

    Provides duration, resolution, codec info, etc. without loading file into memory.
    """

    def __init__(self, probe_data: Dict[str, Any]):
        """
        Initialize from ffprobe JSON data.

        Args:
            probe_data: Raw ffprobe output as dict
        """
        self._data = probe_data
        self._format = probe_data.get("format", {})
        self._streams = probe_data.get("streams", [])

        # Cache common properties
        self._video_stream = None
        self._audio_stream = None
        for stream in self._streams:
            if stream.get("codec_type") == "video" and self._video_stream is None:
                self._video_stream = stream
            elif stream.get("codec_type") == "audio" and self._audio_stream is None:
                self._audio_stream = stream

    @classmethod
    def from_file(cls, file_path: Path) -> "MediaInfo":
        """
        Create MediaInfo from file using ffprobe.

        Args:
            file_path: Path to media file

        Returns:
            MediaInfo: Instance with file information

        Raises:
            FFmpegError: If ffprobe fails or is not available
        """
        if not shutil.which("ffprobe"):
            raise FFmpegError("ffprobe not found in PATH")

        command = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(file_path),
        ]

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            return cls(data)

        except subprocess.CalledProcessError as e:
            raise FFmpegError(
                f"ffprobe failed: {e.stderr}",
                command=command,
                returncode=e.returncode,
                stderr=e.stderr,
            )
        except json.JSONDecodeError as e:
            raise FFmpegError(f"Failed to parse ffprobe output: {e}")

    @property
    def duration(self) -> Optional[float]:
        """Duration in seconds."""
        duration = self._format.get("duration")
        return float(duration) if duration else None

    @property
    def file_size(self) -> Optional[int]:
        """File size in bytes."""
        size = self._format.get("size")
        return int(size) if size else None

    @property
    def bit_rate(self) -> Optional[int]:
        """Overall bit rate in bits/second."""
        bitrate = self._format.get("bit_rate")
        return int(bitrate) if bitrate else None

    @property
    def format_name(self) -> Optional[str]:
        """Format name (e.g. 'mov,mp4,m4a,3gp,3g2,mj2')."""
        return self._format.get("format_name")

    @property
    def has_video(self) -> bool:
        """True if file contains video stream."""
        return self._video_stream is not None

    @property
    def has_audio(self) -> bool:
        """True if file contains audio stream."""
        return self._audio_stream is not None

    @property
    def width(self) -> Optional[int]:
        """Video width in pixels."""
        if self._video_stream:
            width = self._video_stream.get("width")
            return int(width) if width else None
        return None

    @property
    def height(self) -> Optional[int]:
        """Video height in pixels."""
        if self._video_stream:
            height = self._video_stream.get("height")
            return int(height) if height else None
        return None

    @property
    def fps(self) -> Optional[float]:
        """Video frame rate (frames per second)."""
        if self._video_stream:
            fps_str = self._video_stream.get("r_frame_rate")
            if fps_str and "/" in fps_str:
                try:
                    num, den = fps_str.split("/")
                    return float(num) / float(den) if float(den) != 0 else None
                except (ValueError, ZeroDivisionError):
                    return None
        return None

    @property
    def video_codec(self) -> Optional[str]:
        """Video codec name."""
        if self._video_stream:
            return self._video_stream.get("codec_name")
        return None

    @property
    def audio_codec(self) -> Optional[str]:
        """Audio codec name."""
        if self._audio_stream:
            return self._audio_stream.get("codec_name")
        return None

    @property
    def sample_rate(self) -> Optional[int]:
        """Audio sample rate in Hz."""
        if self._audio_stream:
            rate = self._audio_stream.get("sample_rate")
            return int(rate) if rate else None
        return None

    @property
    def channels(self) -> Optional[int]:
        """Number of audio channels."""
        if self._audio_stream:
            channels = self._audio_stream.get("channels")
            return int(channels) if channels else None
        return None

    def __repr__(self) -> str:
        """String representation of media info."""
        parts = []
        if self.duration:
            parts.append(f"duration={self.duration:.2f}s")
        if self.width and self.height:
            parts.append(f"resolution={self.width}x{self.height}")
        if self.fps:
            parts.append(f"fps={self.fps:.2f}")
        if self.has_audio and self.channels:
            parts.append(f"audio={self.channels}ch")

        details = ", ".join(parts) if parts else "no info"
        return f"MediaInfo({details})"
