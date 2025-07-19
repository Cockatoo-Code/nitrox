"""FFmpeg wrapper for executing media operations."""

import shutil
import subprocess
from pathlib import Path
from typing import List

from .exceptions import FFmpegError


class FFmpegRunner:
    """Builds and executes ffmpeg commands from operation queues."""

    def __init__(self):
        """Initialize FFmpeg runner and check if ffmpeg is available."""
        if not self._check_ffmpeg():
            raise FFmpegError(
                "ffmpeg not found in PATH. Please install ffmpeg: "
                "https://ffmpeg.org/download.html"
            )

    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg binary is available."""
        return shutil.which("ffmpeg") is not None

    def build_command(
        self, input_path: Path, output_path: Path, operations: List[tuple], **kwargs
    ) -> List[str]:
        """
        Build ffmpeg command from operations queue.

        Args:
            input_path: Input file path
            output_path: Output file path
            operations: List of (operation, params) tuples
            **kwargs: Additional ffmpeg options

        Returns:
            List[str]: FFmpeg command as list of arguments
        """
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

        # Input options
        input_opts = []
        video_filters = []
        audio_filters = []

        # Process operations in order
        for op_name, params in operations:
            if op_name == "slice":
                start = params.get("start")
                end = params.get("end")

                if start is not None:
                    input_opts.extend(["-ss", str(start)])
                if end is not None:
                    if start is not None:
                        duration = end - start
                        input_opts.extend(["-t", str(duration)])
                    else:
                        input_opts.extend(["-to", str(end)])

            elif op_name == "resize":
                width = params["width"]
                height = params["height"]
                video_filters.append(f"scale={width}:{height}")

            elif op_name == "crop":
                x, y = params["x"], params["y"]
                width, height = params["width"], params["height"]
                video_filters.append(f"crop={width}:{height}:{x}:{y}")

            elif op_name == "extract_audio":
                cmd.extend(["-vn"])  # No video

            elif op_name == "normalize_audio":
                audio_filters.append("loudnorm")

            elif op_name == "set_fps":
                fps = params["fps"]
                video_filters.append(f"fps={fps}")

            elif op_name == "fade_in":
                duration = params["duration"]
                audio_filters.append(f"afade=t=in:d={duration}")

            elif op_name == "fade_out":
                duration = params["duration"]
                audio_filters.append(f"afade=t=out:d={duration}")

            elif op_name == "resample":
                sample_rate = params["sample_rate"]
                audio_filters.append(f"aresample={sample_rate}")

            elif op_name == "to_mono":
                audio_filters.append("pan=mono|c0=0.5*c0+0.5*c1")

        # Add input options and file
        cmd.extend(input_opts)
        cmd.extend(["-i", str(input_path)])

        # Add video filters if any
        if video_filters:
            vf = ",".join(video_filters)
            cmd.extend(["-vf", vf])

        # Add audio filters if any
        if audio_filters:
            af = ",".join(audio_filters)
            cmd.extend(["-af", af])

        # Add encoding options
        self._add_encoding_options(cmd, output_path, **kwargs)

        # Add output file
        cmd.extend(["-y", str(output_path)])  # -y to overwrite

        return cmd

    def _add_encoding_options(
        self, cmd: List[str], output_path: Path, **kwargs
    ) -> None:
        """Add encoding options based on output format and user preferences."""
        ext = output_path.suffix.lower()

        # Video encoding options
        if ext in [".mp4", ".mov", ".mkv", ".avi"]:
            # Default to H.264 for video
            if "codec" not in kwargs:
                cmd.extend(["-c:v", "libx264"])

            # CRF for quality control
            crf = kwargs.get("crf", 23)  # Default quality
            cmd.extend(["-crf", str(crf)])

            # Preset for encoding speed
            preset = kwargs.get("preset", "medium")
            cmd.extend(["-preset", preset])

        # Audio encoding options
        if ext in [".mp3"]:
            if "codec" not in kwargs:
                cmd.extend(["-c:a", "libmp3lame"])
            bitrate = kwargs.get("bitrate", "192k")
            cmd.extend(["-b:a", bitrate])

        elif ext in [".wav"]:
            if "codec" not in kwargs:
                cmd.extend(["-c:a", "pcm_s16le"])

        elif ext in [".aac", ".m4a"]:
            if "codec" not in kwargs:
                cmd.extend(["-c:a", "aac"])
            bitrate = kwargs.get("bitrate", "128k")
            cmd.extend(["-b:a", bitrate])

        # Add any custom codec options
        if "codec" in kwargs:
            cmd.extend(["-c", kwargs["codec"]])

        # Add any additional options
        for key, value in kwargs.items():
            if key not in ["crf", "preset", "bitrate", "codec"]:
                cmd.extend([f"-{key}", str(value)])

    def execute(self, command: List[str]) -> None:
        """
        Execute ffmpeg command and handle errors.

        Args:
            command: FFmpeg command as list of arguments

        Raises:
            FFmpegError: If command fails
        """
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise FFmpegError(
                f"FFmpeg command failed: {e.stderr}",
                command=command,
                returncode=e.returncode,
                stderr=e.stderr,
            )
        except FileNotFoundError:
            raise FFmpegError(
                "FFmpeg binary not found. Please install ffmpeg.", command=command
            )
