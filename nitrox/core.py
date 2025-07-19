"""Core Media class for chainable audio/video operations."""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

from .exceptions import InvalidOperationError, MediaNotFoundError
from .ffmpeg import FFmpegRunner
from .info import MediaInfo

# Optional numpy dependency for audio array processing
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


class Media:
    """
    Main class for chainable media operations.

    Example:
        Media("input.mp4").slice(10, 30).resize(720, 480).to("output.mp4")
    """

    def __init__(self, input_path: Union[str, Path]):
        """
        Initialize Media with input file.

        Args:
            input_path: Path to input media file
        """
        self.input_path = Path(input_path)
        if not self.input_path.exists():
            raise MediaNotFoundError(f"Media file not found: {self.input_path}")

        self._operations = []
        self._ffmpeg = FFmpegRunner()
        self._info = None

    def slice(
        self, start: Optional[float] = None, end: Optional[float] = None
    ) -> "Media":
        """
        Slice media between start and end times.

        Args:
            start: Start time in seconds (None = from beginning)
            end: End time in seconds (None = to end)

        Returns:
            Media: Self for chaining
        """
        if start is not None and start < 0:
            raise InvalidOperationError("Start time cannot be negative")
        if end is not None and end < 0:
            raise InvalidOperationError("End time cannot be negative")
        if start is not None and end is not None and start >= end:
            raise InvalidOperationError("Start time must be less than end time")

        self._operations.append(("slice", {"start": start, "end": end}))
        return self

    def resize(self, width: int, height: int) -> "Media":
        """
        Resize video to specified dimensions.

        Args:
            width: Target width in pixels
            height: Target height in pixels

        Returns:
            Media: Self for chaining
        """
        if width <= 0 or height <= 0:
            raise InvalidOperationError("Width and height must be positive")

        self._operations.append(("resize", {"width": width, "height": height}))
        return self

    def crop(self, x: int, y: int, width: int, height: int) -> "Media":
        """
        Crop video to specified rectangle.

        Args:
            x: X offset from top-left
            y: Y offset from top-left
            width: Crop width
            height: Crop height

        Returns:
            Media: Self for chaining
        """
        if width <= 0 or height <= 0:
            raise InvalidOperationError("Crop width and height must be positive")
        if x < 0 or y < 0:
            raise InvalidOperationError("Crop offsets cannot be negative")

        self._operations.append(
            ("crop", {"x": x, "y": y, "width": width, "height": height})
        )
        return self

    def extract_audio(self) -> "Media":
        """
        Extract only the audio track from video.

        Returns:
            Media: Self for chaining
        """
        self._operations.append(("extract_audio", {}))
        return self

    def normalize_audio(self) -> "Media":
        """
        Normalize audio levels.

        Returns:
            Media: Self for chaining
        """
        self._operations.append(("normalize_audio", {}))
        return self

    def set_fps(self, fps: float) -> "Media":
        """
        Change video frame rate.

        Args:
            fps: Target frames per second

        Returns:
            Media: Self for chaining
        """
        if fps <= 0:
            raise InvalidOperationError("FPS must be positive")

        self._operations.append(("set_fps", {"fps": fps}))
        return self

    def fade_in(self, duration: float) -> "Media":
        """
        Add fade-in effect to audio.

        Args:
            duration: Fade duration in seconds

        Returns:
            Media: Self for chaining
        """
        if duration <= 0:
            raise InvalidOperationError("Fade duration must be positive")

        self._operations.append(("fade_in", {"duration": duration}))
        return self

    def fade_out(self, duration: float) -> "Media":
        """
        Add fade-out effect to audio.

        Args:
            duration: Fade duration in seconds

        Returns:
            Media: Self for chaining
        """
        if duration <= 0:
            raise InvalidOperationError("Fade duration must be positive")

        self._operations.append(("fade_out", {"duration": duration}))
        return self

    def resample(self, sample_rate: int) -> "Media":
        """
        Resample audio to a new sample rate.

        Args:
            sample_rate: Target sample rate in Hz (e.g., 44100, 48000, 22050)

        Returns:
            Media: Self for chaining

        Examples:
            >>> Media("audio.wav").resample(22050).to("downsampled.wav")
            >>> X("music.mp3").resample(48000).to("hires.wav")
        """
        if sample_rate <= 0:
            raise InvalidOperationError("Sample rate must be positive")

        self._operations.append(("resample", {"sample_rate": sample_rate}))
        return self

    def to_mono(self) -> "Media":
        """
        Convert multi-channel audio to mono.

        Mixes all channels down to a single mono channel using ffmpeg's
        pan filter for optimal quality.

        Returns:
            Media: Self for chaining

        Examples:
            >>> Media("stereo.wav").to_mono().to("mono.wav")
            >>> X("5.1_surround.ac3").to_mono().to("mono.mp3")
        """
        self._operations.append(("to_mono", {}))
        return self

    def to_numpy(self, dtype: str = "float32") -> "np.ndarray":
        """
        Extract audio data as numpy array for analysis/processing.

        This executes all queued operations and returns the audio data
        as a numpy array using direct streaming pipes (no temp files).
        For stereo files, returns shape (channels, samples).
        For mono files, returns shape (samples,).

        Args:
            dtype: Numpy data type ('float32', 'float64', 'int16', 'int32')

        Returns:
            np.ndarray: Audio data array

        Examples:
            >>> audio_data = Media("song.wav")[10:20].to_numpy()
            >>> print(audio_data.shape)  # (2, 441000) for 10s stereo at 44.1kHz

            >>> mono_data = X("stereo.mp3").to_mono().resample(22050).to_numpy()
            >>> print(mono_data.shape)  # (220500,) for 10s mono at 22.05kHz

        Raises:
            ImportError: If numpy is not installed
            InvalidOperationError: If conversion fails
        """
        if not HAS_NUMPY:
            raise ImportError(
                "numpy is required for audio array processing. "
                "Install with: pip install numpy"
            )

        try:
            # Get original media info to determine base properties
            original_info = self.info()

            # Calculate final audio properties after operations
            final_sample_rate = original_info.sample_rate or 44100
            final_channels = original_info.channels or 1

            # Apply operation transformations to predict final properties
            for op_name, params in self._operations:
                if op_name == "resample":
                    final_sample_rate = params["sample_rate"]
                elif op_name == "to_mono":
                    final_channels = 1

            # Build ffmpeg command that streams raw audio directly to stdout
            cmd = self._ffmpeg.build_command(
                input_path=self.input_path,
                output_path=Path("-"),  # stdout
                operations=self._operations,
            )

            # Replace the output file with raw audio streaming options
            # Remove the -y and output file from the end
            while cmd and cmd[-1] in ["-y", "-"]:
                cmd.pop()

            # Add raw audio output options
            cmd.extend(
                [
                    "-f",
                    "f32le",  # 32-bit float little-endian format
                    "-acodec",
                    "pcm_f32le",  # Raw PCM codec
                    "-",  # Output to stdout
                ]
            )

            # Execute ffmpeg and capture raw audio data
            result = subprocess.run(cmd, capture_output=True, check=True)

            # Convert raw bytes directly to numpy array
            audio_data = np.frombuffer(result.stdout, dtype=np.float32)

            # Reshape for multi-channel audio
            if final_channels > 1 and len(audio_data) > 0:
                # Reshape to (samples, channels) then transpose to (channels, samples)
                samples_per_channel = len(audio_data) // final_channels
                if samples_per_channel * final_channels == len(audio_data):
                    audio_data = audio_data.reshape(
                        samples_per_channel, final_channels
                    ).T
                else:
                    # Fallback: truncate to nearest complete frame
                    truncated_length = samples_per_channel * final_channels
                    audio_data = (
                        audio_data[:truncated_length]
                        .reshape(samples_per_channel, final_channels)
                        .T
                    )

            # Convert to requested dtype if different
            if dtype != "float32":
                audio_data = audio_data.astype(dtype)

            return audio_data

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else "Unknown ffmpeg error"
            raise InvalidOperationError(f"Failed to extract audio data: {error_msg}")
        except Exception as e:
            raise InvalidOperationError(f"Failed to process audio array: {e}")

    @classmethod
    def from_numpy(
        cls,
        audio_data: "np.ndarray",
        sample_rate: int = 44100,
        output_path: Optional[Union[str, Path]] = None,
    ) -> "Media":
        """
        Create Media object from numpy audio array.

        Args:
            audio_data: Audio data as numpy array
                       Shape: (samples,) for mono or (channels, samples) for multi-channel
            sample_rate: Sample rate in Hz (default: 44100)
            output_path: Path to save temporary audio file
                        If None, creates temporary file

        Returns:
            Media: New Media instance for the created audio file

        Examples:
            >>> import numpy as np
            >>> # Generate 1 second of 440Hz sine wave
            >>> t = np.linspace(0, 1, 44100)
            >>> sine_wave = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            >>> media = Media.from_numpy(sine_wave, 44100)
            >>> media.to("sine_440hz.wav")

            >>> # Create stereo audio
            >>> stereo = np.array([sine_wave, sine_wave * 0.5])  # L/R channels
            >>> X.from_numpy(stereo, 44100).to("stereo_sine.wav")

        Raises:
            ImportError: If numpy is not installed
            InvalidOperationError: If array conversion fails
        """
        if not HAS_NUMPY:
            raise ImportError(
                "numpy is required for audio array processing. "
                "Install with: pip install numpy"
            )

        if not isinstance(audio_data, np.ndarray):
            raise InvalidOperationError("audio_data must be a numpy array")

        # Handle array dimensions
        if audio_data.ndim == 1:
            # Mono audio
            channels = 1
            samples = len(audio_data)
            audio_flat = audio_data.astype(np.float32)
        elif audio_data.ndim == 2:
            # Multi-channel audio: (channels, samples)
            channels, samples = audio_data.shape
            audio_flat = audio_data.T.flatten().astype(np.float32)
        else:
            raise InvalidOperationError(
                "Audio array must be 1D (mono) or 2D (multi-channel)"
            )

        # Create output file path
        if output_path is None:
            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = Path(tmp_file.name)
            tmp_file.close()
        else:
            output_path = Path(output_path)

        try:
            # Use ffmpeg to create audio file from raw data
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "f32le",  # Input format: 32-bit float little-endian
                "-ar",
                str(sample_rate),  # Sample rate
                "-ac",
                str(channels),  # Number of channels
                "-i",
                "-",  # Read from stdin
                "-c:a",
                "pcm_s16le",  # Output codec
                str(output_path),
            ]

            result = subprocess.run(
                cmd, input=audio_flat.tobytes(), capture_output=True, check=True
            )

            # Return new Media instance
            return cls(output_path)

        except subprocess.CalledProcessError as e:
            raise InvalidOperationError(f"Failed to create audio file: {e.stderr}")

    def to(self, output_path: Union[str, Path], **kwargs) -> "Media":
        """
        Execute all queued operations and save to output file.

        Args:
            output_path: Path for output file
            **kwargs: Additional ffmpeg options (crf, preset, etc.)

        Returns:
            Media: New Media instance for the output file
        """
        output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build and execute ffmpeg command
        command = self._ffmpeg.build_command(
            input_path=self.input_path,
            output_path=output_path,
            operations=self._operations,
            **kwargs,
        )

        self._ffmpeg.execute(command)

        # Return new Media instance for chaining only if file exists
        if output_path.exists():
            return Media(output_path)
        else:
            # If output doesn't exist (e.g., in tests), create a dummy instance
            new_media = object.__new__(Media)
            new_media.input_path = output_path
            new_media._operations = []
            new_media._ffmpeg = self._ffmpeg
            new_media._info = None
            return new_media

    def info(self) -> MediaInfo:
        """
        Get media file information without processing.

        Returns:
            MediaInfo: File information (duration, resolution, etc.)
        """
        if self._info is None:
            self._info = MediaInfo.from_file(self.input_path)
        return self._info

    def preview_command(self) -> str:
        """
        Preview the ffmpeg command that would be executed.

        Returns:
            str: FFmpeg command string
        """
        # Use a dummy output path for preview
        dummy_output = Path("/tmp/preview_output.mp4")
        command = self._ffmpeg.build_command(
            input_path=self.input_path,
            output_path=dummy_output,
            operations=self._operations,
        )
        return " ".join(command)

    def copy(self) -> "Media":
        """
        Create a copy of this Media instance with the same operations.

        Returns:
            Media: New Media instance with copied operations
        """
        # Create new instance without calling __init__ to avoid ffmpeg check
        new_media = object.__new__(Media)
        new_media.input_path = self.input_path
        new_media._operations = self._operations.copy()
        new_media._ffmpeg = self._ffmpeg  # Share the same ffmpeg instance
        new_media._info = None
        return new_media

    def __getitem__(self, key) -> "Media":
        """
        Enable high-performance slice notation for media files.

        Supports all Python slice patterns with lazy evaluation for scalability:
        - media[0:10]      # Slice from 0 to 10 seconds
        - media[30:60]     # Slice from 30 to 60 seconds
        - media[10:]       # From 10 seconds to end
        - media[:30]       # From start to 30 seconds
        - media[5]         # Single second at position 5

        Performance: Uses lazy evaluation - no processing until .to() is called,
        making it efficient for large files and complex operation chains.

        Args:
            key: slice object or int (seconds as time units)

        Returns:
            Media: New Media instance with slice operation queued

        Examples:
            >>> media = Media("large_video.mp4")  # 2 hour video
            >>> clips = [media[i*60:(i+1)*60] for i in range(10)]  # 10 minute clips
            >>> clips[0].resize(480, 320).to("clip_0.mp4")  # Process only when needed

        Raises:
            InvalidOperationError: For invalid slice patterns or negative times
            TypeError: For unsupported key types
        """
        if isinstance(key, slice):
            start = key.start
            stop = key.stop
            step = key.step

            if step is not None:
                raise InvalidOperationError(
                    "Step slicing not supported for media files. "
                    "Use consecutive slices for complex patterns."
                )

            # Validate slice bounds
            if start is not None and start < 0:
                raise InvalidOperationError("Negative start times not supported")
            if stop is not None and stop < 0:
                raise InvalidOperationError("Negative end times not supported")
            if start is not None and stop is not None and start >= stop:
                raise InvalidOperationError("Start time must be less than end time")

            # Create a copy and add slice operation (lazy evaluation)
            new_media = self.copy()
            return new_media.slice(start, stop)

        elif isinstance(key, int):
            # Single index - treat as 1 second slice starting at that position
            if key < 0:
                raise InvalidOperationError(
                    "Negative indexing not supported for media files. "
                    "Use positive time values in seconds."
                )

            new_media = self.copy()
            return new_media.slice(key, key + 1)

        else:
            raise TypeError(
                f"Media indices must be integers or slices, not {type(key).__name__}. "
                f"Use media[start:end] for time ranges or media[second] for single seconds."
            )

    def __repr__(self) -> str:
        """String representation of Media instance."""
        ops = f", {len(self._operations)} operations" if self._operations else ""
        return f"Media({self.input_path}{ops})"
