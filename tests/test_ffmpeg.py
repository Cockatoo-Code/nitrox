"""Tests for FFmpeg wrapper functionality."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from nitrox.exceptions import FFmpegError
from nitrox.ffmpeg import FFmpegRunner


class TestFFmpegRunner:
    """Test FFmpeg runner initialization and checks."""

    @patch("nitrox.ffmpeg.shutil.which")
    def test_init_with_ffmpeg_available(self, mock_which):
        """Test initialization when ffmpeg is available."""
        mock_which.return_value = "/usr/bin/ffmpeg"

        runner = FFmpegRunner()
        assert runner is not None
        mock_which.assert_called_once_with("ffmpeg")

    @patch("nitrox.ffmpeg.shutil.which")
    def test_init_without_ffmpeg(self, mock_which):
        """Test initialization when ffmpeg is not available."""
        mock_which.return_value = None

        with pytest.raises(FFmpegError, match="ffmpeg not found in PATH"):
            FFmpegRunner()


class TestCommandBuilding:
    """Test FFmpeg command building."""

    @pytest.fixture
    def runner(self):
        """Create FFmpeg runner with mocked availability check."""
        with patch("nitrox.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg"):
            return FFmpegRunner()

    def test_basic_command_structure(self, runner):
        """Test basic command structure."""
        input_path = Path("input.mp4")
        output_path = Path("output.mp4")
        operations = []

        cmd = runner.build_command(input_path, output_path, operations)

        assert cmd[0] == "ffmpeg"
        assert "-hide_banner" in cmd
        assert "-loglevel" in cmd
        assert "error" in cmd
        assert "-i" in cmd
        assert str(input_path) in cmd
        assert "-y" in cmd
        assert str(output_path) in cmd

    def test_slice_operation_with_start_and_end(self, runner):
        """Test slice operation with both start and end times."""
        input_path = Path("input.mp4")
        output_path = Path("output.mp4")
        operations = [("slice", {"start": 10, "end": 30})]

        cmd = runner.build_command(input_path, output_path, operations)

        ss_index = cmd.index("-ss")
        assert cmd[ss_index + 1] == "10"

        t_index = cmd.index("-t")
        assert cmd[t_index + 1] == "20"  # duration = end - start

    def test_slice_operation_start_only(self, runner):
        """Test slice operation with start time only."""
        input_path = Path("input.mp4")
        output_path = Path("output.mp4")
        operations = [("slice", {"start": 10, "end": None})]

        cmd = runner.build_command(input_path, output_path, operations)

        ss_index = cmd.index("-ss")
        assert cmd[ss_index + 1] == "10"
        assert "-t" not in cmd

    def test_slice_operation_end_only(self, runner):
        """Test slice operation with end time only."""
        input_path = Path("input.mp4")
        output_path = Path("output.mp4")
        operations = [("slice", {"start": None, "end": 30})]

        cmd = runner.build_command(input_path, output_path, operations)

        to_index = cmd.index("-to")
        assert cmd[to_index + 1] == "30"
        assert "-ss" not in cmd

    def test_resize_operation(self, runner):
        """Test resize operation."""
        input_path = Path("input.mp4")
        output_path = Path("output.mp4")
        operations = [("resize", {"width": 720, "height": 480})]

        cmd = runner.build_command(input_path, output_path, operations)

        vf_index = cmd.index("-vf")
        assert cmd[vf_index + 1] == "scale=720:480"

    def test_crop_operation(self, runner):
        """Test crop operation."""
        input_path = Path("input.mp4")
        output_path = Path("output.mp4")
        operations = [("crop", {"x": 10, "y": 20, "width": 640, "height": 360})]

        cmd = runner.build_command(input_path, output_path, operations)

        vf_index = cmd.index("-vf")
        assert cmd[vf_index + 1] == "crop=640:360:10:20"

    def test_extract_audio_operation(self, runner):
        """Test extract audio operation."""
        input_path = Path("input.mp4")
        output_path = Path("output.mp3")
        operations = [("extract_audio", {})]

        cmd = runner.build_command(input_path, output_path, operations)

        assert "-vn" in cmd

    def test_normalize_audio_operation(self, runner):
        """Test normalize audio operation."""
        input_path = Path("input.mp4")
        output_path = Path("output.mp4")
        operations = [("normalize_audio", {})]

        cmd = runner.build_command(input_path, output_path, operations)

        af_index = cmd.index("-af")
        assert cmd[af_index + 1] == "loudnorm"

    def test_set_fps_operation(self, runner):
        """Test set FPS operation."""
        input_path = Path("input.mp4")
        output_path = Path("output.mp4")
        operations = [("set_fps", {"fps": 30.0})]

        cmd = runner.build_command(input_path, output_path, operations)

        vf_index = cmd.index("-vf")
        assert cmd[vf_index + 1] == "fps=30.0"

    def test_fade_operations(self, runner):
        """Test fade in/out operations."""
        input_path = Path("input.mp4")
        output_path = Path("output.mp4")
        operations = [("fade_in", {"duration": 2.0}), ("fade_out", {"duration": 1.5})]

        cmd = runner.build_command(input_path, output_path, operations)

        af_index = cmd.index("-af")
        assert "afade=t=in:d=2.0" in cmd[af_index + 1]
        assert "afade=t=out:d=1.5" in cmd[af_index + 1]

    def test_resample_operation(self, runner):
        """Test audio resampling operation."""
        input_path = Path("input.mp4")
        output_path = Path("output.mp4")
        operations = [("resample", {"sample_rate": 22050})]

        cmd = runner.build_command(input_path, output_path, operations)

        af_index = cmd.index("-af")
        assert cmd[af_index + 1] == "aresample=22050"

    def test_to_mono_operation(self, runner):
        """Test mono conversion operation."""
        input_path = Path("input.mp4")
        output_path = Path("output.mp4")
        operations = [("to_mono", {})]

        cmd = runner.build_command(input_path, output_path, operations)

        af_index = cmd.index("-af")
        assert cmd[af_index + 1] == "pan=mono|c0=0.5*c0+0.5*c1"

    def test_combined_audio_operations(self, runner):
        """Test combining multiple audio operations."""
        input_path = Path("input.mp4")
        output_path = Path("output.mp4")
        operations = [
            ("resample", {"sample_rate": 16000}),
            ("to_mono", {}),
            ("normalize_audio", {}),
            ("fade_in", {"duration": 1.0}),
        ]

        cmd = runner.build_command(input_path, output_path, operations)

        af_index = cmd.index("-af")
        filters = cmd[af_index + 1]

        # Should contain all audio filters
        assert "aresample=16000" in filters
        assert "pan=mono|c0=0.5*c0+0.5*c1" in filters
        assert "loudnorm" in filters
        assert "afade=t=in:d=1.0" in filters

        # Should be comma-separated
        assert filters.count(",") == 3  # Four filters separated by commas

    def test_multiple_video_filters(self, runner):
        """Test multiple video filters are combined."""
        input_path = Path("input.mp4")
        output_path = Path("output.mp4")
        operations = [
            ("resize", {"width": 720, "height": 480}),
            ("crop", {"x": 10, "y": 20, "width": 640, "height": 360}),
            ("set_fps", {"fps": 30.0}),
        ]

        cmd = runner.build_command(input_path, output_path, operations)

        vf_index = cmd.index("-vf")
        filters = cmd[vf_index + 1]
        assert "scale=720:480" in filters
        assert "crop=640:360:10:20" in filters
        assert "fps=30.0" in filters
        assert filters.count(",") == 2  # Three filters separated by commas


class TestEncodingOptions:
    """Test encoding options for different formats."""

    @pytest.fixture
    def runner(self):
        """Create FFmpeg runner with mocked availability check."""
        with patch("nitrox.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg"):
            return FFmpegRunner()

    def test_mp4_encoding_options(self, runner):
        """Test MP4 encoding options."""
        input_path = Path("input.mp4")
        output_path = Path("output.mp4")
        operations = []

        cmd = runner.build_command(input_path, output_path, operations)

        assert "-c:v" in cmd
        assert "libx264" in cmd
        assert "-crf" in cmd
        assert "23" in cmd
        assert "-preset" in cmd
        assert "medium" in cmd

    def test_mp3_encoding_options(self, runner):
        """Test MP3 encoding options."""
        input_path = Path("input.mp4")
        output_path = Path("output.mp3")
        operations = []

        cmd = runner.build_command(input_path, output_path, operations)

        assert "-c:a" in cmd
        assert "libmp3lame" in cmd
        assert "-b:a" in cmd
        assert "192k" in cmd

    def test_wav_encoding_options(self, runner):
        """Test WAV encoding options."""
        input_path = Path("input.mp4")
        output_path = Path("output.wav")
        operations = []

        cmd = runner.build_command(input_path, output_path, operations)

        assert "-c:a" in cmd
        assert "pcm_s16le" in cmd

    def test_custom_encoding_options(self, runner):
        """Test custom encoding options override defaults."""
        input_path = Path("input.mp4")
        output_path = Path("output.mp4")
        operations = []

        cmd = runner.build_command(
            input_path, output_path, operations, crf=18, preset="fast", bitrate="256k"
        )

        crf_index = cmd.index("-crf")
        assert cmd[crf_index + 1] == "18"

        preset_index = cmd.index("-preset")
        assert cmd[preset_index + 1] == "fast"


class TestCommandExecution:
    """Test command execution and error handling."""

    @pytest.fixture
    def runner(self):
        """Create FFmpeg runner with mocked availability check."""
        with patch("nitrox.ffmpeg.shutil.which", return_value="/usr/bin/ffmpeg"):
            return FFmpegRunner()

    @patch("nitrox.ffmpeg.subprocess.run")
    def test_successful_execution(self, mock_run, runner):
        """Test successful command execution."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        command = ["ffmpeg", "-i", "input.mp4", "output.mp4"]
        runner.execute(command)

        mock_run.assert_called_once_with(
            command, capture_output=True, text=True, check=True
        )

    @patch("nitrox.ffmpeg.subprocess.run")
    def test_execution_failure(self, mock_run, runner):
        """Test command execution failure."""
        error = subprocess.CalledProcessError(1, "ffmpeg", stderr="Error message")
        mock_run.side_effect = error

        command = ["ffmpeg", "-i", "input.mp4", "output.mp4"]

        with pytest.raises(FFmpegError) as exc_info:
            runner.execute(command)

        assert "FFmpeg command failed" in str(exc_info.value)
        assert exc_info.value.command == command
        assert exc_info.value.returncode == 1
        assert exc_info.value.stderr == "Error message"

    @patch("nitrox.ffmpeg.subprocess.run")
    def test_execution_file_not_found(self, mock_run, runner):
        """Test execution when ffmpeg binary not found."""
        mock_run.side_effect = FileNotFoundError()

        command = ["ffmpeg", "-i", "input.mp4", "output.mp4"]

        with pytest.raises(FFmpegError, match="FFmpeg binary not found"):
            runner.execute(command)
