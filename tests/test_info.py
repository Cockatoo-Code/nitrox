"""Tests for MediaInfo functionality."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from nitrox.exceptions import FFmpegError
from nitrox.info import MediaInfo


class TestMediaInfoCreation:
    """Test MediaInfo creation and initialization."""

    def test_init_with_probe_data(self):
        """Test initialization with probe data."""
        probe_data = {
            "format": {"duration": "120.5", "size": "1048576", "bit_rate": "1000000"},
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                    "codec_name": "h264",
                },
                {
                    "codec_type": "audio",
                    "sample_rate": "44100",
                    "channels": 2,
                    "codec_name": "aac",
                },
            ],
        }

        info = MediaInfo(probe_data)
        assert info._data == probe_data
        assert info._format == probe_data["format"]
        assert info._streams == probe_data["streams"]

    @patch("nitrox.info.shutil.which")
    @patch("nitrox.info.subprocess.run")
    def test_from_file_success(self, mock_run, mock_which):
        """Test successful creation from file."""
        mock_which.return_value = "/usr/bin/ffprobe"

        probe_output = {
            "format": {"duration": "60.0"},
            "streams": [{"codec_type": "video", "codec_name": "h264"}],
        }

        mock_run.return_value = Mock(
            stdout=json.dumps(probe_output), stderr="", returncode=0
        )

        file_path = Path("test.mp4")
        info = MediaInfo.from_file(file_path)

        assert isinstance(info, MediaInfo)
        mock_run.assert_called_once()

        # Check command structure
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "ffprobe"
        assert str(file_path) in call_args

    @patch("nitrox.info.shutil.which")
    def test_from_file_ffprobe_not_found(self, mock_which):
        """Test error when ffprobe is not available."""
        mock_which.return_value = None

        with pytest.raises(FFmpegError, match="ffprobe not found in PATH"):
            MediaInfo.from_file(Path("test.mp4"))

    @patch("nitrox.info.shutil.which")
    @patch("nitrox.info.subprocess.run")
    def test_from_file_command_failure(self, mock_run, mock_which):
        """Test error when ffprobe command fails."""
        mock_which.return_value = "/usr/bin/ffprobe"

        error = subprocess.CalledProcessError(1, "ffprobe", stderr="File not found")
        mock_run.side_effect = error

        with pytest.raises(FFmpegError, match="ffprobe failed"):
            MediaInfo.from_file(Path("test.mp4"))

    @patch("nitrox.info.shutil.which")
    @patch("nitrox.info.subprocess.run")
    def test_from_file_json_parse_error(self, mock_run, mock_which):
        """Test error when ffprobe output is invalid JSON."""
        mock_which.return_value = "/usr/bin/ffprobe"

        mock_run.return_value = Mock(stdout="invalid json", stderr="", returncode=0)

        with pytest.raises(FFmpegError, match="Failed to parse ffprobe output"):
            MediaInfo.from_file(Path("test.mp4"))


class TestMediaInfoProperties:
    """Test MediaInfo property access."""

    @pytest.fixture
    def video_info(self):
        """Create MediaInfo with video data."""
        probe_data = {
            "format": {
                "duration": "120.5",
                "size": "1048576",
                "bit_rate": "1000000",
                "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            },
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                    "codec_name": "h264",
                },
                {
                    "codec_type": "audio",
                    "sample_rate": "44100",
                    "channels": 2,
                    "codec_name": "aac",
                },
            ],
        }
        return MediaInfo(probe_data)

    @pytest.fixture
    def audio_only_info(self):
        """Create MediaInfo with audio-only data."""
        probe_data = {
            "format": {"duration": "180.0", "size": "5242880", "bit_rate": "192000"},
            "streams": [
                {
                    "codec_type": "audio",
                    "sample_rate": "44100",
                    "channels": 2,
                    "codec_name": "mp3",
                }
            ],
        }
        return MediaInfo(probe_data)

    def test_duration_property(self, video_info):
        """Test duration property."""
        assert video_info.duration == 120.5

    def test_file_size_property(self, video_info):
        """Test file size property."""
        assert video_info.file_size == 1048576

    def test_bit_rate_property(self, video_info):
        """Test bit rate property."""
        assert video_info.bit_rate == 1000000

    def test_format_name_property(self, video_info):
        """Test format name property."""
        assert video_info.format_name == "mov,mp4,m4a,3gp,3g2,mj2"

    def test_has_video_property(self, video_info, audio_only_info):
        """Test has_video property."""
        assert video_info.has_video is True
        assert audio_only_info.has_video is False

    def test_has_audio_property(self, video_info, audio_only_info):
        """Test has_audio property."""
        assert video_info.has_audio is True
        assert audio_only_info.has_audio is True

    def test_video_dimensions(self, video_info):
        """Test video width and height properties."""
        assert video_info.width == 1920
        assert video_info.height == 1080

    def test_fps_property(self, video_info):
        """Test FPS property calculation."""
        assert video_info.fps == 30.0

    def test_video_codec_property(self, video_info):
        """Test video codec property."""
        assert video_info.video_codec == "h264"

    def test_audio_properties(self, video_info):
        """Test audio-related properties."""
        assert video_info.audio_codec == "aac"
        assert video_info.sample_rate == 44100
        assert video_info.channels == 2

    def test_audio_only_properties(self, audio_only_info):
        """Test properties for audio-only files."""
        assert audio_only_info.width is None
        assert audio_only_info.height is None
        assert audio_only_info.fps is None
        assert audio_only_info.video_codec is None
        assert audio_only_info.audio_codec == "mp3"


class TestMediaInfoEdgeCases:
    """Test edge cases and missing data."""

    def test_missing_format_data(self):
        """Test handling of missing format data."""
        probe_data = {"streams": []}
        info = MediaInfo(probe_data)

        assert info.duration is None
        assert info.file_size is None
        assert info.bit_rate is None
        assert info.format_name is None

    def test_missing_stream_data(self):
        """Test handling of missing stream data."""
        probe_data = {"format": {"duration": "60.0"}}
        info = MediaInfo(probe_data)

        assert info.has_video is False
        assert info.has_audio is False
        assert info.width is None
        assert info.height is None
        assert info.fps is None

    def test_fps_calculation_edge_cases(self):
        """Test FPS calculation with edge case values."""
        # Test division by zero
        probe_data = {"streams": [{"codec_type": "video", "r_frame_rate": "30/0"}]}
        info = MediaInfo(probe_data)
        assert info.fps is None

        # Test missing fps data
        probe_data = {"streams": [{"codec_type": "video"}]}
        info = MediaInfo(probe_data)
        assert info.fps is None

    def test_multiple_streams_same_type(self):
        """Test handling multiple streams of same type (uses first one)."""
        probe_data = {
            "streams": [
                {"codec_type": "video", "width": 1920, "codec_name": "h264"},
                {"codec_type": "video", "width": 1280, "codec_name": "h265"},
            ]
        }
        info = MediaInfo(probe_data)

        # Should use first stream
        assert info.width == 1920
        assert info.video_codec == "h264"


class TestMediaInfoRepr:
    """Test string representation."""

    def test_repr_with_video_data(self):
        """Test repr with complete video data."""
        probe_data = {
            "format": {"duration": "120.5"},
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                },
                {"codec_type": "audio", "channels": 2},
            ],
        }
        info = MediaInfo(probe_data)

        repr_str = repr(info)
        assert "duration=120.50s" in repr_str
        assert "resolution=1920x1080" in repr_str
        assert "fps=30.00" in repr_str
        assert "audio=2ch" in repr_str

    def test_repr_with_audio_only(self):
        """Test repr with audio-only data."""
        probe_data = {
            "format": {"duration": "180.0"},
            "streams": [{"codec_type": "audio", "channels": 1}],
        }
        info = MediaInfo(probe_data)

        repr_str = repr(info)
        assert "duration=180.00s" in repr_str
        assert "audio=1ch" in repr_str
        assert "resolution=" not in repr_str

    def test_repr_with_no_data(self):
        """Test repr with minimal data."""
        probe_data = {}
        info = MediaInfo(probe_data)

        repr_str = repr(info)
        assert repr_str == "MediaInfo(no info)"
