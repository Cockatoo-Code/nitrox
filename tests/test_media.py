"""Tests for the core Media class."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Test both Media and X alias
from nitrox import Media, X
from nitrox.exceptions import FFmpegError, InvalidOperationError, MediaNotFoundError


class TestXAlias:
    """Test the X alias functionality."""

    @pytest.fixture
    def test_file(self, tmp_path):
        """Create a test file."""
        test_file = tmp_path / "test.mp4"
        test_file.touch()
        return test_file

    @patch("nitrox.core.FFmpegRunner")
    def test_x_is_media_alias(self, mock_ffmpeg_class, test_file):
        """Test that X is an alias for Media."""
        mock_ffmpeg = Mock()
        mock_ffmpeg_class.return_value = mock_ffmpeg

        # X should be the same class as Media
        assert X is Media

        # Create instances
        media = Media(test_file)
        x = X(test_file)

        # Both should be instances of the same class
        assert type(media) == type(x)
        assert isinstance(x, Media)

    @patch("nitrox.core.FFmpegRunner")
    def test_x_functionality_identical(self, mock_ffmpeg_class, test_file):
        """Test that X has identical functionality to Media."""
        mock_ffmpeg = Mock()
        mock_ffmpeg_class.return_value = mock_ffmpeg

        # Create instances
        media = Media(test_file)
        x = X(test_file)

        # Test method availability
        assert hasattr(x, "slice")
        assert hasattr(x, "resize")
        assert hasattr(x, "extract_audio")
        assert hasattr(x, "to")
        assert hasattr(x, "__getitem__")  # Slice notation

        # Test operations work identically
        media_result = media.slice(10, 30).resize(720, 480)
        x_result = x.slice(10, 30).resize(720, 480)

        assert media_result._operations == x_result._operations

    @patch("nitrox.core.FFmpegRunner")
    def test_x_slice_notation(self, mock_ffmpeg_class, test_file):
        """Test that X supports slice notation."""
        mock_ffmpeg = Mock()
        mock_ffmpeg_class.return_value = mock_ffmpeg

        x = X(test_file)

        # Test basic slice notation
        result = x[10:30]
        assert isinstance(result, Media)  # Should return Media instance
        assert result._operations == [("slice", {"start": 10, "end": 30})]

        # Test chaining with slice notation
        result = x[5:15].resize(640, 480)
        expected_ops = [
            ("slice", {"start": 5, "end": 15}),
            ("resize", {"width": 640, "height": 480}),
        ]
        assert result._operations == expected_ops

    @patch("nitrox.core.FFmpegRunner")
    def test_x_vs_media_equivalence(self, mock_ffmpeg_class, test_file):
        """Test that X and Media produce identical results."""
        mock_ffmpeg = Mock()
        mock_ffmpeg_class.return_value = mock_ffmpeg

        # Same operations with Media and X
        media_result = Media(test_file)[10:30].resize(720, 480)
        x_result = X(test_file)[10:30].resize(720, 480)

        # Should have identical operations
        assert media_result._operations == x_result._operations
        assert media_result.input_path == x_result.input_path

        # Test traditional vs slice notation equivalence
        media_traditional = Media(test_file).slice(10, 30)
        x_slice = X(test_file)[10:30]

        assert media_traditional._operations == x_slice._operations

    @patch("nitrox.core.FFmpegRunner")
    def test_x_error_handling(self, mock_ffmpeg_class, test_file):
        """Test that X has the same error handling as Media."""
        mock_ffmpeg = Mock()
        mock_ffmpeg_class.return_value = mock_ffmpeg

        x = X(test_file)

        # Test slice notation error cases
        with pytest.raises(
            InvalidOperationError, match="Negative start times not supported"
        ):
            x[-5:10]

        with pytest.raises(
            InvalidOperationError, match="Start time must be less than end time"
        ):
            x[30:10]

        with pytest.raises(TypeError, match="Media indices must be integers or slices"):
            x["invalid"]

    def test_x_import_both_styles(self):
        """Test that both Media and X can be imported together."""
        # This test verifies the import works
        from nitrox import Media, X

        assert Media is not None
        assert X is not None
        assert X is Media  # They should be the same class


class TestMediaInit:
    """Test Media class initialization."""

    @patch("nitrox.core.FFmpegRunner")
    def test_init_with_existing_file(self, mock_ffmpeg_class, tmp_path):
        """Test initialization with existing file."""
        test_file = tmp_path / "test.mp4"
        test_file.touch()

        mock_ffmpeg = Mock()
        mock_ffmpeg_class.return_value = mock_ffmpeg

        media = Media(test_file)
        assert media.input_path == test_file
        assert media._operations == []

    def test_init_with_nonexistent_file(self):
        """Test initialization with non-existent file raises error."""
        with pytest.raises(MediaNotFoundError, match="Media file not found"):
            Media("nonexistent.mp4")

    @patch("nitrox.core.FFmpegRunner")
    def test_init_with_string_path(self, mock_ffmpeg_class, tmp_path):
        """Test initialization with string path."""
        test_file = tmp_path / "test.mp4"
        test_file.touch()

        mock_ffmpeg = Mock()
        mock_ffmpeg_class.return_value = mock_ffmpeg

        media = Media(str(test_file))
        assert media.input_path == test_file


class TestMediaOperations:
    """Test Media operation methods."""

    @pytest.fixture
    def media(self, tmp_path):
        """Create test media instance."""
        test_file = tmp_path / "test.mp4"
        test_file.touch()

        with patch("nitrox.core.FFmpegRunner") as mock_ffmpeg_class:
            mock_ffmpeg = Mock()
            mock_ffmpeg_class.return_value = mock_ffmpeg
            return Media(test_file)

    def test_slice_valid_params(self, media):
        """Test slice with valid parameters."""
        result = media.slice(10, 30)
        assert result is media  # Returns self for chaining
        assert media._operations == [("slice", {"start": 10, "end": 30})]

    def test_slice_start_only(self, media):
        """Test slice with start time only."""
        media.slice(10, None)
        assert media._operations == [("slice", {"start": 10, "end": None})]

    def test_slice_end_only(self, media):
        """Test slice with end time only."""
        media.slice(None, 30)
        assert media._operations == [("slice", {"start": None, "end": 30})]

    def test_slice_invalid_negative_start(self, media):
        """Test slice with negative start time."""
        with pytest.raises(
            InvalidOperationError, match="Start time cannot be negative"
        ):
            media.slice(-5, 10)

    def test_slice_invalid_negative_end(self, media):
        """Test slice with negative end time."""
        with pytest.raises(InvalidOperationError, match="End time cannot be negative"):
            media.slice(5, -10)

    def test_slice_invalid_start_after_end(self, media):
        """Test slice with start time after end time."""
        with pytest.raises(
            InvalidOperationError, match="Start time must be less than end time"
        ):
            media.slice(30, 10)

    def test_resize_valid_params(self, media):
        """Test resize with valid parameters."""
        result = media.resize(720, 480)
        assert result is media
        assert media._operations == [("resize", {"width": 720, "height": 480})]

    def test_resize_invalid_dimensions(self, media):
        """Test resize with invalid dimensions."""
        with pytest.raises(
            InvalidOperationError, match="Width and height must be positive"
        ):
            media.resize(0, 480)

        with pytest.raises(
            InvalidOperationError, match="Width and height must be positive"
        ):
            media.resize(720, -480)

    def test_crop_valid_params(self, media):
        """Test crop with valid parameters."""
        result = media.crop(10, 20, 640, 360)
        assert result is media
        assert media._operations == [
            ("crop", {"x": 10, "y": 20, "width": 640, "height": 360})
        ]

    def test_crop_invalid_params(self, media):
        """Test crop with invalid parameters."""
        with pytest.raises(
            InvalidOperationError, match="Crop width and height must be positive"
        ):
            media.crop(10, 20, 0, 360)

        with pytest.raises(
            InvalidOperationError, match="Crop offsets cannot be negative"
        ):
            media.crop(-10, 20, 640, 360)

    def test_extract_audio(self, media):
        """Test extract audio operation."""
        result = media.extract_audio()
        assert result is media
        assert media._operations == [("extract_audio", {})]

    def test_normalize_audio(self, media):
        """Test normalize audio operation."""
        result = media.normalize_audio()
        assert result is media
        assert media._operations == [("normalize_audio", {})]

    def test_set_fps(self, media):
        """Test set FPS operation."""
        result = media.set_fps(30.0)
        assert result is media
        assert media._operations == [("set_fps", {"fps": 30.0})]

    def test_set_fps_invalid(self, media):
        """Test set FPS with invalid value."""
        with pytest.raises(InvalidOperationError, match="FPS must be positive"):
            media.set_fps(0)

    def test_fade_in(self, media):
        """Test fade in operation."""
        result = media.fade_in(2.0)
        assert result is media
        assert media._operations == [("fade_in", {"duration": 2.0})]

    def test_fade_out(self, media):
        """Test fade out operation."""
        result = media.fade_out(2.0)
        assert result is media
        assert media._operations == [("fade_out", {"duration": 2.0})]

    def test_fade_invalid_duration(self, media):
        """Test fade with invalid duration."""
        with pytest.raises(
            InvalidOperationError, match="Fade duration must be positive"
        ):
            media.fade_in(0)


class TestMediaChaining:
    """Test operation chaining."""

    @pytest.fixture
    def media(self, tmp_path):
        """Create test media instance."""
        test_file = tmp_path / "test.mp4"
        test_file.touch()

        with patch("nitrox.core.FFmpegRunner") as mock_ffmpeg_class:
            mock_ffmpeg = Mock()
            mock_ffmpeg_class.return_value = mock_ffmpeg
            return Media(test_file)

    def test_operation_chaining(self, media):
        """Test chaining multiple operations."""
        result = media.slice(10, 30).resize(720, 480).extract_audio().normalize_audio()

        assert result is media
        assert len(media._operations) == 4
        assert media._operations[0] == ("slice", {"start": 10, "end": 30})
        assert media._operations[1] == ("resize", {"width": 720, "height": 480})
        assert media._operations[2] == ("extract_audio", {})
        assert media._operations[3] == ("normalize_audio", {})

    def test_copy_method(self, media):
        """Test copying media instance with operations."""
        media.slice(10, 30).resize(720, 480)

        copy = media.copy()

        assert copy is not media
        assert copy.input_path == media.input_path
        assert copy._operations == media._operations
        assert copy._operations is not media._operations  # Different list instance


class TestMediaExecution:
    """Test media execution and output."""

    @pytest.fixture
    def media(self, tmp_path):
        """Create test media instance."""
        test_file = tmp_path / "test.mp4"
        test_file.touch()

        with patch("nitrox.core.FFmpegRunner") as mock_ffmpeg_class:
            mock_ffmpeg = Mock()
            mock_ffmpeg_class.return_value = mock_ffmpeg
            return Media(test_file)

    @patch("nitrox.core.FFmpegRunner")
    def test_to_method_execution(self, mock_ffmpeg_class, media, tmp_path):
        """Test the to() method executes ffmpeg correctly."""
        mock_ffmpeg = Mock()
        mock_ffmpeg_class.return_value = mock_ffmpeg
        mock_ffmpeg.build_command.return_value = [
            "ffmpeg",
            "-i",
            "input.mp4",
            "output.mp4",
        ]

        # Replace the ffmpeg instance
        media._ffmpeg = mock_ffmpeg

        output_path = tmp_path / "output.mp4"
        result = media.slice(10, 30).to(output_path)

        # Check ffmpeg was called correctly
        mock_ffmpeg.build_command.assert_called_once()
        mock_ffmpeg.execute.assert_called_once()

        # Should return new Media instance
        assert isinstance(result, Media)

    @patch("nitrox.core.FFmpegRunner")
    def test_preview_command(self, mock_ffmpeg_class, media):
        """Test preview command generation."""
        mock_ffmpeg = Mock()
        mock_ffmpeg_class.return_value = mock_ffmpeg
        mock_ffmpeg.build_command.return_value = [
            "ffmpeg",
            "-i",
            "test.mp4",
            "output.mp4",
        ]

        media._ffmpeg = mock_ffmpeg

        command = media.slice(10, 30).preview_command()

        assert command == "ffmpeg -i test.mp4 output.mp4"
        mock_ffmpeg.build_command.assert_called_once()


class TestMediaInfo:
    """Test media info functionality."""

    @pytest.fixture
    def media(self, tmp_path):
        """Create test media instance."""
        test_file = tmp_path / "test.mp4"
        test_file.touch()

        with patch("nitrox.core.FFmpegRunner") as mock_ffmpeg_class:
            mock_ffmpeg = Mock()
            mock_ffmpeg_class.return_value = mock_ffmpeg
            return Media(test_file)

    @patch("nitrox.info.MediaInfo.from_file")
    def test_info_method(self, mock_from_file, media):
        """Test info method returns MediaInfo."""
        mock_info = Mock()
        mock_from_file.return_value = mock_info

        result = media.info()

        assert result is mock_info
        mock_from_file.assert_called_once_with(media.input_path)

    @patch("nitrox.info.MediaInfo.from_file")
    def test_info_caching(self, mock_from_file, media):
        """Test info result is cached."""
        mock_info = Mock()
        mock_from_file.return_value = mock_info

        # Call info twice
        result1 = media.info()
        result2 = media.info()

        assert result1 is result2
        mock_from_file.assert_called_once()  # Should only be called once


class TestMediaRepr:
    """Test string representation."""

    @patch("nitrox.core.FFmpegRunner")
    def test_repr_no_operations(self, mock_ffmpeg_class, tmp_path):
        """Test repr with no operations."""
        test_file = tmp_path / "test.mp4"
        test_file.touch()

        mock_ffmpeg = Mock()
        mock_ffmpeg_class.return_value = mock_ffmpeg

        media = Media(test_file)
        assert repr(media) == f"Media({test_file})"

    @patch("nitrox.core.FFmpegRunner")
    def test_repr_with_operations(self, mock_ffmpeg_class, tmp_path):
        """Test repr with operations."""
        test_file = tmp_path / "test.mp4"
        test_file.touch()

        mock_ffmpeg = Mock()
        mock_ffmpeg_class.return_value = mock_ffmpeg

        media = Media(test_file)
        media.slice(10, 30).resize(720, 480)

        assert repr(media) == f"Media({test_file}, 2 operations)"


class TestMediaSliceNotation:
    """Test slice notation (__getitem__) functionality."""

    @pytest.fixture
    def media(self, tmp_path):
        """Create test media instance."""
        test_file = tmp_path / "test.mp4"
        test_file.touch()

        with patch("nitrox.core.FFmpegRunner") as mock_ffmpeg_class:
            mock_ffmpeg = Mock()
            mock_ffmpeg_class.return_value = mock_ffmpeg
            return Media(test_file)

    def test_basic_slice_notation(self, media):
        """Test basic slice notation patterns."""
        # Test [start:end]
        result = media[0:10]
        assert isinstance(result, Media)
        assert result is not media  # Should be a copy
        assert result._operations == [("slice", {"start": 0, "end": 10})]

        # Test [start:end] with different values
        result = media[30:60]
        assert result._operations == [("slice", {"start": 30, "end": 60})]

    def test_open_ended_slices(self, media):
        """Test open-ended slice patterns."""
        # Test [start:] (to end)
        result = media[10:]
        assert result._operations == [("slice", {"start": 10, "end": None})]

        # Test [:end] (from start)
        result = media[:30]
        assert result._operations == [("slice", {"start": None, "end": 30})]

        # Test [:] (entire file)
        result = media[:]
        assert result._operations == [("slice", {"start": None, "end": None})]

    def test_single_index_notation(self, media):
        """Test single index notation."""
        # Test [5] -> slice from 5 to 6 seconds
        result = media[5]
        assert result._operations == [("slice", {"start": 5, "end": 6})]

        # Test [0] -> slice from 0 to 1 second
        result = media[0]
        assert result._operations == [("slice", {"start": 0, "end": 1})]

    def test_slice_chaining_with_operations(self, media):
        """Test chaining slice notation with other operations."""
        # Chain slice notation with resize
        result = media[10:30].resize(720, 480)
        expected_ops = [
            ("slice", {"start": 10, "end": 30}),
            ("resize", {"width": 720, "height": 480}),
        ]
        assert result._operations == expected_ops

        # Chain multiple operations
        result = media[5:15].resize(640, 360).extract_audio().normalize_audio()
        assert len(result._operations) == 4
        assert result._operations[0] == ("slice", {"start": 5, "end": 15})
        assert result._operations[1] == ("resize", {"width": 640, "height": 360})
        assert result._operations[2] == ("extract_audio", {})
        assert result._operations[3] == ("normalize_audio", {})

    def test_slice_notation_independence(self, media):
        """Test that slice notation creates independent copies."""
        clip1 = media[0:10]
        clip2 = media[20:30]

        # Modify one clip
        clip1.resize(720, 480)

        # Other clip should be unaffected
        assert clip1._operations == [
            ("slice", {"start": 0, "end": 10}),
            ("resize", {"width": 720, "height": 480}),
        ]
        assert clip2._operations == [("slice", {"start": 20, "end": 30})]

        # Original should be unaffected
        assert media._operations == []

    def test_slice_notation_error_cases(self, media):
        """Test error cases for slice notation."""
        # Test step slicing (not supported)
        with pytest.raises(InvalidOperationError, match="Step slicing not supported"):
            media[0:10:2]

        # Test negative start
        with pytest.raises(
            InvalidOperationError, match="Negative start times not supported"
        ):
            media[-5:10]

        # Test negative end
        with pytest.raises(
            InvalidOperationError, match="Negative end times not supported"
        ):
            media[5:-10]

        # Test start >= end
        with pytest.raises(
            InvalidOperationError, match="Start time must be less than end time"
        ):
            media[30:10]

        # Test negative single index
        with pytest.raises(
            InvalidOperationError, match="Negative indexing not supported"
        ):
            media[-1]

        # Test invalid key type
        with pytest.raises(TypeError, match="Media indices must be integers or slices"):
            media["invalid"]

        with pytest.raises(TypeError, match="Media indices must be integers or slices"):
            media[1.5]  # Float not supported

    def test_high_performance_iteration_pattern(self, media):
        """Test high-performance patterns for large file processing."""
        # Create multiple clips from large file (simulated)
        clips = []
        for i in range(10):
            start = i * 60  # 1-minute clips
            end = (i + 1) * 60
            clip = media[start:end]
            clips.append(clip)

        # Verify all clips are independent and correctly configured
        for i, clip in enumerate(clips):
            expected_start = i * 60
            expected_end = (i + 1) * 60
            assert clip._operations == [
                ("slice", {"start": expected_start, "end": expected_end})
            ]
            assert clip is not media
            assert clip.input_path == media.input_path

    def test_complex_slice_and_processing_chains(self, media):
        """Test complex processing patterns using slice notation."""
        # Create different versions of same time slice
        base_clip = media[30:60]

        # Different processing for same time slice
        small_version = base_clip.copy().resize(480, 320)
        audio_version = base_clip.copy().extract_audio()
        normalized_version = base_clip.copy().normalize_audio().fade_in(1.0)

        # Verify each has correct operations
        assert small_version._operations == [
            ("slice", {"start": 30, "end": 60}),
            ("resize", {"width": 480, "height": 320}),
        ]

        assert audio_version._operations == [
            ("slice", {"start": 30, "end": 60}),
            ("extract_audio", {}),
        ]

        assert normalized_version._operations == [
            ("slice", {"start": 30, "end": 60}),
            ("normalize_audio", {}),
            ("fade_in", {"duration": 1.0}),
        ]

    def test_slice_notation_preserves_lazy_evaluation(self, media):
        """Test that slice notation maintains lazy evaluation."""
        # Create sliced media
        clip = media[10:30]

        # Verify no processing has occurred yet
        assert len(clip._operations) == 1
        assert clip._operations[0] == ("slice", {"start": 10, "end": 30})

        # Add more operations - still lazy
        processed_clip = clip.resize(720, 480).extract_audio()
        assert len(processed_clip._operations) == 3

        # Operations should only execute when .to() is called
        # (This would be tested in integration tests with actual ffmpeg execution)


class TestAudioProcessing:
    """Test audio processing features (resample, mono, numpy)."""

    @pytest.fixture
    def media(self, tmp_path):
        """Create test media instance."""
        test_file = tmp_path / "test.mp4"
        test_file.touch()

        with patch("nitrox.core.FFmpegRunner") as mock_ffmpeg_class:
            mock_ffmpeg = Mock()
            mock_ffmpeg_class.return_value = mock_ffmpeg
            return Media(test_file)

    def test_resample_operation(self, media):
        """Test audio resampling."""
        result = media.resample(22050)
        assert result is media  # Returns self for chaining
        assert media._operations == [("resample", {"sample_rate": 22050})]

        # Test different sample rates
        media._operations.clear()
        media.resample(48000)
        assert media._operations == [("resample", {"sample_rate": 48000})]

    def test_resample_invalid_rate(self, media):
        """Test resample with invalid sample rate."""
        with pytest.raises(InvalidOperationError, match="Sample rate must be positive"):
            media.resample(0)

        with pytest.raises(InvalidOperationError, match="Sample rate must be positive"):
            media.resample(-44100)

    def test_to_mono_operation(self, media):
        """Test mono conversion."""
        result = media.to_mono()
        assert result is media  # Returns self for chaining
        assert media._operations == [("to_mono", {})]

    def test_audio_processing_chaining(self, media):
        """Test chaining audio processing operations."""
        result = media.resample(22050).to_mono().normalize_audio()

        expected_ops = [
            ("resample", {"sample_rate": 22050}),
            ("to_mono", {}),
            ("normalize_audio", {}),
        ]
        assert result._operations == expected_ops

    def test_audio_with_slice_notation(self, media):
        """Test audio processing with slice notation."""
        result = media[10:30].resample(16000).to_mono()

        expected_ops = [
            ("slice", {"start": 10, "end": 30}),
            ("resample", {"sample_rate": 16000}),
            ("to_mono", {}),
        ]
        assert result._operations == expected_ops

    @patch("nitrox.core.HAS_NUMPY", False)
    def test_to_numpy_without_numpy(self, media):
        """Test to_numpy when numpy is not available."""
        with pytest.raises(
            ImportError, match="numpy is required for audio array processing"
        ):
            media.to_numpy()

    @patch("nitrox.core.HAS_NUMPY", True)
    @patch("nitrox.core.np")
    @pytest.mark.skip(
        reason="Mocking issue causing recursion - TODO: fix complex numpy mocking"
    )
    def test_to_numpy_mock(self, mock_np, media):
        """Test to_numpy with mocked numpy."""
        # Mock numpy functions
        mock_array = Mock()
        mock_array.astype.return_value = mock_array
        mock_np.frombuffer.return_value = mock_array

        # Mock subprocess and MediaInfo
        with patch("nitrox.core.subprocess.run") as mock_run, patch(
            "nitrox.core.MediaInfo.from_file"
        ) as mock_info:

            mock_run.return_value.stdout = b"mock_audio_data"
            mock_info_instance = Mock()
            mock_info_instance.sample_rate = 44100
            mock_info_instance.channels = 2
            mock_info.return_value = mock_info_instance

            # Mock the to() method
            with patch.object(media, "to") as mock_to:
                result = media.to_numpy()

                # Verify calls
                mock_to.assert_called_once()
                mock_run.assert_called_once()
                mock_np.frombuffer.assert_called_with(
                    b"mock_audio_data", dtype=mock_np.float32
                )

    @patch("nitrox.core.HAS_NUMPY", False)
    def test_from_numpy_without_numpy(self):
        """Test from_numpy when numpy is not available."""
        with pytest.raises(
            ImportError, match="numpy is required for audio array processing"
        ):
            Media.from_numpy([1, 2, 3], 44100)  # Mock array

    @patch("nitrox.core.HAS_NUMPY", True)
    @patch("nitrox.core.np")
    @pytest.mark.skip(
        reason="Mocking issue causing recursion - TODO: fix complex numpy mocking"
    )
    def test_from_numpy_mock(self, mock_np):
        """Test from_numpy with mocked numpy."""
        # Mock numpy array
        mock_array = Mock()
        mock_array.ndim = 1
        mock_array.astype.return_value = mock_array
        mock_array.tobytes.return_value = b"mock_bytes"

        # Mock isinstance check
        with patch("builtins.isinstance", return_value=True):
            with patch("nitrox.core.subprocess.run") as mock_run, patch(
                "nitrox.core.len", return_value=44100
            ):

                with patch("nitrox.core.tempfile.NamedTemporaryFile") as mock_temp:
                    mock_temp.return_value.__enter__.return_value.name = "/tmp/test.wav"

                    result = Media.from_numpy(mock_array, 44100)

                    # Verify subprocess was called with correct parameters
                    mock_run.assert_called_once()
                    call_args = mock_run.call_args
                    assert "-ar" in call_args[0][0]
                    assert "44100" in call_args[0][0]

    def test_from_numpy_invalid_array(self):
        """Test from_numpy with invalid array."""
        with patch("nitrox.core.HAS_NUMPY", True):
            with pytest.raises(
                InvalidOperationError, match="audio_data must be a numpy array"
            ):
                Media.from_numpy("not_an_array", 44100)


class TestAudioProcessingWithX:
    """Test that X alias supports audio processing identically."""

    @pytest.fixture
    def test_file(self, tmp_path):
        """Create a test file."""
        test_file = tmp_path / "test.mp4"
        test_file.touch()
        return test_file

    @patch("nitrox.core.FFmpegRunner")
    def test_x_audio_processing(self, mock_ffmpeg_class, test_file):
        """Test that X supports all audio processing methods."""
        mock_ffmpeg = Mock()
        mock_ffmpeg_class.return_value = mock_ffmpeg

        x = X(test_file)

        # Test method availability
        assert hasattr(x, "resample")
        assert hasattr(x, "to_mono")
        assert hasattr(x, "to_numpy")
        assert hasattr(x, "from_numpy")

        # Test operations work
        result = x.resample(22050).to_mono()
        expected_ops = [("resample", {"sample_rate": 22050}), ("to_mono", {})]
        assert result._operations == expected_ops

    @patch("nitrox.core.FFmpegRunner")
    def test_x_vs_media_audio_equivalence(self, mock_ffmpeg_class, test_file):
        """Test that X and Media produce identical audio operations."""
        mock_ffmpeg = Mock()
        mock_ffmpeg_class.return_value = mock_ffmpeg

        # Same audio operations with Media and X
        media_result = Media(test_file)[10:20].resample(16000).to_mono()
        x_result = X(test_file)[10:20].resample(16000).to_mono()

        # Should have identical operations
        assert media_result._operations == x_result._operations
