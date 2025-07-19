#!/usr/bin/env python3
"""
Nitrox Basic Usage Examples

This script demonstrates common nitrox operations with both syntax styles:
- Media (clear and professional)
- X (ultra-minimal for power users)

Make sure you have some test media files to work with!
"""

from pathlib import Path

from nitrox import Media, X


def example_dual_syntax():
    """Example: Show both Media and X syntax doing the same thing."""
    print("=== Dual Syntax Demo ===")

    try:
        # Same operation, two styles:
        print("Professional style with Media:")
        Media("input.mp4").slice(10, 25).to("clip_media.mp3")
        print("Created clip_media.mp3 using Media syntax")

        print("\nUltra-minimal style with X:")
        X("input.mp4")[10:25].to("clip_x.mp3")
        print("Created clip_x.mp3 using X syntax")

        print("\nBoth files are identical - same functionality, different style!")

    except Exception as e:
        print(f"Error: {e}")


def example_slice_and_convert():
    """Example: Slice a video and convert to different format."""
    print("\n=== Slice and Convert ===")

    # Traditional method calls vs slice notation
    try:
        Media("input.mp4").slice(10, 25).to("clip.mp3")
        print("Media with .slice() method")

        X("input.mp4")[10:25].to("clip_x.mp3")
        print("X with [10:25] slice notation")
    except Exception as e:
        print(f"Error: {e}")


def example_slice_notation():
    """Example: NEW! Python slice notation for intuitive time slicing."""
    print("\n=== NEW: Slice Notation ===")

    try:
        # Show both styles with slice notation
        media = Media("input.mp4")
        x = X("input.mp4")

        # Python slice notation works with both!
        media[0:10].to("intro_media.mp4")
        x[0:10].to("intro_x.mp4")

        media[30:60].to("middle_media.mp4")
        x[30:60].to("middle_x.mp4")

        print("Slice notation works with both Media and X!")

    except Exception as e:
        print(f"Error: {e}")


def example_power_user_patterns():
    """Example: Ultra-minimal patterns for power users."""
    print("\n=== Power User X Patterns ===")

    try:
        # Ultra-compact power user style
        X("input.mp4")[5:15].resize(480, 320).to("power.mp4")
        X("input.mp4")[20:30].extract_audio().to("audio.mp3")
        X("input.mp4")[0:5].normalize_audio().to("norm.wav")

        print("Power user X syntax - maximum efficiency!")

    except Exception as e:
        print(f"Error: {e}")


def example_batch_processing():
    """Example: High-performance batch processing with slice notation."""
    print("\n=== Batch Processing ===")

    try:
        # Show both styles for batch processing
        print("Media style:")
        media = Media("input.mp4")
        clips = [media[i * 10 : (i + 1) * 10] for i in range(3)]

        for i, clip in enumerate(clips):
            clip.resize(480, 320).to(f"batch_media_{i}.mp4")

        print("Media batch processing complete")

        print("\nX style (more compact):")
        clips_x = [X("input.mp4")[i * 15 : (i + 1) * 15] for i in range(3)]

        for i, clip in enumerate(clips_x):
            clip.extract_audio().to(f"batch_x_{i}.mp3")

        print("X batch processing complete")

    except Exception as e:
        print(f"Error: {e}")


def example_slice_notation_chaining():
    """Example: Chain slice notation with other operations."""
    print("\n=== Slice Notation + Chaining ===")

    try:
        # Compare both styles
        Media("input.mp4")[15:45].resize(720, 480).extract_audio().to("chain_media.wav")
        X("input.mp4")[15:45].resize(720, 480).extract_audio().to("chain_x.wav")

        print("Chaining works beautifully with both styles!")

    except Exception as e:
        print(f"Error: {e}")


def example_chain_operations():
    """Example: Chain multiple operations together."""
    print("\n=== Chain Multiple Operations ===")

    try:
        # Complex pipeline: slice, resize, normalize audio, save
        Media("video.mov").slice(30, 60).resize(720, 480).normalize_audio().to(
            "processed_clip.mp4", crf=20, preset="fast"
        )

        print("Applied slice + resize + audio normalization")
    except Exception as e:
        print(f"Error: {e}")


def example_audio_operations():
    """Example: Audio-focused operations."""
    print("\n=== Audio Operations ===")

    try:
        # Extract audio from video with fade effects
        Media("video.mp4").extract_audio().fade_in(2.0).fade_out(1.5).to(
            "audio_with_fades.wav"
        )

        print("Extracted audio with fade in/out effects")
    except Exception as e:
        print(f"Error: {e}")


def example_video_operations():
    """Example: Video-focused operations."""
    print("\n=== Video Operations ===")

    try:
        # Resize and crop video
        Media("large_video.avi").resize(1280, 720).crop(100, 50, 1080, 620).set_fps(
            24
        ).to("cropped_video.mp4")

        print("Resized, cropped, and changed FPS")
    except Exception as e:
        print(f"Error: {e}")


def example_get_info():
    """Example: Get media file information."""
    print("\n=== Media Information ===")

    try:
        media = Media("input.mp4")
        info = media.info()

        print(f"File: {media.input_path}")
        print(f"Duration: {info.duration:.2f} seconds")

        if info.has_video:
            print(f"Video: {info.width}x{info.height} @ {info.fps:.2f}fps")
            print(f"Video codec: {info.video_codec}")

        if info.has_audio:
            print(f"Audio: {info.channels} channels @ {info.sample_rate}Hz")
            print(f"Audio codec: {info.audio_codec}")

        print(f"File size: {info.file_size / (1024*1024):.1f} MB")

    except Exception as e:
        print(f"Error: {e}")


def example_preview_command():
    """Example: Preview ffmpeg command without executing."""
    print("\n=== Preview FFmpeg Command ===")

    try:
        media = Media("input.mp4")
        pipeline = media.slice(10, 20).resize(640, 480)

        print("FFmpeg command that would be executed:")
        print(pipeline.preview_command())

    except Exception as e:
        print(f"Error: {e}")


def example_copy_and_modify():
    """Example: Copy media instance and modify differently."""
    print("\n=== Copy and Modify ===")

    try:
        # Start with base media
        media = Media("input.mp4").slice(5, 30)

        # Create different versions
        small_version = media.copy().resize(480, 320).to("small.mp4")
        audio_version = media.copy().extract_audio().to("audio.mp3")

        print("Created multiple versions from same base")

    except Exception as e:
        print(f"Error: {e}")


def example_format_specific_options():
    """Example: Format-specific encoding options."""
    print("\n=== Format-Specific Options ===")

    try:
        media = Media("input.mp4")

        # High quality MP4
        media.copy().to("high_quality.mp4", crf=18, preset="slow")

        # High bitrate MP3
        media.copy().extract_audio().to("high_quality.mp3", bitrate="320k")

        # Uncompressed WAV
        media.copy().extract_audio().to("uncompressed.wav")

        print("Created files with different quality settings")

    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all examples."""
    print("Nitrox Basic Usage Examples")
    print("=" * 40)

    # Check if we have a test file
    if not Path("input.mp4").exists():
        print("\nWarning: No 'input.mp4' found.")
        print("Some examples may fail. Add a test video file to see all features!")

    # Run examples
    example_dual_syntax()
    example_slice_and_convert()
    example_slice_notation()
    example_power_user_patterns()
    example_batch_processing()
    example_slice_notation_chaining()
    example_chain_operations()
    example_audio_operations()
    example_video_operations()
    example_get_info()
    example_preview_command()
    example_copy_and_modify()
    example_format_specific_options()

    print("\nExamples complete!")
    print("\nNext steps:")
    print("- Try with your own media files")
    print("- Check the generated output files")
    print("- Experiment with different options")
    print("- Try the new slice notation: media[start:end]")


if __name__ == "__main__":
    main()
