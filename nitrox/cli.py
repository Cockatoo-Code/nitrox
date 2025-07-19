"""Command-line interface for nitrox."""

import argparse
import sys
from pathlib import Path

from . import Media, __version__
from .exceptions import NitroxException


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Nitrox: Lightning-fast media slicing and conversion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nitrox slice input.mp4 output.mp4 --start 10 --end 30
  nitrox convert video.avi audio.mp3 --extract-audio
  nitrox resize large.mp4 small.mp4 --width 720 --height 480
  nitrox info video.mp4
        """,
    )

    parser.add_argument("--version", action="version", version=f"nitrox {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Slice command
    slice_parser = subparsers.add_parser("slice", help="Slice media file")
    slice_parser.add_argument("input", help="Input file path")
    slice_parser.add_argument("output", help="Output file path")
    slice_parser.add_argument("--start", type=float, help="Start time in seconds")
    slice_parser.add_argument("--end", type=float, help="End time in seconds")
    slice_parser.add_argument(
        "--crf", type=int, default=23, help="Video quality (lower = better)"
    )
    slice_parser.add_argument("--preset", default="medium", help="Encoding preset")

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert media format")
    convert_parser.add_argument("input", help="Input file path")
    convert_parser.add_argument("output", help="Output file path")
    convert_parser.add_argument(
        "--extract-audio", action="store_true", help="Extract audio only"
    )
    convert_parser.add_argument(
        "--normalize", action="store_true", help="Normalize audio"
    )
    convert_parser.add_argument("--bitrate", help="Audio bitrate (e.g. 192k)")
    convert_parser.add_argument("--crf", type=int, default=23, help="Video quality")

    # Resize command
    resize_parser = subparsers.add_parser("resize", help="Resize video")
    resize_parser.add_argument("input", help="Input file path")
    resize_parser.add_argument("output", help="Output file path")
    resize_parser.add_argument("--width", type=int, required=True, help="Target width")
    resize_parser.add_argument(
        "--height", type=int, required=True, help="Target height"
    )
    resize_parser.add_argument("--crf", type=int, default=23, help="Video quality")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show media file information")
    info_parser.add_argument("input", help="Input file path")
    info_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Preview command
    preview_parser = subparsers.add_parser("preview", help="Preview ffmpeg command")
    preview_parser.add_argument("input", help="Input file path")
    preview_parser.add_argument("output", help="Output file path")
    preview_parser.add_argument(
        "--slice", nargs=2, type=float, metavar=("START", "END")
    )
    preview_parser.add_argument(
        "--resize", nargs=2, type=int, metavar=("WIDTH", "HEIGHT")
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "slice":
            slice_media(args)
        elif args.command == "convert":
            convert_media(args)
        elif args.command == "resize":
            resize_media(args)
        elif args.command == "info":
            show_info(args)
        elif args.command == "preview":
            preview_command(args)

    except NitroxException as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled.", file=sys.stderr)
        sys.exit(1)


def slice_media(args):
    """Handle slice command."""
    media = Media(args.input)

    if args.start is not None or args.end is not None:
        media = media.slice(args.start, args.end)

    print(f"Slicing {args.input} -> {args.output}")
    media.to(args.output, crf=args.crf, preset=args.preset)
    print("Done!")


def convert_media(args):
    """Handle convert command."""
    media = Media(args.input)

    if args.extract_audio:
        media = media.extract_audio()

    if args.normalize:
        media = media.normalize_audio()

    kwargs = {"crf": args.crf}
    if args.bitrate:
        kwargs["bitrate"] = args.bitrate

    print(f"Converting {args.input} -> {args.output}")
    media.to(args.output, **kwargs)
    print("Done!")


def resize_media(args):
    """Handle resize command."""
    media = Media(args.input)
    media = media.resize(args.width, args.height)

    print(f"Resizing {args.input} to {args.width}x{args.height} -> {args.output}")
    media.to(args.output, crf=args.crf)
    print("Done!")


def show_info(args):
    """Handle info command."""
    media = Media(args.input)
    info = media.info()

    if args.json:
        import json

        data = {
            "file": str(media.input_path),
            "duration": info.duration,
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "has_video": info.has_video,
            "has_audio": info.has_audio,
            "video_codec": info.video_codec,
            "audio_codec": info.audio_codec,
            "sample_rate": info.sample_rate,
            "channels": info.channels,
            "file_size": info.file_size,
            "bit_rate": info.bit_rate,
            "format_name": info.format_name,
        }
        print(json.dumps(data, indent=2))
    else:
        print(f"File: {media.input_path}")
        print(
            f"Duration: {info.duration:.2f}s" if info.duration else "Duration: unknown"
        )

        if info.has_video:
            print(
                f"Video: {info.width}x{info.height} @ {info.fps:.2f}fps ({info.video_codec})"
            )

        if info.has_audio:
            channels_str = f"{info.channels}ch" if info.channels else "unknown channels"
            rate_str = f"{info.sample_rate}Hz" if info.sample_rate else "unknown rate"
            print(f"Audio: {channels_str}, {rate_str} ({info.audio_codec})")

        if info.file_size:
            size_mb = info.file_size / (1024 * 1024)
            print(f"Size: {size_mb:.1f} MB")


def preview_command(args):
    """Handle preview command."""
    media = Media(args.input)

    if args.slice:
        start, end = args.slice
        media = media.slice(start, end)

    if args.resize:
        width, height = args.resize
        media = media.resize(width, height)

    # Create dummy output for preview
    dummy_output = Path(args.output)
    command = media._ffmpeg.build_command(
        input_path=media.input_path,
        output_path=dummy_output,
        operations=media._operations,
    )

    print("FFmpeg command:")
    print(" ".join(command))


if __name__ == "__main__":
    main()
