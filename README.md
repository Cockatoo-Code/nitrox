# 🫧 Nitrox

> **Deep, fast, clean. Media processing at maximum depth.**

Nitrox is a Python library for high performance audio and video processing. Built on ffmpeg with a clean, intuitive API.

## Installation

```bash
pip install nitrox
```

**With audio processing:**
```bash
pip install nitrox[audio]
```

### FFmpeg Requirement

Nitrox requires ffmpeg to be installed on your system.

**macOS:**
```bash
# Using Homebrew
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Linux (RHEL/CentOS):**
```bash
sudo dnf install ffmpeg
# or
sudo yum install ffmpeg
```

**Other platforms:** Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Quick Start

```python
from nitrox import Media

# Basic usage
Media("input.mp4").slice(10, 30).to("output.mp4")

# Audio extraction
Media("video.mp4").extract_audio().to("audio.wav")

# Video processing
Media("input.mp4").resize(720, 480).to("resized.mp4")
```

### Minimal Syntax

For power users, Nitrox offers an ultra-minimal `X` alias:

```python
from nitrox import X

# Same functionality, minimal syntax
X("input.mp4")[10:30].to("output.mp4")
X("video.mov").extract_audio().to("audio.mp3")
```

## Core Features

### Intuitive Slicing
```python
media = Media("video.mp4")

# Time-based slicing
media[0:10]              # First 10 seconds
media[30:60]             # From 30s to 60s
media[10:]               # From 10s to end
media[:30]               # First 30 seconds
```

### Chainable Operations
```python
# Video processing
Media("input.mp4") \
    .slice(10, 60) \
    .resize(1280, 720) \
    .to("processed.mp4")

# Audio processing
Media("audio.wav") \
    .normalize_audio() \
    .fade_in(1.0) \
    .fade_out(2.0) \
    .to("clean.wav")
```

### Audio Processing
```python
# Basic audio operations
Media("input.mp3").resample(44100).to("output.wav")
Media("stereo.wav").to_mono().to("mono.wav")

# With numpy integration
audio_data = Media("song.wav")[10:20].to_numpy()
Media.from_numpy(audio_data, 44100).to("processed.wav")
```

## Examples

### Extract Highlights
```python
# Extract multiple clips
video = Media("long_video.mp4")
highlights = [
    video[0:30],      # Opening
    video[300:330],   # Middle segment  
    video[1200:1230]  # Ending
]

for i, clip in enumerate(highlights):
    clip.to(f"highlight_{i}.mp4")
```

### Podcast Processing
```python
# Clean up podcast audio
Media("raw_recording.wav") \
    .normalize_audio() \
    .fade_in(0.5) \
    .fade_out(1.0) \
    .to("podcast_ready.mp3")
```

### Batch Processing
```python
# Process multiple files
import os

for filename in os.listdir("raw_videos/"):
    if filename.endswith('.mp4'):
        Media(f"raw_videos/{filename}") \
            .resize(720, 480) \
            .to(f"processed/{filename}")
```

## API Reference

### Core Operations

**Video:**
- `resize(width, height)` - Resize video
- `crop(x, y, width, height)` - Crop video
- `set_fps(fps)` - Change frame rate

**Audio:**
- `extract_audio()` - Extract audio track
- `normalize_audio()` - Normalize audio levels
- `resample(rate)` - Change sample rate
- `to_mono()` - Convert to mono
- `fade_in(duration)` - Add fade in
- `fade_out(duration)` - Add fade out

**Slicing:**
- `slice(start, end)` - Extract time range
- `[start:end]` - Python slice notation

**Output:**
- `to(path)` - Save processed media
- `to_numpy()` - Extract as numpy array *(requires numpy)*
- `from_numpy(data, rate)` - Create from numpy array *(requires numpy)*

**Utility:**
- `info()` - Get media information
- `preview_command()` - Show ffmpeg command

### Command Line

```bash
# Basic usage
nitrox slice input.mp4 output.mp4 --start 10 --end 30
nitrox resize video.mp4 small.mp4 --width 720 --height 480
nitrox convert video.mp4 audio.mp3 --extract-audio

# Get media info
nitrox info video.mp4
```

## Requirements

- Python 3.8+
- ffmpeg (see installation instructions above)
- numpy *(optional, for array processing)*

## License

MIT License - see [LICENSE](LICENSE) file.

---

**🫧 Nitrox ** 