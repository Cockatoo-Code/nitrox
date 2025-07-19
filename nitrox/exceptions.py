"""Custom exceptions for nitrox."""


class NitroxException(Exception):
    """Base exception for all nitrox errors."""

    pass


class FFmpegError(NitroxException):
    """Raised when ffmpeg command fails."""

    def __init__(self, message, command=None, returncode=None, stderr=None):
        super().__init__(message)
        self.command = command
        self.returncode = returncode
        self.stderr = stderr


class MediaNotFoundError(NitroxException):
    """Raised when input media file cannot be found."""

    pass


class InvalidOperationError(NitroxException):
    """Raised when an invalid operation is attempted."""

    pass


class UnsupportedFormatError(NitroxException):
    """Raised when an unsupported media format is used."""

    pass
