"""Errors that carry scientific or input meaning across interface boundaries."""


class ModeLensError(Exception):
    """Base class for expected application failures."""


class InvalidVideoError(ModeLensError):
    """Raised when a file cannot be decoded or violates hard limits."""


class InvalidExperimentError(ModeLensError):
    """Raised when image coordinates or physical measurements are inconsistent."""


class InsufficientSignalError(ModeLensError):
    """Raised when the capture cannot support a modal estimate."""


class TrackingError(ModeLensError):
    """Raised when too few beam points survive tracking."""


class NonIdentifiableError(ModeLensError):
    """Raised when the requested physical parameters cannot be separated."""
