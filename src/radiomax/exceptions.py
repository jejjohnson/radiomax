"""Exception hierarchy for `radiomax`.

Every error raised by this package derives from
[`RadiomaxError`][radiomax.RadiomaxError], so a caller can trap the whole
package with a single ``except`` clause while still being able to discriminate
between failure modes when it matters.

The base class is a `ValueError`, so code written against the standard library
keeps working without importing anything from this package.
"""

from __future__ import annotations


__all__ = [
    "BackendError",
    "RadiomaxError",
    "ValidationError",
]


class RadiomaxError(ValueError):
    """Base class for every exception raised by ``radiomax``.

    Examples:
        >>> from radiomax import RadiomaxError, ValidationError
        >>> issubclass(ValidationError, RadiomaxError)
        True
        >>> issubclass(RadiomaxError, ValueError)
        True
    """


class BackendError(RadiomaxError):
    """Raised for an array of an unknown library, or of several in one call."""


class ValidationError(RadiomaxError):
    """Raised for a bad argument: its shape, a dimension name or its value."""
