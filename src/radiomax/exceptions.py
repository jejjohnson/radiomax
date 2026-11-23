"""Exception hierarchy for `radiomax`.

Every error raised by this package derives from
[`RadiomaxError`][radiomax.RadiomaxError], so a caller can trap the whole
package with a single ``except`` clause while still being able to discriminate
between failure modes when it matters.

The exception is the array backend, which is pipekit-array's: an input that is
no array of one known library raises pipekit's `BackendError`, and an input
that is not real, or an unlabelled array next to a labelled one, raises a plain
`ValueError`.

Both are a `ValueError`, as the base class is, so code written against the
standard library keeps working without importing anything from this package.
"""

from __future__ import annotations

from pipekit_array.functions import BackendError


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


class ValidationError(RadiomaxError):
    """Raised for a bad argument: its shape, a dimension name or its value."""
