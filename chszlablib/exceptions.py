"""Custom exception hierarchy for CHSZLabLib.

All exceptions inherit from both :class:`CHSZLabLibError` and their
corresponding built-in type (``ValueError``, ``RuntimeError``), so existing
``except ValueError`` / ``except RuntimeError`` handlers continue to work.
"""


class CHSZLabLibError(Exception):
    """Base exception for all CHSZLabLib errors."""


class InvalidModeError(CHSZLabLibError, ValueError):
    """Raised when an invalid mode or algorithm string is passed."""


class InvalidGraphError(CHSZLabLibError, ValueError):
    """Raised when a graph has invalid structure (bad CSR, out-of-bounds nodes)."""


class GraphNotFinalizedError(CHSZLabLibError, RuntimeError):
    """Raised when an operation requires a finalized graph but it is not."""
