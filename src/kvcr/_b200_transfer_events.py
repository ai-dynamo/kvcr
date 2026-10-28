"""Compatibility import for explicit compact capture; no AST observer installed."""
from .compact_telemetry import (  # noqa: F401
    BufferedWriter, CaptureInvalid, SCHEMA_VERSION, _error_message, _shutdown,
    emit, iter_key_journal, validate_capture, validate_key_journal, writer,
)
