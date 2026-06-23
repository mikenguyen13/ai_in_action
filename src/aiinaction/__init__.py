"""AI in Action: reusable, tested reference implementations.

This package holds production-grade, type-hinted, validated implementations of the
algorithms developed in the book. Python is the executed reference language; the
Julia package (``julia/AIInAction``) and the Rust crate (``rust/aiinaction``) mirror
the same public APIs and are tested in CI so the three stay at parity.

Chapters import from this package and demonstrate usage, rather than re-deriving the
implementation inline, so readers can ``pip install`` and reuse the same code.
"""
from __future__ import annotations

from . import metrics

__all__ = ["metrics"]
__version__ = "0.1.0"
