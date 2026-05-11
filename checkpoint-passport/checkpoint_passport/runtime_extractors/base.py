"""
Base protocol for runtime passport seed extractors.

Each backend (openpi, ...) implements extract_seed() which returns a
passport seed dict.  The CLI dispatches to the right extractor by name.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict, List, Optional


SUPPORTED_BACKENDS = {"openpi"}


class UnsupportedBackendError(Exception):
    """Raised when the requested backend is not implemented."""

    def __init__(self, backend: str):
        self.backend = backend
        supported = sorted(SUPPORTED_BACKENDS)
        super().__init__(
            f"extractor not implemented for architecture '{backend}'; "
            f"supported: {', '.join(supported)}; stop and ask"
        )


class MissingRuntimeError(Exception):
    """Raised when a required runtime dependency cannot be imported."""

    def __init__(self, module: str, backend: str):
        self.module = module
        self.backend = backend
        super().__init__(
            f"{backend.capitalize()} seed extraction must run inside the "
            f"{backend.capitalize()} runtime environment.\n"
            f"Missing import: {module}. "
            f"Activate the target environment or container and rerun."
        )


class BaseExtractor(abc.ABC):
    """Protocol for runtime passport seed extractors."""

    @abc.abstractmethod
    def extract_seed(
        self,
        checkpoint_dir: Path,
        *,
        config_name: str,
        default_prompt: Optional[str] = None,
        resize_size: Optional[int] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract a passport seed from a checkpoint.

        When *device* is provided (e.g. ``"cuda"`` or ``"cpu"``), the
        extractor may load the model via the deployment adapter to
        collect runtime enrichment: library versions, parameter summary,
        and numerical health / smoke results.  Without *device*, only
        config-level extraction is performed (no weight loading).

        Returns:
            A dict conforming to the passport seed schema
            (see passport_seed.validate_seed).

        Raises:
            MissingRuntimeError: framework is not installed.
            ValueError: required arguments missing or config invalid.
        """
        ...
