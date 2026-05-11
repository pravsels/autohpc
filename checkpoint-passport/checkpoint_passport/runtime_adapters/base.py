"""
Runtime adapter protocol for model enrichment.

Each backend (openpi, diffusion, ...) implements the four methods to
load a model, count parameters, run a smoke inference, and report
library versions.  The enrichment path in the seed extractor calls
these methods — it never imports backend-specific types.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict, Optional


class RuntimeAdapter(abc.ABC):
    """What enrichment needs from a loaded model — nothing more."""

    @abc.abstractmethod
    def load(self, checkpoint_dir: Path, *, device: str, **kwargs) -> None:
        """Load model weights onto *device*.

        Backend-specific kwargs (``config_name``, ``config_path``, etc.)
        are passed through from the extractor.

        Raises:
            ImportError / ModuleNotFoundError if the framework is missing.
            ValueError if required kwargs are absent or config is invalid.
        """

    @abc.abstractmethod
    def count_parameters(self) -> Optional[Dict[str, int]]:
        """Return ``{"total_params": N}`` (and optionally
        ``"trainable_params"``) or *None* if unavailable."""

    @abc.abstractmethod
    def smoke_inference(self) -> Optional[Dict[str, Any]]:
        """Run one forward pass with a minimal dummy input.

        Return ``{"status": "pass", "elapsed_ms": float}`` on success,
        ``{"status": "fail", "error": str}`` on failure, or *None* if
        the adapter cannot construct a valid dummy input.
        """

    @abc.abstractmethod
    def library_versions(self) -> Dict[str, str]:
        """Return ``{lib_name: version}`` for libraries in this env."""

    @abc.abstractmethod
    def extract_reference_sample(
        self,
        dataset_path: Path,
        *,
        episode_index: int = 0,
        start_frame: int = 0,
        num_frames: int = 10,
    ) -> Dict[str, Any]:
        """Load consecutive frames from a real dataset as reference data.

        Returns a dict with:
            - ``"states"``: numpy array shaped ``(num_frames, state_dim)``
            - ``"images"``: ``{camera_key: [numpy_array per frame]}``
            - ``"prompt"``: task/language string if present
            - ``"episode_index"``, ``"start_frame"``, ``"num_frames"``
            - ``"dataset_path"``: str

        Raises:
            ValueError if the dataset is unreadable or the requested
            range is out of bounds.
        """
