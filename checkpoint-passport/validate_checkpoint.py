#!/usr/bin/env python3
"""Top-level shim so `python validate_checkpoint.py <ckpt>` keeps working
without `pip install`.  The real implementation lives at
`checkpoint_passport.cli.validate`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running the script directly from a sibling-cloned autohpc/ without
# requiring `pip install -e .` first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint_passport.cli.validate import main  # noqa: E402

if __name__ == "__main__":
    main()
