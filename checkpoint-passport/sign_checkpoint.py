#!/usr/bin/env python3
"""Top-level shim so `python sign_checkpoint.py <ckpt>` keeps working
without `pip install`.  The real implementation lives at
`checkpoint_passport.cli.sign`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint_passport.cli.sign import main  # noqa: E402

if __name__ == "__main__":
    main()
