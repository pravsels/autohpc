"""
Layer 1 deterministic kernel: per-category check modules.

Each module exposes an `ALL_CHECKS` list and a `category` constant. The
runner collects across modules and groups by category in the report.
"""
