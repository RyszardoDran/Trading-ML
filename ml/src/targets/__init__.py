"""Target creation module.

Purpose:
    Create binary target labels for model training (win/loss).

Exports:
    - make_target(): Backtest-based target creation
"""

from .target_maker import make_target

__all__ = ["make_target"]
