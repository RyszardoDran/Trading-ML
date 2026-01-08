"""Canonical XGBoost hyperparameter profiles for the sequence model.

The default profile is heavily regularized to close the gap between
backtest win rate (~80%) and live data performance (<20%). Profiles are
kept intentionally small so they can be enumerated in logs and easily
overridden via CLI flags.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Optional, Union


@dataclass(frozen=True)
class XGBParamProfile:
    """Immutable container describing a vetted hyperparameter set."""

    name: str
    learning_rate: float
    max_depth: int
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    colsample_bynode: float
    reg_lambda: float
    reg_alpha: float
    gamma: float
    n_estimators: int
    max_delta_step: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        data = asdict(self)
        data.pop("name", None)
        return data


XGB_PARAM_PROFILES: Dict[str, XGBParamProfile] = {
    "legacy": XGBParamProfile(
        name="legacy",
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=1.0,
        subsample=0.7,
        colsample_bytree=0.6,
        colsample_bynode=1.0,
        reg_lambda=1.0,
        reg_alpha=0.1,
        gamma=0.0,
        n_estimators=600,
        max_delta_step=0.0,
    ),
    "regularized": XGBParamProfile(
        name="regularized",
        learning_rate=0.018,
        max_depth=4,
        min_child_weight=8.0,
        subsample=0.55,
        colsample_bytree=0.45,
        colsample_bynode=0.8,
        reg_lambda=2.5,
        reg_alpha=1.2,
        gamma=1.0,
        n_estimators=850,
        max_delta_step=1.0,
    ),
    "conservative": XGBParamProfile(
        name="conservative",
        learning_rate=0.015,
        max_depth=3,
        min_child_weight=10.0,
        subsample=0.5,
        colsample_bytree=0.4,
        colsample_bynode=0.75,
        reg_lambda=3.5,
        reg_alpha=1.5,
        gamma=1.5,
        n_estimators=900,
        max_delta_step=1.5,
    ),
}

DEFAULT_XGB_PROFILE = "regularized"


def available_profiles() -> Iterable[str]:
    """Return supported profile names."""

    return sorted(XGB_PARAM_PROFILES.keys())


def build_xgb_params(
    profile: str,
    overrides: Optional[Dict[str, Optional[Union[float, int]]]] = None,
) -> Dict[str, float]:
    """Return concrete params for XGBClassifier.

    Args:
        profile: Name of the base profile.
        overrides: Optional per-parameter overrides coming from CLI.

    Returns:
        Dictionary ready to be passed into XGBClassifier.
    """

    if profile not in XGB_PARAM_PROFILES:
        raise ValueError(
            f"Unknown XGBoost profile '{profile}'. Available: {', '.join(available_profiles())}"
        )

    resolved = XGB_PARAM_PROFILES[profile].to_dict()

    if overrides:
        for key, value in overrides.items():
            if value is not None:
                resolved[key] = value

    return resolved
