"""Unit tests for the training pipeline health checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.pipelines.training_pipeline import (
    list_data_files,
    read_sample,
    run_health_check,
)


REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def _write_csv(path: Path, rows: list[dict[str, float]] | None = None) -> Path:
    """Create a small CSV file using the canonical column set."""
    default_rows = rows or [
        {
            "Date": "2025.12.08 08:00",
            "Open": 1.0,
            "High": 1.1,
            "Low": 0.9,
            "Close": 1.05,
            "Volume": 10.0,
        },
        {
            "Date": "2025.12.08 08:01",
            "Open": 1.05,
            "High": 1.12,
            "Low": 1.0,
            "Close": 1.1,
            "Volume": 8.0,
        },
    ]
    frame = pd.DataFrame(default_rows)
    frame.to_csv(path, sep=";", index=False)
    return path


def test_list_data_files_requires_existing_directory(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        list_data_files(missing_dir)


def test_list_data_files_requires_csv_files(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "notes.txt").write_text("placeholder", encoding="utf-8")

    with pytest.raises(ValueError):
        list_data_files(data_dir)


def test_list_data_files_returns_sorted_paths(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "b.csv").write_text("Date;Open", encoding="utf-8")
    (data_dir / "a.csv").write_text("Date;Open", encoding="utf-8")

    files = list_data_files(data_dir)

    assert files == [data_dir / "a.csv", data_dir / "b.csv"]


def test_read_sample_rejects_missing_columns(tmp_path: Path) -> None:
    file_path = tmp_path / "invalid.csv"
    pd.DataFrame({"Date": ["2025.12.08 08:00"], "Close": [1.0]}).to_csv(
        file_path,
        sep=";",
        index=False,
    )

    with pytest.raises(ValueError):
        read_sample(file_path, n_rows=1)


def test_read_sample_returns_requested_number_of_rows(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.csv"
    _write_csv(file_path)

    sample = read_sample(file_path, n_rows=1)

    assert len(sample) == 1
    assert list(sample.columns) == REQUIRED_COLUMNS


def test_run_health_check_reports_summary(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_csv(data_dir / "A.csv")
    _write_csv(data_dir / "B.csv")

    summary = run_health_check(data_dir=data_dir, max_files=1, sample_rows=1)

    assert summary["file_count"] == 2
    assert summary["files_sampled"] == 1
    assert summary["total_rows_sampled"] == 1
    assert summary["data_dir"] == str(data_dir)
    assert summary["required_columns"] == REQUIRED_COLUMNS
