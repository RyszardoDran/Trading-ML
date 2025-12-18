#!/usr/bin/env python3
"""Convert XAU 1m CSV files (2025) to the same format as the 2024 files.

Output format:
- delimiter: semicolon
- header: Date;Open;High;Low;Close;Volume
- Date format: YYYY.MM.DD HH:MM (no seconds)

The script tolerates common input delimiters and several date formats.
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


PREFERRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def detect_delimiter(sample: str) -> str:
    # try common delimiters
    for d in [';', ',', '\t', '|']:
        if d in sample:
            return d
    return ','


def read_csv_flexible(path: Path) -> pd.DataFrame:
    # read a small sample to detect delimiter
    text = path.read_text(encoding='utf-8', errors='ignore')
    first_line = text.splitlines()[0] if text else ''
    delim = detect_delimiter(first_line)
    # try reading and be forgiving with decimal commas
    try:
        df = pd.read_csv(path, sep=delim, engine='python')
    except Exception:
        # fallback: let pandas try to infer
        df = pd.read_csv(path, engine='python')
    return df


def normalize_column_names(cols: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for c in cols:
        lc = c.strip().lower()
        if 'date' in lc or 'time' in lc:
            mapping[c] = 'Date'
        elif lc in ('open', 'o'):
            mapping[c] = 'Open'
        elif lc in ('high', 'h'):
            mapping[c] = 'High'
        elif lc in ('low', 'l'):
            mapping[c] = 'Low'
        elif lc in ('close', 'c'):
            mapping[c] = 'Close'
        elif 'volume' in lc or lc in ('v', 'vol'):
            mapping[c] = 'Volume'
    return mapping


def parse_dates_column(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col].astype(str).str.strip()
    # replace common separators
    # try multiple formats
    formats = [
        '%Y.%m.%d %H:%M',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%d.%m.%Y %H:%M',
        '%d/%m/%Y %H:%M',
        '%Y/%m/%d %H:%M',
        '%m/%d/%Y %H:%M',
    ]
    for fmt in formats:
        try:
            return pd.to_datetime(s, format=fmt, errors='raise')
        except Exception:
            continue
    # last resort: let pandas infer
    return pd.to_datetime(s, errors='coerce')


def ensure_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col].astype(str).str.replace(',', '.', regex=False).str.replace(' ', '')
    return pd.to_numeric(s, errors='coerce')


def convert(in_path: Path, out_path: Path, overwrite: bool = False) -> None:
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {out_path} (use --overwrite to replace)")

    df = read_csv_flexible(in_path)

    # normalize columns
    col_map = normalize_column_names(list(df.columns))
    df = df.rename(columns=col_map)

    # If Date column not found try first column
    if 'Date' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'Date'})

    # Ensure required columns exist (fill missing with NaN/0)
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c not in df.columns:
            df[c] = pd.NA

    # parse dates
    df['Date_parsed'] = parse_dates_column(df, 'Date')
    if df['Date_parsed'].isna().all():
        raise ValueError('Could not parse any dates from the Date column')

    # format numbers
    for c in ['Open', 'High', 'Low', 'Close']:
        df[c] = ensure_numeric(df, c)
    df['Volume'] = ensure_numeric(df, 'Volume').fillna(0).astype(int)

    # use parsed date, drop originals and format
    df['Date'] = df['Date_parsed'].dt.strftime('%Y.%m.%d %H:%M')
    out = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # sort
    out = out.sort_values('Date')

    # write with semicolon delimiter
    out.to_csv(out_path, sep=';', index=False, header=PREFERRED_COLUMNS, float_format='%.6f')


def main(argv: Optional[list[str]] = None) -> None:
    p = argparse.ArgumentParser(description='Convert XAU 1m CSV to 2024 format')
    p.add_argument('input', type=Path, help='Input CSV path (2025 raw data)')
    p.add_argument('output', type=Path, nargs='?', help='Output CSV path', default=Path('ml/src/data/XAU_1m_data_2025_converted.csv'))
    p.add_argument('--overwrite', action='store_true', help='Overwrite output if exists')
    args = p.parse_args(argv)

    try:
        convert(args.input, args.output, overwrite=args.overwrite)
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
