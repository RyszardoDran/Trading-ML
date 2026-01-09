#!/usr/bin/env python3
"""Generic CSV format converter utility.

This small utility normalizes CSV files into a consistent OHLCV format
and delimiter suitable for downstream pipelines. It is tolerant of
common input delimiters and several date/time formats.

Default output format (can be customized by editing the script):
- delimiter: semicolon
- header: Date;Open;High;Low;Close;Volume
- Date format: YYYY.MM.DD HH:MM (no seconds)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


PREFERRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def detect_delimiter(sample: str) -> str:
    for d in [';', ',', '\t', '|']:
        if d in sample:
            return d
    return ','


def read_csv_flexible(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding='utf-8', errors='ignore')
    first_line = text.splitlines()[0] if text else ''
    delim = detect_delimiter(first_line)
    
    # Try tab first (common format)
    try:
        df = pd.read_csv(path, sep='\t', engine='python')
        if len(df.columns) > 1:
            return df
    except Exception:
        pass
    
    # Try whitespace
    try:
        df = pd.read_csv(path, sep=r'\s+', engine='python')
        if len(df.columns) > 1:
            return df
    except Exception:
        pass
    
    # Try detected delimiter
    try:
        df = pd.read_csv(path, sep=delim, engine='python')
        if len(df.columns) > 1:
            return df
    except Exception:
        pass
    
    # Fallback
    return pd.read_csv(path, engine='python')


def normalize_column_names(cols: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for c in cols:
        lc = c.strip().lower()
        # Ignore columns with "not_" in name
        if 'not_' in lc or 'irrelevant' in lc:
            continue
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
    formats = [
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d %H:%M:%S',
        '%Y.%m.%d %H:%M',
        '%Y.%m.%d %H:%M:%S',
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
    cols_list = df.columns.tolist()
    
    # Check if this is a headerless space-delimited file (date in first col)
    has_date_col = False
    try:
        pd.to_datetime(cols_list[0], format='%Y-%m-%d', errors='raise')
        has_date_col = True
    except:
        pass
    
    if has_date_col and len(cols_list) >= 7:
        # Merge first two columns as date+time
        df['Date'] = df[cols_list[0]].astype(str) + ' ' + df[cols_list[1]].astype(str)
        df['Open'] = df[cols_list[2]]
        df['High'] = df[cols_list[3]]
        df['Low'] = df[cols_list[4]]
        df['Close'] = df[cols_list[5]]
        df['Volume'] = df[cols_list[6]]
    else:
        # Normalize columns normally
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
    p = argparse.ArgumentParser(description='Generic CSV format converter')
    p.add_argument('input', type=Path, nargs='?', help='Input CSV path')
    p.add_argument('output', type=Path, nargs='?', help='Output CSV path', default=Path('ml/src/data/converted.csv'))
    p.add_argument('--source', '-s', type=Path, help='Source input CSV path (alternative to positional input)')
    p.add_argument('--dest', '-d', type=Path, help='Destination output CSV path (alternative to positional output)')
    p.add_argument('--overwrite', action='store_true', help='Overwrite output if exists')
    args = p.parse_args(argv)

    in_path: Optional[Path] = None
    out_path: Optional[Path] = None

    if args.source:
        in_path = args.source
    elif args.input:
        in_path = args.input

    if args.dest:
        out_path = args.dest
    elif args.output:
        out_path = args.output

    if in_path is None:
        print('Error: no input file provided (positional or --source)', file=sys.stderr)
        raise SystemExit(2)

    try:
        convert(in_path, out_path, overwrite=args.overwrite)
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
