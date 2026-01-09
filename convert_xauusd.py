#!/usr/bin/env python3
"""Konwertuj XAUUSD_M1 (2).csv do docelowego formatu 2026."""
import pandas as pd
from pathlib import Path

# Odczytaj z poprawą dla dodatkowej kolumny
with open('ml/src/data/XAUUSD_M1 (2).csv', 'r') as f:
    lines = f.readlines()

header = lines[0].rstrip('\n').split('\t') + ['Extra']
data_lines = [l.rstrip('\n').split('\t') for l in lines[1:]]

# Zbuduj DataFrame
df = pd.DataFrame(data_lines, columns=header)

# Kolumny: Time, Open, High, Low, Close, Volume, Extra
# Konwertuj do docelowego formatu
df_out = pd.DataFrame()
df_out['Date'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y.%m.%d %H:%M')
df_out['Open'] = pd.to_numeric(df['Open'], errors='coerce')
df_out['High'] = pd.to_numeric(df['High'], errors='coerce')
df_out['Low'] = pd.to_numeric(df['Low'], errors='coerce')
df_out['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df_out['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0).astype(int)

# Sortuj
df_out = df_out.sort_values('Date')

# Zapisz
df_out.to_csv('ml/src/data/XAU_1m_data_2026_additional.csv', sep=';', index=False, float_format='%.6f')

print(f"✓ Skonwertowano {len(df_out)} wierszy")
print(f"Format: {df_out.iloc[0].to_dict()}")
