import pandas as pd

# Czytaj bez specjalnej interpretacji
with open('ml/src/data/XAUUSD_M1 (2).csv', 'r') as f:
    lines = f.readlines()
    for i in range(5):
        print(f"Line {i}: {repr(lines[i])}")
