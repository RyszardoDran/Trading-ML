import pandas as pd

# Czytaj bez header
df = pd.read_csv('ml/src/data/XAUUSD_M1 (2).csv', sep='\t', engine='python', header=None)
print("Bez headera:")
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("\nFirst 2 rows:")
print(df.head(2))
