import pandas as pd

df = pd.read_csv('ml/src/data/XAUUSD_M1 (2).csv', sep='\t', engine='python')
print("Czytane kolumny:", df.columns.tolist())
print("Shape:", df.shape)
print("\nPierwszy wiersz:")
for col in df.columns:
    print(f"  {col}: {df.iloc[0][col]}")
