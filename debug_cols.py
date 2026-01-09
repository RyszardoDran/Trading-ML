import pandas as pd

df = pd.read_csv('ml/src/data/XAUUSD_M1 (2).csv', sep='\t', engine='python')
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("\nFirst row:")
print(df.iloc[0])
