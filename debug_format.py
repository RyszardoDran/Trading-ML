import pandas as pd

# Spróbuj różne separatory
df_whitespace = pd.read_csv('ml/src/data/XAUUSD_M1 (2).csv', sep=r'\s+', engine='python')
print("Whitespace separator:")
print(f"  Columns: {df_whitespace.columns.tolist()}")
print(f"  Shape: {df_whitespace.shape}")
print(f"  First row:\n{df_whitespace.iloc[0]}\n")

df_tab = pd.read_csv('ml/src/data/XAUUSD_M1 (2).csv', sep='\t', engine='python')
print("Tab separator:")
print(f"  Columns: {df_tab.columns.tolist()}")
print(f"  Shape: {df_tab.shape}")
print(f"  First row:\n{df_tab.iloc[0]}\n")
