import pandas as pd

# Czytaj wszystkie linie
with open('ml/src/data/XAUUSD_M1 (2).csv', 'r') as f:
    lines = f.readlines()

# Napraw header - dodaj kolumnę
header = lines[0].rstrip('\n') + '\tExtra\n'

# Napisz fixed version
with open('ml/src/data/XAUUSD_M1_fixed.csv', 'w') as f:
    f.write(header)
    f.writelines(lines[1:])

print("✓ Plik naprawiony")

# Verify
df = pd.read_csv('ml/src/data/XAUUSD_M1_fixed.csv', sep='\t')
print(f"Kolumny: {df.columns.tolist()}")
print(f"Shape: {df.shape}")
print(f"\nFirst row:")
print(df.iloc[0])
