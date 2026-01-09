import pandas as pd

# Czytaj plik
df = pd.read_csv('ml/src/data/XAU_1m_data_2026_additional.csv', sep=';')

print(f"Przed filtrowaniem: {len(df)} wierszy")
print(f"Próbka dat:\n{df['Date'].head(10).tolist()}")
print(f"Próbka dat (koniec):\n{df['Date'].tail(5).tolist()}")

# Filtruj - zostaw tylko wiersze z 2026
df_filtered = df[~df['Date'].str.startswith('2025.')].copy()

print(f"\nPo filtrowaniu: {len(df_filtered)} wierszy")
if len(df_filtered) > 0:
    print(f"Nowe daty (początek):\n{df_filtered['Date'].head(3).tolist()}")
    print(f"Nowe daty (koniec):\n{df_filtered['Date'].tail(3).tolist()}")

# Zapisz
df_filtered.to_csv('ml/src/data/XAU_1m_data_2026_additional.csv', sep=';', index=False)

print(f"\n✓ Zapisano {len(df_filtered)} wierszy do pliku")
