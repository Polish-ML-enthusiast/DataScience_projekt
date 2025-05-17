import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Ustawienia wyjścia
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

# Ścieżka do pliku
RAW_PATH = Path("data/raw/apartments_pl_2024_06.csv")
PROCESSED_PATH = Path("data/processed/apartments_cleaned.csv")

# --- 1. Wczytanie danych ---
df = pd.read_csv(RAW_PATH)
print("\n1. Wczytano dane. Rozmiar ramki danych:", df.shape)

# --- 2. Typy danych ---
print("\n2. Typy danych:")
print(df.dtypes)

# --- 3. Duplikaty ---
duplicates = df.duplicated().sum()
print(f"\n3. Liczba duplikatów: {duplicates}")

# --- 4. Braki danych ---
print("\n4. Braki danych:")
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_table = pd.DataFrame({
    'missing_values': missing,
    'percent': missing_percent
})
print(missing_table[missing_table.missing_values > 0].sort_values(by='percent', ascending=False))

# --- 5. Wizualizacja braków danych ---
plt.figure(figsize=(12, 6))
missing_table[missing_table.missing_values > 0].sort_values(by='percent', ascending=False)['percent'].plot(kind='bar')
plt.title("Procent brakujących wartości w kolumnach")
plt.ylabel("% braków")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 6. Wizualizacja typów danych ---
dtype_counts = df.dtypes.value_counts()
plt.figure(figsize=(6, 4))
dtype_counts.plot(kind='barh')
plt.title("Typy danych w ramce danych")
plt.xlabel("Liczba kolumn")
plt.tight_layout()
plt.show()

# --- 7. Przekształcenia kolumn ---
## Zamiana "yes"/"no" na 1/0
binary_cols = ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']
df[binary_cols] = df[binary_cols].replace({'yes': 1, 'no': 0})

## Miasta z wielkiej litery
df['city'] = df['city'].str.title()

## Uzupełnianie kolumn liczbowych medianą
num_cols_to_fill = ['floor', 'floorCount', 'buildYear']
for col in num_cols_to_fill:
    median_val = df[col].median()
    print(f"Uzupełniono medianą ({median_val}) brakujące dane w kolumnie '{col}'")
    df[col] = df[col].fillna(median_val)

## Uzupełnianie kategorii wartością "unknown"
cat_cols_to_fill = ['type', 'buildingMaterial', 'condition']
for col in cat_cols_to_fill:
    print(f"Uzupełniono wartości 'unknown' w kolumnie '{col}'")
    df[col] = df[col].fillna("unknown")

# --- 8. Dodanie kolumny logarytmicznej ceny ---
df['price_log'] = np.log1p(df['price'])

# --- 9. Zapis oczyszczonych danych ---
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(PROCESSED_PATH, index=False)

print("\nZapisano oczyszczone dane do:", PROCESSED_PATH)
