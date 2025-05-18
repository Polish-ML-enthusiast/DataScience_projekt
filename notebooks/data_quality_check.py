import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Konfiguracja
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

# Ścieżki
RAW_PATH = Path("data/raw/apartments_pl_2024_06.csv")
PROCESSED_PATH = Path("data/processed/apartments_cleaned.csv")

# 1. Wczytanie danych
df = pd.read_csv(RAW_PATH)
print("\n1. Wczytano dane. Rozmiar ramki danych:", df.shape)

# 2. Typy danych
print("\n2. Typy danych:")
print(df.dtypes)

# 3. Duplikaty
duplicates = df.duplicated().sum()
print(f"\n3. Liczba duplikatów: {duplicates}")

# 4. Braki danych
print("\n4. Braki danych:")
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_table = pd.DataFrame({
    'missing_values': missing,
    'percent': missing_percent
})
print(missing_table[missing_table.missing_values > 0].sort_values(by='percent', ascending=False))

# 5. Wizualizacja braków danych
plt.figure(figsize=(12, 6))
missing_table[missing_table.missing_values > 0].sort_values(by='percent', ascending=False)['percent'].plot(kind='bar')
plt.title("Procent brakujących wartości w kolumnach")
plt.ylabel("% braków")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. Wizualizacja typów danych
dtype_counts = df.dtypes.value_counts()
plt.figure(figsize=(6, 4))
dtype_counts.plot(kind='barh')
plt.title("Typy danych w ramce danych")
plt.xlabel("Liczba kolumn")
plt.tight_layout()
plt.show()

# 7. Czyszczenie danych

# Zamiana 'yes'/'no' na 1/0 i konwersja typu
binary_cols = ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0}).astype('Int64')

# Miasta z wielkiej litery
df['city'] = df['city'].str.title()

# Tłumaczenie nazw miast (polskie znaki)
city_name_corrections = {
    "Lodz": "Łódź",
    "Wroclaw": "Wrocław",
    "Poznan": "Poznań",
    "Krakow": "Kraków",
    "Gdansk": "Gdańsk",
    "Bialystok": "Białystok",
    "Rzeszow": "Rzeszów",
    "Zielona Gora": "Zielona Góra",
    "Torun": "Toruń",
    "Plock": "Płock",
    "Gorzow Wielkopolski": "Gorzów Wielkopolski"
}
df['city'] = df['city'].replace(city_name_corrections)

# Uzupełnianie wartości liczbowych medianą
num_cols_to_fill = ['floor', 'floorCount', 'buildYear']
for col in num_cols_to_fill:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# Kategoryczne: brakujące jako "unknown"
cat_cols_to_fill = ['type', 'buildingMaterial', 'condition']
for col in cat_cols_to_fill:
    df[col] = df[col].fillna("unknown")

# Kolumny pomocnicze
df['price_log'] = np.log1p(df['price'])
df['price_per_m2'] = df['price'] / df['squareMeters']

# 8. Tłumaczenie nazw kolumn (na końcu!)
column_translations = {
    "id": "id",
    "city": "miasto",
    "type": "typ_nieruchomosci",
    "squareMeters": "powierzchnia_m2",
    "rooms": "liczba_pokoi",
    "floor": "pietro",
    "floorCount": "liczba_pieter",
    "buildYear": "rok_budowy",
    "latitude": "szerokosc_geo",
    "longitude": "dlugosc_geo",
    "centreDistance": "dystans_do_centrum_km",
    "poiCount": "liczba_punktow_poi",
    "schoolDistance": "odleglosc_szkola_km",
    "clinicDistance": "odleglosc_przychodnia_km",
    "postOfficeDistance": "odleglosc_poczta_km",
    "kindergartenDistance": "odleglosc_przedszkole_km",
    "restaurantDistance": "odleglosc_restauracja_km",
    "collegeDistance": "odleglosc_uczelnia_km",
    "pharmacyDistance": "odleglosc_apteka_km",
    "ownership": "forma_wlasnosci",
    "buildingMaterial": "material_budynku",
    "condition": "stan_techniczny",
    "hasParkingSpace": "miejsce_parkingowe",
    "hasBalcony": "balkon",
    "hasElevator": "winda",
    "hasSecurity": "ochrona",
    "hasStorageRoom": "komorka_lokatorska",
    "price": "cena",
    "price_log": "cena_log",
    "price_per_m2": "cena_za_m2"
}
df.rename(columns=column_translations, inplace=True)

# 9. Zapis
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(PROCESSED_PATH, index=False)
print("\nZapisano oczyszczone dane do:", PROCESSED_PATH)
