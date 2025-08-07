# =======================
# IMPORT BIBLIOTEK
# =======================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import osmnx as ox
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm

# =======================
# USTAWIENIA I ŚCIEŻKI
# =======================
pd.set_option("display.max_columns", None)
sns.set(style="whitegrid")

RAW_PATH = Path("data/raw/apartments_pl_2024_06.csv")
PROCESSED_PATH = Path("data/processed/apartments_cleaned.csv")
ULICE_PATH = Path("data/processed/ulice_w_miastach.csv")
REVERSE_CACHE_PATH = Path("data/processed/reverse_cache.csv")

force_reclean = True
skip_ulica_if_exists = True

mapa_typow = {
    'blockOfFlats': 'blok mieszkalny',
    'apartmentBuilding': 'budynek apartamentowy',
    'tenement': 'kamienica',
    'unknown': 'nieznany'
}

# =======================
# WCZYTYWANIE I CZYSZCZENIE
# =======================
print("\n Wczytywanie danych...")
df = pd.read_csv(RAW_PATH)
print("Kształt danych:", df.shape)

# Sprawdzenie typów danych 
print("\nTypy danych we wszystkich kolumnach przed konwersją:")
print(df.dtypes)

# Konwersja kolumn numerycznych
num_cols = ['price', 'squareMeters', 'floor', 'floorCount', 'buildYear']

for col in num_cols:
    if not pd.api.types.is_numeric_dtype(df[col]):
        print(f"Konwertuję kolumnę '{col}' na typ numeryczny.")
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"Kolumna '{col}' już jest typu numerycznego – pomijam konwersję.")

# Wizualizacja braków danych
print("\n📊 Braki danych:")
missing_counts = df.isnull().sum()
missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

if not missing_counts.empty:
    plt.figure(figsize=(10, 6))
    missing_counts.plot(kind='bar')
    plt.title("🔍 Liczba braków danych w kolumnach")
    plt.ylabel("Liczba braków")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
else:
    print("Brak braków danych do wizualizacji.")

# Wykrywanie błędnych cen i powierzchni
negatives = df[(df['squareMeters'] <= 0) | (df['price'] <= 0)]
print(f"❗ Znaleziono {len(negatives)} rekordów z ceną lub powierzchnią ≤ 0")

# Wykres błędnych danych
if not negatives.empty:
    sns.scatterplot(data=negatives, x="squareMeters", y="price")
    plt.title("Błędne wartości: cena i powierzchnia")
    plt.show()

# Usuwanie błędnych rekordów
before_removal = df.shape[0]
df = df[(df['squareMeters'] > 0) & (df['price'] > 0)]
after_removal = df.shape[0]
print(f"🧹 Usunięto {before_removal - after_removal} rekordów z błędną ceną lub powierzchnią.")

# Usunięcie duplikatów
before_dups = df.shape[0]
df.drop_duplicates(inplace=True)
df.drop_duplicates(subset='id', inplace=True)
after_dups = df.shape[0]
print(f"Usunięto {before_dups - after_dups} duplikatów.")

# Usunięcie zbędnych kolumn
columns_to_drop = [
    "id", "schoolDistance", "clinicDistance", "postOfficeDistance",
    "kindergartenDistance", "restaurantDistance", "collegeDistance",
    "pharmacyDistance", "ownership", "buildingMaterial", "condition"
]
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Imputacja medianą
df['floor'] = df['floor'].fillna(df['floor'].median())
df['floorCount'] = df['floorCount'].fillna(df['floorCount'].median())
df['buildYear'] = df['buildYear'].fillna(df['buildYear'].median())
df['type'] = df['type'].fillna('unknown')
df['type'] = df['type'].map(mapa_typow).fillna(df['type'])

# Standaryzacja nazw miast
df['city'] = df['city'].str.title()
city_name_corrections = {
    "Lodz": "Łódź", "Wroclaw": "Wrocław", "Poznan": "Poznań",
    "Krakow": "Kraków", "Gdansk": "Gdańsk", "Bialystok": "Białystok",
    "Rzeszow": "Rzeszów", "Zielona Gora": "Zielona Góra",
    "Torun": "Toruń", "Plock": "Płock", "Gorzow Wielkopolski": "Gorzów Wielkopolski"
}
df['city'] = df['city'].replace(city_name_corrections)

# =======================
# PRZEKSZTAŁCANIE
# =======================
binary_cols = ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({'yes': 1, 'no': 0}).astype('Int64')

df['price_log'] = np.log1p(df['price'])
df['price_per_m2'] = df['price'] / df['squareMeters']

# =======================
# REVERSE GEOCODING
# =======================
if 'ulica' not in df.columns:
    df['ulica'] = None

if REVERSE_CACHE_PATH.exists():
    reverse_cache_df = pd.read_csv(REVERSE_CACHE_PATH)
    reverse_cache = {
        (round(row['lat'], 6), round(row['lon'], 6)): row['ulica']
        for _, row in reverse_cache_df.iterrows()
    }
else:
    reverse_cache_df = pd.DataFrame(columns=['lat', 'lon', 'ulica'])
    reverse_cache = {}

if skip_ulica_if_exists and 'ulica' in df.columns and df['ulica'].notna().all():
    print("Wszystkie rekordy mają przypisaną ulicę – pomijam reverse geocoding.")
else:
    print("Uzupełniam brakujące ulice...")
    geolocator = Nominatim(user_agent="real-estate-app")
    geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1)

    def get_street(lat, lon):
        key = (round(lat, 6), round(lon, 6))
        if key in reverse_cache:
            return reverse_cache[key]
        try:
            location = geocode((lat, lon), language='pl')
            if location:
                address = location.raw.get("address", {})
                for k in ["road", "pedestrian", "footway", "path", "residential"]:
                    if k in address:
                        reverse_cache[key] = address[k]
                        return address[k]
            reverse_cache[key] = None
        except:
            reverse_cache[key] = None
        return None

    missing = df[df['ulica'].isna() & df['latitude'].notna() & df['longitude'].notna()]
    for i, row in tqdm(missing.iterrows(), total=len(missing), desc="Geokodowanie ulic"):
        street = get_street(row['latitude'], row['longitude'])
        df.at[i, 'ulica'] = street
        key = (round(row['latitude'], 6), round(row['longitude'], 6))
        if key not in reverse_cache_df.set_index(['lat', 'lon']).index:
            reverse_cache_df.loc[len(reverse_cache_df)] = {
                'lat': key[0],
                'lon': key[1],
                'ulica': street
            }

    reverse_cache_df.drop_duplicates(subset=['lat', 'lon'], inplace=True)
    reverse_cache_df.to_csv(REVERSE_CACHE_PATH, index=False)
    print(f"Cache ulic zapisany do: {REVERSE_CACHE_PATH}")

# =======================
# ZAPIS PRZETWORZONYCH DANYCH
# =======================
translation = {
    "city": "miasto", "type": "typ_nieruchomosci", "squareMeters": "powierzchnia_m2",
    "rooms": "liczba_pokoi", "floor": "pietro", "floorCount": "liczba_pieter",
    "buildYear": "rok_budowy", "latitude": "szerokosc_geo", "longitude": "dlugosc_geo",
    "centreDistance": "dystans_do_centrum_km", "poiCount": "liczba_punktow_poi",
    "hasParkingSpace": "miejsce_parkingowe", "hasBalcony": "balkon",
    "hasElevator": "winda", "hasSecurity": "ochrona", "hasStorageRoom": "komorka_lokatorska",
    "price": "cena", "price_log": "cena_log", "price_per_m2": "cena_za_m2", "ulica": "ulica"
}
df.rename(columns=translation, inplace=True)
df.to_csv(PROCESSED_PATH, index=False)
print(f"Dane zapisano do {PROCESSED_PATH}")

# =======================
# GENEROWANIE LISTY ULIC
# =======================
if ULICE_PATH.exists():
    print(f"Plik z ulicami już istnieje: {ULICE_PATH} – pomijam generowanie.")
else:
    print("Tworzę listę ulic z OpenStreetMap…")
    miasta = df["miasto"].dropna().unique()
    ulice_data = []

    for miasto in miasta:
        try:
            G = ox.graph_from_place(f"{miasto}, Poland", network_type="all", simplify=True)
            edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
            if 'name' in edges.columns:
                names = edges['name'].dropna().apply(lambda x: x if isinstance(x, list) else [x])
                for nlist in names:
                    for ulica in nlist:
                        if isinstance(ulica, str) and len(ulica.strip()) > 1:
                            ulice_data.append({"miasto": miasto, "ulica": ulica.strip()})
        except Exception as e:
            print(f"Błąd: {miasto} – {e}")

    df_ulice = pd.DataFrame(ulice_data).drop_duplicates().sort_values(["miasto", "ulica"])
    ULICE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_ulice.to_csv(ULICE_PATH, index=False, encoding="utf-8")
    print(f"Lista ulic zapisana do: {ULICE_PATH}")
