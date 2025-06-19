import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import osmnx as ox

# Konfiguracja
pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

# Ścieżki
RAW_PATH = Path("data/raw/apartments_pl_2024_06.csv")
PROCESSED_PATH = Path("data/processed/apartments_cleaned.csv")
ULICE_PATH = Path("data/processed/ulice_w_miastach.csv")

# Flaga wymuszenia ponownego przetworzenia
force_reclean = False

# Mapowanie typów nieruchomości
mapa_typow = {
    'blockOfFlats': 'blok mieszkalny',
    'apartmentBuilding': 'budynek apartamentowy',
    'tenement': 'kamienica',
    'unknown': 'nieznany'
}

# Wczytanie lub przetworzenie danych
if PROCESSED_PATH.exists() and not force_reclean:
    print(f"📄 Wczytuję dane z: {PROCESSED_PATH}")
    df = pd.read_csv(PROCESSED_PATH)

    if 'typ_nieruchomosci' in df.columns:
        df['typ_nieruchomosci'] = df['typ_nieruchomosci'].map(mapa_typow).fillna(df['typ_nieruchomosci'])
else:
    print("📥 Wczytywanie surowych danych i czyszczenie…")
    df = pd.read_csv(RAW_PATH)
    print("\n1. Wczytano dane. Rozmiar ramki danych:", df.shape)

    binary_cols = ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0}).astype('Int64')

    df['city'] = df['city'].str.title()
    city_name_corrections = {
        "Lodz": "Łódź", "Wroclaw": "Wrocław", "Poznan": "Poznań",
        "Krakow": "Kraków", "Gdansk": "Gdańsk", "Bialystok": "Białystok",
        "Rzeszow": "Rzeszów", "Zielona Gora": "Zielona Góra",
        "Torun": "Toruń", "Plock": "Płock", "Gorzow Wielkopolski": "Gorzów Wielkopolski"
    }
    df['city'] = df['city'].replace(city_name_corrections)

    for col in ['floor', 'floorCount', 'buildYear']:
        df[col] = df[col].fillna(df[col].median())

    df['type'] = df['type'].fillna("unknown")
    df['price_log'] = np.log1p(df['price'])
    df['price_per_m2'] = df['price'] / df['squareMeters']

    df['type'] = df['type'].map(mapa_typow).fillna(df['type'])

    if 'ulica' not in df.columns:
        df['ulica'] = None

    columns_to_drop = [
        "id", "schoolDistance", "clinicDistance", "postOfficeDistance",
        "kindergartenDistance", "restaurantDistance", "collegeDistance",
        "pharmacyDistance", "ownership", "buildingMaterial", "condition"
    ]
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    column_translations = {
        "city": "miasto", "type": "typ_nieruchomosci", "squareMeters": "powierzchnia_m2",
        "rooms": "liczba_pokoi", "floor": "pietro", "floorCount": "liczba_pieter",
        "buildYear": "rok_budowy", "latitude": "szerokosc_geo", "longitude": "dlugosc_geo",
        "centreDistance": "dystans_do_centrum_km", "poiCount": "liczba_punktow_poi",
        "hasParkingSpace": "miejsce_parkingowe", "hasBalcony": "balkon",
        "hasElevator": "winda", "hasSecurity": "ochrona", "hasStorageRoom": "komorka_lokatorska",
        "price": "cena", "price_log": "cena_log", "price_per_m2": "cena_za_m2", "ulica": "ulica"
    }
    df.rename(columns=column_translations, inplace=True)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"✅ Zapisano dane do: {PROCESSED_PATH}")

if df['ulica'].isna().sum() > 0:
    print("\n🔍 Uzupełniam brakujące ulice na podstawie współrzędnych…")

    geolocator = Nominatim(user_agent="real-estate-reverse")
    geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1, error_wait_seconds=2.0)
    reverse_cache = {}

    def get_street_cached(lat, lon):
        key = (round(lat, 6), round(lon, 6))
        if key in reverse_cache:
            return reverse_cache[key]
        try:
            location = geocode((lat, lon), language='pl', exactly_one=True, timeout=10)
            if location:
                address = location.raw.get("address", {})
                for key_part in ["road", "pedestrian", "footway", "path", "residential"]:
                    if key_part in address:
                        reverse_cache[key] = address[key_part]
                        return address[key_part]
            reverse_cache[key] = None
            return None
        except Exception as e:
            print(f"❌ Błąd dla ({lat}, {lon}): {e}")
            reverse_cache[key] = None
            return None

    to_process = df[df['ulica'].isna() & df['szerokosc_geo'].notna() & df['dlugosc_geo'].notna()]
    print(f"🔄 Do uzupełnienia: {len(to_process)} rekordów")

    for i, row in enumerate(to_process.itertuples(), 1):
        idx = row.Index
        ulica = get_street_cached(row.szerokosc_geo, row.dlugosc_geo)
        df.at[idx, "ulica"] = ulica
        if i % 50 == 0 or i == len(to_process):
            print(f"🔹 Przetworzono {i}/{len(to_process)}...")

    df.to_csv(PROCESSED_PATH, index=False)
    print(f"✅ Zapisano ulice do: {PROCESSED_PATH}")
else:
    print("✅ Wszystkie rekordy mają przypisaną ulicę – pomijam reverse geocoding.")

if ULICE_PATH.exists():
    print(f"📄 Plik z ulicami już istnieje: {ULICE_PATH} – pomijam generowanie.")
else:
    print("\n🌍 Generowanie listy ulic z OpenStreetMap dla wszystkich miast…")
    miasta = df["miasto"].dropna().unique()
    ulice_data = []

    for miasto in miasta:
        print(f"📦 Miasto: {miasto}")
        try:
            G = ox.graph_from_place(f"{miasto}, Poland", network_type="all", simplify=True)
            edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
            if 'name' in edges.columns:
                names = edges['name'].dropna().apply(lambda x: x if isinstance(x, list) else [x])
                for name_list in names:
                    for ulica in name_list:
                        if isinstance(ulica, str) and len(ulica.strip()) > 1:
                            ulice_data.append({"miasto": miasto, "ulica": ulica.strip()})
        except Exception as e:
            print(f"❌ Błąd dla miasta {miasto}: {e}")

    df_ulice = pd.DataFrame(ulice_data).drop_duplicates().sort_values(["miasto", "ulica"])
    ULICE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_ulice.to_csv(ULICE_PATH, index=False, encoding="utf-8")
    print(f"✅ Zapisano listę {len(df_ulice)} ulic do: {ULICE_PATH}")
