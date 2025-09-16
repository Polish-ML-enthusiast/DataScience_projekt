# =======================
# DATA_QUALITY_CHECK.py
# =======================

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
from sklearn.model_selection import train_test_split
import json
import warnings

warnings.filterwarnings("ignore")

# =======================
# USTAWIENIA I ŚCIEŻKI
# =======================
pd.set_option("display.max_columns", None)
sns.set(style="whitegrid")

RAW_PATH = Path("data/raw/apartments_pl_2024_06.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_PATH = PROCESSED_DIR / "apartments_train.csv"
TEST_PATH  = PROCESSED_DIR / "apartments_test.csv"
ULICE_PATH = PROCESSED_DIR / "ulice_w_miastach.csv"
REVERSE_CACHE_PATH = PROCESSED_DIR / "reverse_cache.csv"
IMPUTE_PARAMS_PATH = PROCESSED_DIR / "imputation_params.json"

# flaga narzędziowa – przydatna w iteracji roboczej
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
print("\n📥 Wczytywanie danych...")
df = pd.read_csv(RAW_PATH)
print("Kształt danych:", df.shape)

# Sprawdzenie typów danych 
print("\nTypy danych we wszystkich kolumnach przed konwersją:")
print(df.dtypes)

# Konwersja kolumn numerycznych (tylko tam, gdzie potrzeba)
num_cols = ['price', 'squareMeters', 'floor', 'floorCount', 'buildYear']
for col in num_cols:
    if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
        print(f"Konwertuję kolumnę '{col}' na typ numeryczny.")
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"Kolumna '{col}' już jest typu numerycznego lub nie istnieje – pomijam konwersję.")

# =======================
# WIZUALIZACJA BRAKÓW DANYCH — % (0–100) NA SUROWYM ZBIORZE
# =======================
print("\n📊 Braki danych – procent w surowym zbiorze:")
missing_pct = df.isna().mean().sort_values(ascending=False) * 100
missing_pct = missing_pct[missing_pct > 0]

if not missing_pct.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    missing_pct.plot(kind='bar', ax=ax)
    ax.set_title("🔍 Procent braków danych w kolumnach (surowe dane)")
    ax.set_ylabel("Braki [%]")
    ax.set_xlabel("Kolumny")
    ax.set_ylim(0, min(100, missing_pct.max() * 1.15))
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    plt.show()
else:
    print("Brak braków danych do wizualizacji.")

# Wykrywanie błędnych cen i powierzchni
if {'squareMeters','price'}.issubset(df.columns):
    negatives = df[(df['squareMeters'] <= 0) | (df['price'] <= 0)]
    print(f"❗ Znaleziono {len(negatives)} rekordów z ceną lub powierzchnią ≤ 0")
    # Wykres błędnych danych (opcjonalny)
    if not negatives.empty:
        sns.scatterplot(data=negatives, x="squareMeters", y="price")
        plt.title("Błędne wartości: cena i powierzchnia")
        plt.show()

    # Usuwanie błędnych rekordów
    before_removal = df.shape[0]
    df = df[(df['squareMeters'] > 0) & (df['price'] > 0)]
    after_removal = df.shape[0]
    print(f"🧹 Usunięto {before_removal - after_removal} rekordów z błędną ceną lub powierzchnią.")

# Usunięcie duplikatów (w tym po id, jeśli jest)
before_dups = df.shape[0]
df = df.drop_duplicates()
if 'id' in df.columns:
    df = df.drop_duplicates(subset='id')
after_dups = df.shape[0]
print(f"🧽 Usunięto {before_dups - after_dups} duplikatów.")

# Usunięcie zbędnych kolumn
columns_to_drop = [
    "id", "schoolDistance", "clinicDistance", "postOfficeDistance",
    "kindergartenDistance", "restaurantDistance", "collegeDistance",
    "pharmacyDistance", "ownership", "buildingMaterial", "condition"
]
df = df.drop(columns=[c for c in columns_to_drop if c in df.columns], errors="ignore")

# Mapowanie typu (transformacja — nie imputacja)
if 'type' in df.columns:
    df['type'] = df['type'].map(mapa_typow).fillna(df['type'])

# Standaryzacja nazw miast
if 'city' in df.columns:
    df['city'] = df['city'].astype(str).str.title()
    city_name_corrections = {
        "Lodz": "Łódź", "Wroclaw": "Wrocław", "Poznan": "Poznań",
        "Krakow": "Kraków", "Gdansk": "Gdańsk", "Bialystok": "Białystok",
        "Rzeszow": "Rzeszów", "Zielona Gora": "Zielona Góra",
        "Torun": "Toruń", "Plock": "Płock", "Gorzow Wielkopolski": "Gorzów Wielkopolski"
    }
    df['city'] = df['city'].replace(city_name_corrections)

# =======================
# REVERSE GEOCODING (uzupełnienie 'ulica' – używa cache)
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

if skip_ulica_if_exists and df['ulica'].notna().all():
    print("Wszystkie rekordy mają przypisaną ulicę – pomijam reverse geocoding.")
else:
    print("Uzupełniam brakujące ulice (reverse geocoding z cache)...")
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

    if {'latitude','longitude'}.issubset(df.columns):
        missing = df[df['ulica'].isna() & df['latitude'].notna() & df['longitude'].notna()]
        # przygotuj index dla szybkiej kontroli duplikatów
        existing_idx = set()
        if not reverse_cache_df.empty:
            existing_idx = set(
                tuple(x) for x in reverse_cache_df[['lat','lon']].round(6).values.tolist()
            )
        for i, row in tqdm(missing.iterrows(), total=len(missing), desc="Geokodowanie ulic"):
            street = get_street(row['latitude'], row['longitude'])
            df.at[i, 'ulica'] = street
            key = (round(row['latitude'], 6), round(row['longitude'], 6))
            if key not in existing_idx:
                reverse_cache_df.loc[len(reverse_cache_df)] = {
                    'lat': key[0],
                    'lon': key[1],
                    'ulica': street
                }
                existing_idx.add(key)

        reverse_cache_df.drop_duplicates(subset=['lat', 'lon'], inplace=True)
        reverse_cache_df.to_csv(REVERSE_CACHE_PATH, index=False)
        print(f"Cache ulic zapisany do: {REVERSE_CACHE_PATH}")

# =======================
# PRZEKSZTAŁCANIE (przed podziałem)
# =======================
binary_cols = ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({'yes': 1, 'no': 0}).astype('Int64')

if {'price','squareMeters'}.issubset(df.columns):
    df['price_log'] = np.log1p(df['price'])
    df['price_per_m2'] = df['price'] / df['squareMeters']

# =======================
# TŁUMACZENIE KOLUMN NA PL (przed podziałem)
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
df = df.rename(columns=translation)

# =======================
# PODZIAŁ NA TRAIN/TEST (stratyfikacja po mieście)
# =======================
print("\n✂️ Podział na zbiór treningowy i testowy (stratyfikacja po 'miasto')...")
if 'miasto' in df.columns:
    stratify_series = df['miasto'].fillna('Unknown')
else:
    stratify_series = None

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=stratify_series if stratify_series is not None else None
)

print(f"TRAIN: {train_df.shape}, TEST: {test_df.shape}")

# =======================
# IMPUTACJA — TYLKO NA DANYCH TRENINGOWYCH
# =======================
# (feature’y; etykiet nie imputujemy tutaj)
impute_numeric_cols = [c for c in ['pietro','liczba_pieter','rok_budowy'] if c in train_df.columns]
impute_categorical_cols = [c for c in ['typ_nieruchomosci'] if c in train_df.columns]

impute_params = {"numeric_median": {}, "categorical_mode": {}}

for c in impute_numeric_cols:
    med = float(train_df[c].median())
    impute_params["numeric_median"][c] = med
    train_df[c] = train_df[c].fillna(med)

for c in impute_categorical_cols:
    if train_df[c].isna().all():
        mode_val = "nieznany"
    else:
        mode_val = train_df[c].mode(dropna=True)[0]
    impute_params["categorical_mode"][c] = mode_val
    train_df[c] = train_df[c].fillna(mode_val)

# TEST pozostaje z NaN; parametry zapisujemy do pliku
with open(IMPUTE_PARAMS_PATH, "w", encoding="utf-8") as f:
    json.dump(impute_params, f, ensure_ascii=False, indent=2)
print(f"✅ Parametry imputacji zapisane do: {IMPUTE_PARAMS_PATH}")

# =======================
# ZAPIS ZBIORÓW
# =======================
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)
print(f"✅ Zapisano TRAIN do: {TRAIN_PATH}")
print(f"✅ Zapisano TEST  do: {TEST_PATH}")

# =======================
# GENEROWANIE LISTY ULIC (z całego korpusu – metadane)
# =======================
if ULICE_PATH.exists():
    print(f"Plik z ulicami już istnieje: {ULICE_PATH} – pomijam generowanie.")
else:
    print("🏙️ Tworzę listę ulic z OpenStreetMap…")
    if "miasto" not in df.columns:
        print("Brak kolumny 'miasto' — nie generuję listy ulic.")
    else:
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
        print(f"✅ Lista ulic zapisana do: {ULICE_PATH}")
