import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point

# Ścieżka do danych
DATA_PATH = Path("data/processed/apartments_cleaned.csv")
MAP_PATH = Path("data/maps/ne_10m_admin_0_countries.shp")
REGIONS_PATH = Path("data/maps/wojewodztwa.shp")  # plik z granicami województw

# Wczytanie danych
df = pd.read_csv(DATA_PATH)

st.title("📊 Eksploracyjna Analiza Danych - EDA")

# --- 1. Podstawowe informacje ---
st.header("Podstawowe podsumowanie danych")
st.write(f"Liczba rekordów: {len(df)}")
st.write(f"Dostępne miasta: {', '.join(sorted(df['city'].unique()))}")
st.write(f"Najmniejsza powierzchnia: {df['squareMeters'].min()} m²")
st.write(f"Największa powierzchnia: {df['squareMeters'].max()} m²")
st.write(f"Najczęstsza powierzchnia: {df['squareMeters'].mode()[0]} m²")

# --- 2. Liczba ofert na miasto ---
st.header("Liczba ofert według miasta")
fig1, ax1 = plt.subplots(figsize=(10, 5))
df['city'].value_counts().plot(kind='bar', ax=ax1, color='#1f77b4')
ax1.set_title("Liczba ofert na miasto")
ax1.set_xlabel("Miasto")
ax1.set_ylabel("Liczba ofert")
ax1.tick_params(axis='x', rotation=45)
st.pyplot(fig1)

# --- 3. Mapa Polski z oznaczonymi miastami i województwami ---
st.header("Mapa lokalizacji miast z ofertami")
city_grouped = df.groupby("city").agg({"latitude": "mean", "longitude": "mean", "city": "count"}).rename(columns={"city": "count"}).reset_index()

try:
    world = gpd.read_file(MAP_PATH)
    poland = world[world["NAME"] == "Poland"]

    geometry = [Point(xy) for xy in zip(city_grouped.longitude, city_grouped.latitude)]
    cities_gdf = gpd.GeoDataFrame(city_grouped, geometry=geometry, crs="EPSG:4326")

    regions = gpd.read_file(REGIONS_PATH)
    regions = regions.to_crs("EPSG:4326")

    fig2, ax2 = plt.subplots(figsize=(10, 12))
    poland.plot(ax=ax2, color='#f0f0f0', edgecolor='#444444', linewidth=1.2)
    regions.boundary.plot(ax=ax2, color='gray', linewidth=1, linestyle='--')

    # Zmienna wielkość punktów wg liczby rekordów
    size_scaled = cities_gdf['count'] / cities_gdf['count'].max() * 300
    cities_gdf.plot(ax=ax2, color='crimson', markersize=size_scaled, edgecolor='black', alpha=0.8)

    for x, y, label in zip(city_grouped.longitude, city_grouped.latitude, city_grouped.city):
        ax2.text(x + 0.2, y, label, fontsize=9, ha='left', va='center', fontweight='bold')

    ax2.set_title("Miasta z ofertami w zbiorze danych (wielkość punktu = liczba ofert)", fontsize=14, fontweight='bold')
    ax2.axis('off')
    st.pyplot(fig2)

except Exception as e:
    st.warning("Nie udało się wczytać mapy Polski lub granic województw. Upewnij się, że odpowiednie pliki SHP znajdują się w folderze `data/maps/`.")
    st.error(e)
