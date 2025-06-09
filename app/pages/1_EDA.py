import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import osmnx as ox
from geopy.distance import geodesic
from pathlib import Path

# Ścieżki do danych
DATA_PATH = Path("data/processed/apartments_cleaned.csv")
MAP_PATH = Path("data/maps/ne_10m_admin_0_countries.shp")
REGIONS_PATH = Path("data/maps/wojewodztwa.shp")

# Buforowanie danych
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.title("📊 Eksploracyjna Analiza Danych (EDA)")

# 1. Podstawowe informacje
st.header("📌 Podstawowe informacje o danych")
st.write(f"Liczba rekordów: {len(df)}")
st.write(f"Dostępne miasta: {', '.join(sorted(df['miasto'].unique()))}")
st.write(f"Zakres powierzchni: {df['powierzchnia_m2'].min()} m² – {df['powierzchnia_m2'].max()} m²")

# 2. Liczba ofert według miasta
st.header("🏙️ Liczba ofert według miasta")

fig1, ax1 = plt.subplots(figsize=(10, 5))
df['miasto'].value_counts().plot(kind='bar', ax=ax1, color='#1f77b4')
ax1.set_title("Liczba ofert na miasto")
ax1.set_xlabel("Miasto")
ax1.set_ylabel("Liczba ofert")
ax1.tick_params(axis='x', rotation=45)
sns.despine(ax=ax1)
st.pyplot(fig1)

st.subheader("📊 Liczba ofert według miasta i rodzaju nieruchomości")
sorted_cities = sorted(df['miasto'].unique())
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.countplot(data=df, x='miasto', hue='typ_nieruchomosci', ax=ax2, order=sorted_cities)
ax2.set_title("Liczba ofert na miasto z podziałem na typ nieruchomości")
ax2.set_xlabel("Miasto")
ax2.set_ylabel("Liczba ofert")
ax2.tick_params(axis='x', rotation=45)
ax2.legend(title="Typ nieruchomości", bbox_to_anchor=(1.05, 1), loc='upper left')
sns.despine(ax=ax2)
st.pyplot(fig2)

# 3. Histogram ceny za 1 m²
st.header("💰 Analiza ceny za 1 m²")
try:
    df_valid = df[
        df['cena_za_m2'].notna()
        & (df['cena_za_m2'] < 30000)
        & np.isfinite(df['cena_za_m2'])
    ]

    if df_valid.empty:
        st.warning("⚠️ Brak danych do analizy.")
    else:
        st.subheader("📊 Kluczowe statystyki ceny za 1 m²")
        stats = {
            "Średnia": round(df_valid['cena_za_m2'].mean(), 2),
            "Mediana": round(df_valid['cena_za_m2'].median(), 2),
            "Max": round(df_valid['cena_za_m2'].max(), 2),
            "Min": round(df_valid['cena_za_m2'].min(), 2),
            "Odch. standardowe": round(df_valid['cena_za_m2'].std(), 2),
            "Liczba ofert": int(df_valid['cena_za_m2'].count())
        }
        for key, val in stats.items():
            st.write(f"**{key}**: {val}")

        st.subheader("📈 Histogram ceny za 1 m²")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.histplot(df_valid['cena_za_m2'], bins=15, kde=True, color='skyblue', ax=ax2)
        ax2.set_title("Histogram ceny za 1 m²")
        ax2.set_xlabel("Cena za 1 m² (PLN)")
        ax2.set_ylabel("Liczba ogłoszeń")
        ax2.grid(False)
        sns.despine(ax=ax2)
        st.pyplot(fig2)
except Exception as e:
    st.error("❌ Błąd podczas generowania danych:")
    st.code(str(e))

# 4. Mapa lokalizacji miast
st.header("🗺️ Mapa lokalizacji miast z ofertami")
try:
    city_grouped = df.groupby("miasto").agg({
        "szerokosc_geo": "mean",
        "dlugosc_geo": "mean",
        "miasto": "count"
    }).rename(columns={"miasto": "count"}).reset_index()

    geometry = [Point(xy) for xy in zip(city_grouped.dlugosc_geo, city_grouped.szerokosc_geo)]
    cities_gdf = gpd.GeoDataFrame(city_grouped, geometry=geometry, crs="EPSG:4326")

    world = gpd.read_file(MAP_PATH)
    poland = world[world["NAME"] == "Poland"]
    regions = gpd.read_file(REGIONS_PATH).to_crs("EPSG:4326")

    fig3, ax3 = plt.subplots(figsize=(10, 12))
    poland.plot(ax=ax3, color='#f0f0f0', edgecolor='#444444')
    regions.boundary.plot(ax=ax3, color='gray', linestyle='--')
    size_scaled = cities_gdf['count'] / cities_gdf['count'].max() * 300
    cities_gdf.plot(ax=ax3, color='crimson', markersize=size_scaled, edgecolor='black', alpha=0.8)

    for x, y, label in zip(city_grouped.dlugosc_geo, city_grouped.szerokosc_geo, city_grouped.miasto):
        ax3.text(x + 0.2, y, label, fontsize=9, ha='left', va='center', fontweight='bold')

    ax3.set_title("Miasta z ofertami (wielkość punktu = liczba ofert)")
    ax3.axis('off')
    st.pyplot(fig3)
except Exception as e:
    st.warning("Nie udało się wygenerować mapy.")
    st.code(str(e))

# 5. Macierz korelacji
st.header("🔍 Macierz korelacji")
numeric_df = df.select_dtypes(include=['float64', 'int64']).copy()

if numeric_df.isnull().any().any():
    numeric_df = numeric_df.fillna(0)

corr_matrix = numeric_df.corr()

st.write("Kształt macierzy korelacji:", corr_matrix.shape)
st.write("Czy macierz zawiera NaN?", corr_matrix.isnull().any().any())

st.subheader("Tabela korelacji:")
st.dataframe(corr_matrix)

st.subheader("Mapa cieplna korelacji:")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f", ax=ax)
st.pyplot(fig)

# 6. Interaktywny wykres zależności
st.header("📈 Interaktywny wykres zależności (Scatterplot)")

numeric_columns = numeric_df.columns.tolist()
default_x = numeric_columns.index("powierzchnia_m2") if "powierzchnia_m2" in numeric_columns else 0
default_y = numeric_columns.index("cena") if "cena" in numeric_columns else 1

if len(numeric_columns) < 2:
    st.warning("⚠️ Za mało kolumn numerycznych do stworzenia scatterplotu.")
else:
    col_x = st.selectbox("Wybierz kolumnę na oś X", options=numeric_columns, index=default_x)
    col_y = st.selectbox("Wybierz kolumnę na oś Y", options=numeric_columns, index=default_y)

    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x=col_x, y=col_y, alpha=0.6, ax=ax_scatter)
    ax_scatter.set_title(f"Zależność: {col_x} vs {col_y}")
    ax_scatter.set_xlabel(col_x)
    ax_scatter.set_ylabel(col_y)
    ax_scatter.grid(True)
    sns.despine(ax=ax_scatter)
    st.pyplot(fig_scatter)

# 7. Mapa z dynamicznymi POI i dynamicznymi ulicami z OSM
st.header("🗺️ Mapa nieruchomości i dynamicznych POI z OSM")
miasto = st.selectbox("Wybierz miasto:", df["miasto"].unique())

# Pobieranie ulic z OSM
st.info("Pobieranie ulic z OSM…")
try:
    G = ox.graph_from_place(miasto + ", Poland", network_type="all")
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    ulice = edges['name'].dropna().unique().tolist()
    if not ulice:
        st.warning("Nie znaleziono ulic w tym mieście – spróbuj inne miasto.")
        ulica = None
    else:
        ulica = st.selectbox("Wybierz ulicę:", sorted(ulice))
except Exception as e:
    st.error(f"Błąd podczas pobierania ulic: {e}")
    ulica = None

# Filtr ofert tylko po mieście (bo nie ma kolumny "ulica")
nieruchomosci = df[df["miasto"] == miasto]

if not nieruchomosci.empty:
    # Jeśli jest wybrana ulica, znajdź środek tej ulicy z OSM
    if ulica:
        ulica_edges = edges[edges['name'] == ulica]
        if not ulica_edges.empty:
            center_lat = ulica_edges.geometry.centroid.y.mean()
            center_lon = ulica_edges.geometry.centroid.x.mean()
        else:
            center_lat = nieruchomosci["szerokosc_geo"].mean()
            center_lon = nieruchomosci["dlugosc_geo"].mean()
    else:
        center_lat = nieruchomosci["szerokosc_geo"].mean()
        center_lon = nieruchomosci["dlugosc_geo"].mean()

    promien = st.slider("Wybierz promień (km):", 0.5, 10.0, 2.0, step=0.5)

    mapa = folium.Map(location=[center_lat, center_lon], zoom_start=15)
    folium.Circle(
        location=[center_lat, center_lon],
        radius=promien * 1000,
        color='blue',
        fill=True,
        fill_opacity=0.1,
    ).add_to(mapa)

    nieruchomosci_cluster = MarkerCluster().add_to(mapa)
    for _, row in nieruchomosci.iterrows():
        folium.Marker(
            location=[row["szerokosc_geo"], row["dlugosc_geo"]],
            popup=f"Cena: {row['cena']} zł\nPowierzchnia: {row['powierzchnia_m2']} m2",
            icon=folium.Icon(color="green", icon="home")
        ).add_to(nieruchomosci_cluster)

    st.info("Pobieranie POI z OSM…")
    poi_tags = {
        'amenity': ['school', 'hospital', 'kindergarten', 'pharmacy', 'restaurant', 'cafe'],
        'shop': True
    }

    poi_gdf = ox.geometries_from_point((center_lat, center_lon), tags=poi_tags, dist=promien * 1000)
    if not poi_gdf.empty:
        for _, row in poi_gdf.iterrows():
            if row.geometry.geom_type == 'Point':
                poi_lat = row.geometry.y
                poi_lon = row.geometry.x
                poi_name = row.get('name', 'Brak nazwy')
                poi_type = row.get('amenity') or row.get('shop', 'unknown')
                distance = geodesic((center_lat, center_lon), (poi_lat, poi_lon)).km
                if distance <= promien:
                    folium.Marker(
                        location=[poi_lat, poi_lon],
                        popup=f"{poi_type.capitalize()}: {poi_name}\nOdległość: {distance:.2f} km",
                        icon=folium.Icon(color="red", icon="info-sign")
                    ).add_to(mapa)
    else:
        st.warning("Brak POI w tym promieniu – spróbuj zwiększyć promień.")

    st_folium(mapa, width=900, height=600)
else:
    st.warning("Brak nieruchomości w wybranym mieście.")
