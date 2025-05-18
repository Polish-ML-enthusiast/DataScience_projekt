import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

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

# Wykres 1: ogólna liczba ofert na miasto
fig1, ax1 = plt.subplots(figsize=(10, 5))
df['miasto'].value_counts().plot(kind='bar', ax=ax1, color='#1f77b4')
ax1.set_title("Liczba ofert na miasto")
ax1.set_xlabel("Miasto")
ax1.set_ylabel("Liczba ofert")
ax1.tick_params(axis='x', rotation=45)
sns.despine(ax=ax1)
st.pyplot(fig1)

# Wykres 2: liczba ofert na miasto z podziałem na typ nieruchomości
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
    df_valid = df.copy()
    df_valid = df_valid[
        df_valid['cena_za_m2'].notna()
        & (df_valid['cena_za_m2'] < 30000)
        & np.isfinite(df_valid['cena_za_m2'])
    ]

    if df_valid.empty:
        st.warning("⚠️ Brak danych do analizy.")
    else:
        st.subheader("📊 Kluczowe statystyki ceny za 1 m²")
        mean_val = round(df_valid['cena_za_m2'].mean(), 2)
        median_val = round(df_valid['cena_za_m2'].median(), 2)
        max_val = round(df_valid['cena_za_m2'].max(), 2)
        min_val = round(df_valid['cena_za_m2'].min(), 2)
        std_val = round(df_valid['cena_za_m2'].std(), 2)
        count_val = int(df_valid['cena_za_m2'].count())

        def render_card(title, value, unit):
            return f"""
            <div style="background-color: #fff8dc; padding: 20px; border-radius: 12px; text-align: center; box-shadow: 1px 1px 6px rgba(0,0,0,0.1); margin-bottom: 10px;">
                <div style="font-weight: bold; font-size: 16px;">{title}</div>
                <div style="font-size: 24px; font-weight: 600; margin: 5px 0;">{value}</div>
                <div style="font-size: 12px; color: #555;">{unit}</div>
            </div>
            """

        cols = st.columns(3)
        for col, val, name in zip(cols, [mean_val, median_val, max_val], ["Średnia", "Mediana", "Max"]):
            with col:
                st.markdown(render_card(name, val, "PLN/m²"), unsafe_allow_html=True)
        cols = st.columns(3)
        for col, val, name, unit in zip(cols, [min_val, std_val, count_val], ["Min", "Odch. standardowe", "Liczba ofert"], ["PLN/m²", "PLN/m²", "obserwacji"]):
            with col:
                st.markdown(render_card(name, val, unit), unsafe_allow_html=True)

        st.subheader("📈 Histogram ceny za 1 m²")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.histplot(df_valid['cena_za_m2'], bins=15, kde=True, color='skyblue', ax=ax2)
        ax2.set_title("Histogram ceny za 1 m²", fontsize=14)
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


st.header("📈 Interaktywny wykres zależności (Scatterplot)")

# Wybór kolumn do analizy (tylko numeryczne)
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

# Domyślne indeksy dla squareMeters i price
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




st.header("🗺️ Analiza przestrzenna dla wybranego miasta")

miasta = sorted(df['miasto'].dropna().unique())
default_city_index = miasta.index("Warszawa") if "Warszawa" in miasta else 0
selected_city = st.selectbox("Wybierz miasto", miasta, index=default_city_index)

df_city = df[df['miasto'] == selected_city].copy()

if df_city[['szerokosc_geo', 'dlugosc_geo']].isnull().any().any():
    st.warning("⚠️ Brakuje danych geolokalizacyjnych dla wybranego miasta.")
else:
    st.subheader("📍 Oferty nieruchomości – intensywność koloru = cena")
    lat_center = df_city['szerokosc_geo'].mean()
    lon_center = df_city['dlugosc_geo'].mean()
    m1 = folium.Map(location=[lat_center, lon_center], zoom_start=12)
    price_min = df_city['cena'].min()
    price_max = df_city['cena'].max()

    def get_color(price):
        norm_price = (price - price_min) / (price_max - price_min + 1e-5)
        return f"rgba(255, 0, 0, {norm_price:.2f})"

    for _, row in df_city.iterrows():
        folium.CircleMarker(
            location=[row['szerokosc_geo'], row['dlugosc_geo']],
            radius=5,
            color=get_color(row['cena']),
            fill=True,
            fill_opacity=0.7,
            popup=f"{row['cena']:,.0f} PLN"
        ).add_to(m1)

    st_folium(m1, width=900, height=500)




    # Mapa 2 – z punktami POI (szkoły, apteki itd.)
 
    st.subheader("🏫 Mapa z punktami usług publicznych")
    m2 = folium.Map(location=[lat_center, lon_center], zoom_start=13)
    offers_cluster = MarkerCluster(name="Oferty").add_to(m2)
    for _, row in df_city.iterrows():
        folium.Marker(
            location=[row['szerokosc_geo'], row['dlugosc_geo']],
            popup=f"Oferta: {row['cena']:,.0f} PLN",
            icon=folium.Icon(color="blue", icon="home", prefix="fa")
        ).add_to(offers_cluster)

    poi_categories = {
        "Apteki": "data/poi_apteki.csv",
        "Szkoły": "data/poi_szkoly.csv",
        "Restauracje": "data/poi_restauracje.csv",
        "Przedszkola": "data/poi_przedszkola.csv",
        "Urzędy": "data/poi_urzedy.csv",
        "Poczta": "data/poi_poczta.csv",
        "Szpitale i przychodnie": "data/poi_szpitale.csv",
        "Szkoły wyższe": "data/poi_uczelnie.csv"
    }

    for label, filepath in poi_categories.items():
        try:
            poi_df = pd.read_csv(filepath)
            poi_df_city = poi_df[poi_df['miasto'] == selected_city]
            poi_group = folium.FeatureGroup(name=label)
            for _, row in poi_df_city.iterrows():
                folium.Marker(
                    location=[row['szerokosc_geo'], row['dlugosc_geo']],
                    popup=row.get("name", label),
                    icon=folium.Icon(color="green", icon="plus", prefix="fa")
                ).add_to(poi_group)
            poi_group.add_to(m2)
        except Exception as e:
            st.info(f"ℹ️ Nie udało się wczytać danych dla: {label} ({filepath})")

    folium.LayerControl().add_to(m2)
    st_folium(m2, width=900, height=600)