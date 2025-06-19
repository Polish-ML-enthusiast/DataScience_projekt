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
from geopy.distance import geodesic
from pathlib import Path
import plotly.graph_objects as go
from io import BytesIO
import branca.colormap as cm
import requests


# Ścieżki do danych
DATA_PATH = Path("data/processed/apartments_cleaned.csv")
MAP_PATH = Path("data/maps/ne_10m_admin_0_countries.shp")
REGIONS_PATH = Path("data/maps/wojewodztwa.shp")
ULICE_PATH = Path("data/processed/ulice_w_miastach.csv")

# Buforowanie danych
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_ulice():
    return pd.read_csv(ULICE_PATH)

df = load_data()
ulice_df = load_ulice()
ulice_df = ulice_df[ulice_df["miasto"].isin(df["miasto"].unique())]  # spójność miast

st.title("📊 Eksploracyjna Analiza Danych (EDA)")

# 1. Podstawowe informacje
st.header("Podstawowe informacje o danych")

stats = {
    "Liczba ofert": f"{len(df):,}".replace(",", " "),
    "Liczba miast": f"{df['miasto'].nunique():,}".replace(",", " "),
    "Najczęstszy typ": df['typ_nieruchomosci'].mode()[0],
    "Średnia cena za m²": f"{df['cena_za_m2'].mean():,.2f} zł".replace(",", " "),
    "Mediana ceny za m²": f"{df['cena_za_m2'].median():,.2f} zł".replace(",", " "),
    "Min cena za m²": f"{df['cena_za_m2'].min():,.2f} zł".replace(",", " "),
    "Max cena za m²": f"{df['cena_za_m2'].max():,.2f} zł".replace(",", " "),
    "Zakres pow.": f"{df['powierzchnia_m2'].min():,.2f}–{df['powierzchnia_m2'].max():,.2f} m²".replace(",", " ")
}

box_color = "#e9f0f5"
box_style = """
    background-color: {color};
    padding: 16px;
    border-radius: 12px;
    text-align: center;
    height: 120px;
    width: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    box-sizing: border-box;
"""

# Rozbijanie na wiersze po 4 elementy
stat_items = list(stats.items())
num_per_row = 4
rows = [stat_items[i:i + num_per_row] for i in range(0, len(stat_items), num_per_row)]

for row in rows:
    cols = st.columns(num_per_row)
    for col, (label, value) in zip(cols, row):
        with col:
            st.markdown(f"""
                <div style='{box_style.format(color=box_color)}'>
                    <div style='font-size:14px; font-weight:600; margin-bottom:8px;'>{label}</div>
                    <div style='font-size:20px; font-weight:bold'>{value}</div>
                </div>
            """, unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)



# 2. Liczba ofert według miasta
# 2. Liczba ofert według miasta – interaktywna wersja Plotly

import plotly.express as px

st.header("Liczba ofert według miasta")

# Czyszczenie danych wejściowych
df_clean = df.copy()
df_clean['miasto'] = df_clean['miasto'].astype(str).str.strip()
df_clean = df_clean[df_clean['miasto'].notna() & (df_clean['miasto'] != '')]

# Wykres 1: liczba ofert per miasto
df_city_counts = (
    df_clean
    .groupby('miasto')
    .size()
    .reset_index(name='liczba_ofert')
    .sort_values('miasto')
)

fig_city = px.bar(
    df_city_counts,
    x="miasto",
    y="liczba_ofert",
    text="liczba_ofert",
    color_discrete_sequence=["#014D65"]
)

fig_city.update_traces(
    textposition="outside",
    marker_line_width=1,
    marker_line_color="black",
    hovertemplate="<b>%{x}</b><br>Liczba ofert: %{y}<extra></extra>"
)
fig_city.update_layout(
    xaxis_title="Miasto",
    yaxis_title="Liczba ofert",
    xaxis_tickangle=45,
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(size=12),
    margin=dict(t=50, l=40, r=40, b=80)
)
st.plotly_chart(fig_city, use_container_width=True)


# 4. Mapa lokalizacji miast

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

    ax3.set_title("(wielkość punktu = liczba ofert)")
    ax3.axis('off')
    st.pyplot(fig3)
except Exception as e:
    st.warning("Nie udało się wygenerować mapy.")
    st.code(str(e))


# Wykres 2: liczba ofert per miasto i typ nieruchomości



st.subheader("Liczba ofert według miasta i rodzaju nieruchomości ")

df_grouped = (
    df
    .groupby(["miasto", "typ_nieruchomosci"])
    .size()
    .reset_index(name="liczba_ofert")
    .sort_values("miasto")
)

custom_palette = ["#bedbea", "#779cae", "#5688a0", "#37718f", "#165978"]

fig_type = px.bar(
    df_grouped,
    x="miasto",
    y="liczba_ofert",
    color="typ_nieruchomosci",
    barmode="group",
    category_orders={"miasto": sorted(df_grouped["miasto"].unique())},
    color_discrete_sequence=custom_palette
)

fig_type.update_traces(
    text=None,
    hovertemplate="<b>Miasto:</b> %{x}<br><b>Typ:</b> %{customdata[0]}<br><b>Liczba ofert:</b> %{y}<extra></extra>",
    customdata=np.stack([df_grouped['typ_nieruchomosci']], axis=-1)
)

fig_type.update_layout(
    xaxis_title="Miasto",
    yaxis_title="Liczba ofert",
    xaxis_tickangle=45,
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(size=12),
    legend_title_text="Typ nieruchomości",
    legend=dict(orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5),
    margin=dict(t=60, l=40, r=40, b=100)
)
st.plotly_chart(fig_type, use_container_width=True)


# 7. Mapa nieruchomości i wybranej ulicy


st.header("🗌️ Mapa nieruchomości i wybranej ulicy")

if "ulica" not in df.columns or df["ulica"].isnull().all():
    st.warning("⚠️ Kolumna 'ulica' jest pusta lub nie istnieje. Sprawdź preprocessing.")
else:
    miasto = st.selectbox("Wybierz miasto:", df["miasto"].unique())
    ulice_miasta = ulice_df[ulice_df["miasto"] == miasto]["ulica"].dropna().unique().tolist()
    if not ulice_miasta:
        st.warning("⚠️ Brak ulic w wybranym mieście w pliku.")
        ulica = None
    else:
        ulica = st.selectbox("Wybierz ulicę:", sorted(ulice_miasta))

    nieruchomosci = df[df["miasto"] == miasto]
    center_lat = nieruchomosci["szerokosc_geo"].mean()
    center_lon = nieruchomosci["dlugosc_geo"].mean()

    if ulica:
        nieruchomosci_ulica = nieruchomosci[nieruchomosci["ulica"].str.contains(ulica, case=False, na=False)]
        if not nieruchomosci_ulica.empty:
            center_lat = nieruchomosci_ulica["szerokosc_geo"].mean()
            center_lon = nieruchomosci_ulica["dlugosc_geo"].mean()

    promien = st.slider("Wybierz promień (km):", 0.5, 10.0, 2.0, step=0.5)
    mapa = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")

    folium.Circle(
        location=[center_lat, center_lon],
        radius=promien * 1000,
        color='blue',
        fill=True,
        fill_opacity=0.15,
    ).add_to(mapa)

    # Skala kolorów wg ceny
    min_price = nieruchomosci["cena"].min()
    max_price = nieruchomosci["cena"].max()
    colormap = cm.linear.Reds_09.scale(min_price, max_price)
    colormap.caption = 'Cena nieruchomości'
    mapa.add_child(colormap)

    for _, row in nieruchomosci.iterrows():
        if pd.notna(row["szerokosc_geo"]) and pd.notna(row["dlugosc_geo"]):
            folium.CircleMarker(
                location=[row["szerokosc_geo"], row["dlugosc_geo"]],
                radius=6,
                color=colormap(row["cena"]),
                fill=True,
                fill_opacity=0.8,
                weight=1
            ).add_to(mapa)

    # Filtrowanie POI
    st.subheader("🔍 Wybierz rodzaje punktów POI do wyświetlenia")
    POI_TYPES = {
        "school": ("graduation-cap", "Szkoła"),
        "hospital": ("hospital", "Szpital"),
        "restaurant": ("cutlery", "Restauracja"),
        "pharmacy": ("plus-square", "Apteka"),
        "kindergarten": ("child", "Przedszkole"),
        "bank": ("university", "Placówka bankowa"),
        "fuel": ("tint", "Stacja paliw"),
        "supermarket": ("shopping-cart", "Supermarket"),
        "park": ("tree", "Park")
    }

    selected_poi_types = st.multiselect("Rodzaje POI:", options=list(POI_TYPES.keys()), default=list(POI_TYPES.keys()), format_func=lambda x: POI_TYPES[x][1])

    def get_poi_osm(lat, lon, radius_m=1000):
        overpass_url = "http://overpass-api.de/api/interpreter"
        selected = "|".join(selected_poi_types)
        query = f"""
        [out:json];
        (
          node["amenity"~"{selected}"](around:{radius_m},{lat},{lon});
          node["shop"="supermarket"](around:{radius_m},{lat},{lon});
          node["leisure"="park"](around:{radius_m},{lat},{lon});
        );
        out center;
        """
        response = requests.get(overpass_url, params={'data': query})
        data = response.json()
        return data.get("elements", [])

    poi_data = get_poi_osm(center_lat, center_lon, radius_m=promien * 1000)

    legend_html = '<div style="position: fixed; bottom: 50px; left: 50px; width: 200px; height: auto; background-color: white; z-index:9999; font-size:14px; border:1px solid grey; border-radius:8px; padding: 10px">'
    legend_html += '<b>Legenda POI</b><br>'
    for k in selected_poi_types:
        icon, label = POI_TYPES[k]
        legend_html += f'<i class="fa fa-{icon}"></i> {label}<br>'
    legend_html += '</div>'
    mapa.get_root().html.add_child(folium.Element(legend_html))

    for elem in poi_data:
        lat_poi = elem.get("lat")
        lon_poi = elem.get("lon")
        tags = elem.get("tags", {})
        amenity = tags.get("amenity") or tags.get("shop") or tags.get("leisure")
        icon_data = POI_TYPES.get(amenity)
        if lat_poi and lon_poi and icon_data and amenity in selected_poi_types:
            icon, label = icon_data
            folium.Marker(
                location=[lat_poi, lon_poi],
                icon=folium.Icon(icon=icon, prefix="fa", color="gray")
            ).add_to(mapa)

    st_folium(mapa, width=900, height=600)


#boxploty 

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode
import numpy as np

st.header("Wykresy pudełkowe ceny za 1m2")

# Przygotowanie danych
df_plot = df.copy()
df_plot = df_plot[df_plot["cena_za_m2"].notna() & (df_plot["cena_za_m2"] < 30000)]
df_plot["miasto"] = df_plot["miasto"].astype(str).str.strip().str.title()

# Obliczamy modę ceny dla każdego miasta
moda_dict = (
    df_plot.groupby("miasto")["cena_za_m2"]
    .agg(lambda x: mode(x, keepdims=True)[0][0])  # najczęściej występująca wartość
    .to_dict()
)

# Normalizacja mody (0–1)
moda_values = np.array(list(moda_dict.values()))
moda_scaled = (moda_values - moda_values.min()) / (moda_values.max() - moda_values.min())

# Generujemy paletę z jednego koloru (np. niebieski)
base_color = "blue"
palette = sns.light_palette(base_color, n_colors=len(moda_scaled), reverse=True)

# Mapujemy miasta na kolory według posortowanej listy
sorted_cities = sorted(moda_dict.keys())
city_color_map = dict(zip(sorted_cities, palette))

# Rysujemy boxplot
fig, ax = plt.subplots(figsize=(14, 8))

sns.boxplot(
    data=df_plot,
    x="miasto",
    y="cena_za_m2",
    order=sorted_cities,
    palette=city_color_map,
    showmeans=True,
    meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"},
    boxprops=dict(alpha=0.9),
    ax=ax
)

ax.set_title("Boxplot ceny za m² według miasta (intensywność koloru = najczęstsza cena)")
ax.set_xlabel("Miasto")
ax.set_ylabel("Cena za 1 m² (PLN)")
ax.tick_params(axis='x', rotation=45)
sns.despine(ax=ax)
st.pyplot(fig)


# 3. Histogram ceny za 1 m²

st.subheader("Histogram ceny za 1m²")
miasta_dostepne = sorted(df['miasto'].dropna().unique())
wybrane_miasta = st.multiselect("Miasta:", options=miasta_dostepne, default=miasta_dostepne[:1])

if not wybrane_miasta:
            st.warning("⚠️ Wybierz co najmniej jedno miasto.")
else:
    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("Set2", n_colors=len(wybrane_miasta))

    for miasto, color in zip(wybrane_miasta, colors):
        subset = df[df['miasto'] == miasto]
        sns.histplot(
            subset['cena_za_m2'],
            bins=20,
            kde=True,
            label=miasto,
            color=color,
            alpha=0.5,
            ax=ax_hist
        )

    ax_hist.set_title("Histogram ceny za 1 m² dla wybranych miast")
    ax_hist.set_xlabel("Cena za 1 m² (PLN)")
    ax_hist.set_ylabel("Liczba ogłoszeń")
    ax_hist.legend(title="Miasto")
    ax_hist.grid(False)
    sns.despine(ax=ax_hist)
    st.pyplot(fig_hist)





# 5. Macierz korelacji

st.header("🔍 Macierz korelacji między zmiennymi")

numeric_df = df.select_dtypes(include=["float64", "int64"]).copy()
if numeric_df.isnull().any().any():
    numeric_df = numeric_df.fillna(0)

corr_matrix = numeric_df.corr().round(2)

fig_heatmap = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.index,
    colorscale="Inferno",
    zmin=-1,
    zmax=1,
    hoverongaps=False,
    colorbar=dict(
        title="Korelacja",
        thickness=15,
        len=1.0,
        lenmode="fraction",
        x=1.02,
        xpad=0,
        xanchor="left"
    )
))

n_vars = len(corr_matrix.columns)
cell_size = 40
size_px = n_vars * cell_size

fig_heatmap.update_layout(
    title="Macierz korelacji między zmiennymi",
    font=dict(size=12),
    plot_bgcolor="white",
    paper_bgcolor="white",
    width=size_px,
    height=size_px,
    margin=dict(t=80, l=80, r=80, b=80),
    xaxis=dict(scaleanchor="y", ticks="", showgrid=False),
    yaxis=dict(autorange="reversed", ticks="", showgrid=False)
)

st.plotly_chart(fig_heatmap, use_container_width=False)

# Zapis do pliku PNG
save_btn = st.button("💾 Zapisz wykres do pliku PNG")
if save_btn:
    buf = BytesIO()
    fig_heatmap.write_image(buf, format="png")
    st.download_button(
        label="📥 Pobierz jako PNG",
        data=buf.getvalue(),
        file_name="macierz_korelacji.png",
        mime="image/png"
    )






# 6. Interaktywny wykres zależności
st.header("📈 Interaktywny wykres zależności pomiędzy wybranymi zmiennymi")
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

