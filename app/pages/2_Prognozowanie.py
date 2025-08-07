import os
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import osmnx as ox

DATA_PATH = Path("data/processed/apartments_cleaned.csv")
STREETS_PATH = Path("data/processed/ulice_w_miastach.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, encoding='utf-8')

@st.cache_data
def load_streets():
    return pd.read_csv(STREETS_PATH)

def app():
    st.title('Wybór parametrów nieruchomości')

    df = load_data()
    ulice_df = load_streets()

    miasta_dostepne = sorted(df['miasto'].dropna().unique())
    selected_city = st.selectbox('Wybierz miasto', miasta_dostepne)

    df_city = df[df['miasto'] == selected_city]

    ulice_city = ulice_df[ulice_df["miasto"] == selected_city]["ulica"].dropna().unique()
    if len(ulice_city) == 0:
        st.warning("Brak danych o ulicach w wybranym mieście.")
        return
    selected_street = st.selectbox('Wybierz ulicę', sorted(ulice_city))

    st.subheader('Parametry nieruchomości')

    area = st.slider('Powierzchnia (m²)', int(df_city['powierzchnia_m2'].min()), int(df_city['powierzchnia_m2'].max()))
    rooms = st.slider('Liczba pokoi', int(df_city['liczba_pokoi'].min()), int(df_city['liczba_pokoi'].max()))
    floor = st.slider('Piętro', int(df_city['pietro'].min()), int(df_city['pietro'].max()))

    min_year = int(df_city['rok_budowy'].dropna().min())
    max_year = int(df_city['rok_budowy'].dropna().max())
    build_year = st.selectbox('Rok budowy', list(range(min_year, max_year + 1)))

    balkon = st.radio('Czy posiada balkon', ('Tak', 'Nie'))
    balkon = 1 if balkon == 'Tak' else 0

    parking = st.radio('Miejsce parkingowe', ('Tak', 'Nie'))
    parking = 1 if parking == 'Tak' else 0

    winda = st.radio('Winda', ('Tak', 'Nie'))
    winda = 1 if winda == 'Tak' else 0

    ochrona = st.radio('Ochrona', ('Tak', 'Nie'))
    ochrona = 1 if ochrona == 'Tak' else 0

    komorka = st.radio('Komórka lokatorska', ('Tak', 'Nie'))
    komorka = 1 if komorka == 'Tak' else 0

    liczba_punktow_poi = st.slider('Liczba punktów POI w okolicy',
                                   int(df_city['liczba_punktow_poi'].min()),
                                   int(df_city['liczba_punktow_poi'].max()))

    st.subheader('Wybierz modele ML do prognozy')

    ml_models = [
        'Regresja liniowa',
        'Regresja wielomianowa',
        'Drzewa decyzyjne',
        'Lasy losowe',
        'K Najbliższych Sąsiadów (KNN)',
        'XGBoost'
    ]
    selected_models = st.multiselect('Modele ML', ml_models, default=ml_models)

    if st.button('Prognozuj cenę'):
        st.session_state['input_data'] = [
            selected_city,
            selected_street,
            area,
            rooms,
            floor,
            build_year,
            balkon,
            parking,
            winda,
            ochrona,
            komorka,
            liczba_punktow_poi,
            selected_models
        ]
        st.session_state['current_page'] = 'Wyniki'
        st.session_state['navigate_to_results'] = True
        st.experimental_rerun()

if __name__ == "__main__":
    app()
