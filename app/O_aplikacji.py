import streamlit as st

st.title("Interaktywna analiza rynku nieruchomości w Polsce z wykorzystaniem wizualizacji danych i algorytmów uczenia maszynowego ")
st.markdown("""
    ## Informacje ogólne

    Niniejsza aplikacja została przygotowana jako **projekt zaliczeniowy** w ramach  
    **studiów podyplomowych „Data Science w zastosowaniach biznesowych – praktyczne warsztaty”**  
    realizowanych na **Wydziale Nauk Ekonomicznych Uniwersytetu Warszawskiego**.
    """, unsafe_allow_html=True)

st.markdown("""
    ## Cel aplikacji

    Celem projektu było stworzenie **interaktywnego narzędzia analitycznego**, które umożliwia:
    - eksplorację rynku nieruchomości w największych miastach Polski,
    - analizę danych za pomocą wizualizacji EDA,
    - prognozowanie cen mieszkań z wykorzystaniem algorytmów uczenia maszynowego.

    Projekt wykorzystuje rzeczywiste dane ogłoszeniowe oraz zaawansowane techniki z zakresu analizy danych i machine learning.
    """, unsafe_allow_html=True)



st.markdown("""
    ## Struktura aplikacji

    Aplikacja składa się z trzech głównych sekcji:

    1. **O aplikacji** – obecna strona z informacjami ogólnymi o projekcie.  
    2. **Analiza EDA** – interaktywna analiza eksploracyjna zbioru danych nieruchomości.  
    3. **Prognozowanie** – moduł predykcyjny pozwalający użytkownikowi na estymację ceny mieszkania na podstawie wybranych parametrów.
    """, unsafe_allow_html=True)



st.markdown("""
    ## Wykorzystane modele ML

    Do predykcji cen mieszkań zaimplementowano następujące modele uczenia maszynowego:

    - Regresja liniowa  
    - Regresja wielomianowa  
    - Drzewa decyzyjne  
    - Lasy losowe (Random Forest)  
    - K Najbliższych Sąsiadów (KNN)  
    - XGBoost  

    Użytkownik może porównać wyniki i błędy predykcji dla różnych modeli oraz samodzielnie wskazać preferowany algorytm.
    """, unsafe_allow_html=True)

st.markdown("""
    ## 📂 Repozytorium projektu

    Pełen kod źródłowy aplikacji dostępny jest publicznie na GitHub:  
    [https://github.com/Polish-ML-enthusiast/DataScience_projekt](https://github.com/Polish-ML-enthusiast/DataScience_projekt)
    """, unsafe_allow_html=True)