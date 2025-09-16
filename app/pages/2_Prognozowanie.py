# ======================================================================
# 2_Prognozowanie.py  (czytelne etykiety — bez interakcji/wielomianów, bez "Ulica (hash)" na wykresie)
# ======================================================================

from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.feature_extraction import FeatureHasher

from sklearn.linear_model import RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

# === XGBoost: pokaż przyczynę braku dostępności zamiast "ciszy"
XGB_AVAILABLE = False
XGB_IMPORT_REASON = None
XGB_VERSION = None
try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
    XGB_VERSION = getattr(xgb, "__version__", "unknown")
except Exception as e:
    XGB_AVAILABLE = False
    XGB_IMPORT_REASON = str(e)

# ----------------- Ścieżki -----------------

DATA_CLEAN_PATH = Path("data/processed/apartments_cleaned.csv")
TRAIN_PATH = Path("data/processed/apartments_train.csv")
TEST_PATH  = Path("data/processed/apartments_test.csv")
ULICE_PATH = Path("data/processed/ulice_w_miastach.csv")

st.set_page_config(page_title="Predykcja cen – Prognozowanie", layout="wide")

# ===================== Helpers =====================

@st.cache_data
def load_ulice():
    if ULICE_PATH.exists():
        return pd.read_csv(ULICE_PATH)
    return pd.DataFrame(columns=["miasto", "ulica"])

@st.cache_data
def load_train_test_or_clean():
    """Użyj gotowych TRAIN/TEST; w przeciwnym razie podziel apartments_cleaned.csv (stratyfikacja po 'miasto')."""
    if TRAIN_PATH.exists() and TEST_PATH.exists():
        return pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH), True
    df = pd.read_csv(DATA_CLEAN_PATH)
    strat = df['miasto'].fillna("Unknown") if 'miasto' in df.columns else None
    tr, te = train_test_split(df, test_size=0.2, random_state=42, shuffle=True,
                              stratify=strat if strat is not None else None)
    return tr.reset_index(drop=True), te.reset_index(drop=True), False

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

def compute_metrics(y_true, y_pred):
    return {
        "MAPE [%]": mape(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "R²": float(r2_score(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
        "N_test": int(len(y_true))
    }

def make_ohe():
    """OneHotEncoder zgodny z wersją scikit-learna (sparse_output vs sparse)."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# --- ulica -> lista słowników (dla FeatureHasher) ---
def street_to_dict(X):
    if isinstance(X, pd.DataFrame):
        col = X.columns[0] if X.shape[1] >= 1 else 0
        vals = X[col].astype(str).fillna("").tolist()
    elif isinstance(X, pd.Series):
        vals = X.astype(str).fillna("").tolist()
    else:
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 1:
            vals = ["" if v is None else str(v) for v in arr.tolist()]
        else:
            vals = ["" if v is None else str(v) for v in arr[:, 0].tolist()]
    return [{"ulica": v} for v in vals]

def build_street_hasher(n_features=64):
    to_dict = FunctionTransformer(street_to_dict, validate=False)
    to_dense = FunctionTransformer(
        func=lambda A: A.toarray() if hasattr(A, "toarray") else np.asarray(A),
        accept_sparse=True, validate=False
    )
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("to_dict", to_dict),
        ("hasher", FeatureHasher(n_features=n_features, input_type="dict")),
        ("to_dense", to_dense),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])

# ===== preprocessor z rozdzieleniem poly tylko dla cech ciągłych =====

def build_preprocessor(
    numeric_poly_cols,      # tylko ciągłe: pow., pokoje, piętro, POI
    numeric_plain_cols,     # binarne / bez poly
    use_poly=False,
    poly_degree=2,
    use_city_ohe=False,
    use_street_hash=False,
    street_hash_dim=64
):
    transformers = []

    if use_poly and len(numeric_poly_cols) > 0:
        num_poly_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("poly",    PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ("scaler",  StandardScaler(with_mean=True, with_std=True))
        ]
        num_poly_tf = Pipeline(steps=num_poly_steps)
        transformers.append(("num_poly", num_poly_tf, numeric_poly_cols))

    if len(numeric_plain_cols) > 0:
        num_plain_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler(with_mean=True, with_std=True))
        ]
        num_plain_tf = Pipeline(steps=num_plain_steps)
        transformers.append(("num_plain", num_plain_tf, numeric_plain_cols))

    if use_city_ohe:
        city_tf = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", make_ohe()),
        ])
        transformers.append(("city", city_tf, ["miasto"]))

    if use_street_hash:
        transformers.append(("street", build_street_hasher(street_hash_dim), ["ulica"]))

    return ColumnTransformer(transformers=transformers, remainder="drop")

# ===================== Konfiguracja jakości =====================

USE_LOG_TARGET = True          # log1p/expm1 dla WSZYSTKICH modeli
RIDGE_ALPHAS = np.logspace(-3, 3, 13)  # dla RidgeCV

def wrap_with_ttr(est):
    """Opcjonalne owinięcie estymatora log-transformacją celu."""
    if not USE_LOG_TARGET:
        return est
    return TransformedTargetRegressor(
        regressor=est,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False
    )

# === MODELE z poprawionymi defaultami + log-target wrapper
def get_models_dict(rf_trees=400, knn_k=7, xgb_estimators=600):
    models = {
        # Liniowa -> RidgeCV (stabilna, lepsza generalizacja)
        "Regresja liniowa": wrap_with_ttr(
            RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True, cv=None)
        ),
        # Drzewo -> ograniczenie głębokości, liść + lekkie ccp_alpha
        "Drzewa decyzyjne": wrap_with_ttr(
            DecisionTreeRegressor(
                random_state=42, max_depth=14, min_samples_leaf=5, ccp_alpha=1e-4
            )
        ),
        # Las -> stabilniejsze parametry + oob_score
        "Lasy losowe": wrap_with_ttr(
            RandomForestRegressor(
                n_estimators=rf_trees, random_state=42, n_jobs=-1,
                max_depth=None, min_samples_leaf=2, max_features="sqrt",
                bootstrap=True, oob_score=True
            )
        ),
        # KNN -> „distance” zwykle wygrywa na rozproszonych cechach
        "K Najbliższych Sąsiadów (KNN)": wrap_with_ttr(
            KNeighborsRegressor(n_neighbors=knn_k, weights="distance")
        ),
        # Wielomianowa -> PolynomialFeatures w preprocesorze + RidgeCV jako regresor
        "Regresja wielomianowa": wrap_with_ttr(
            RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True, cv=None)
        ),
    }

    if XGB_AVAILABLE:
        xgb_core = XGBRegressor(
            n_estimators=xgb_estimators,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.2,
            reg_alpha=0.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="rmse"
        )
        # Bez early stopping / callbacks — maksymalna kompatybilność
        models["XGBoost"] = wrap_with_ttr(xgb_core)

    return models

# ---------- mapowanie czytelnych nazw bazowych ----------
HUMAN_MAP = {
    "powierzchnia_m2": "Powierzchnia (m²)",
    "liczba_pokoi": "Liczba pokoi",
    "pietro": "Piętro",
    "liczba_punktow_poi": "Liczba punktów POI",
    "balkon": "Balkon",
    "miejsce_parkingowe": "Miejsce parkingowe",
    "winda": "Winda",
    "ochrona": "Ochrona",
    "komorka_lokatorska": "Komórka lokatorska",
}

def _tokenize_poly_expr(name: str):
    parts = re.split(r"\s+", name.strip())
    bases = []
    for p in parts:
        base = p.split("^", 1)[0]
        if base:
            bases.append(base)
    return bases

def aggregate_importances_to_base(raw_names, importances):
    agg = {}
    for n, v in zip(raw_names, importances):
        if n.startswith("ulica_hash_"):
            continue
        v = float(abs(v))
        if n.startswith("miasto_"):
            agg["Miasto"] = agg.get("Miasto", 0.0) + v
            continue
        bases = _tokenize_poly_expr(n)
        if not bases:
            key = n.replace("_", " ").title()
            agg[key] = agg.get(key, 0.0) + v
            continue
        share = v / len(bases)
        for b in bases:
            key = HUMAN_MAP.get(b, b.replace("_", " ").title())
            agg[key] = agg.get(key, 0.0) + share
    df = pd.DataFrame({"feature": list(agg.keys()), "importance": list(agg.values())})
    return df

def build_dataframe_for_plot(y_true, y_pred):
    return pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "residual": y_true - y_pred})

def style_pl(df: pd.DataFrame):
    return df.style.format(precision=2, thousands=" ", decimal=",")

# ===================== część widoczna dla użytkownika =====================

st.title("Prognoza ceny nieruchomości")

# — ukryte ustawienia
TEST_PLOT_SAMPLE = 2000
STREET_HASH_DIM = 32
RF_TREES = 400
KNN_K = 7
XGB_EST = 600
USE_PERM_IMPORTANCE = False
POLY_DEGREE = 2

USE_CITY_OHE = True
USE_STREET_HASH = True   # model nadal korzysta z ulicy (hash), ale nie pokażemy jej na wykresie

# ===================== Wczytanie danych =====================

ulice_df = load_ulice()
train_df, test_df, _ = load_train_test_or_clean()

required_base = {"miasto", "cena", "powierzchnia_m2", "liczba_pokoi", "pietro"}
missing = required_base - set(train_df.columns)
if missing:
    st.error(f"Brakuje kolumn w TRAIN: {missing}.")
    st.stop()

# Informacja o XGB w UI
with st.expander("Stan środowiska ML", expanded=False):
    if XGB_AVAILABLE:
        st.info(f"✅ XGBoost dostępny (wersja: {XGB_VERSION})")
    else:
        st.warning(f"⚠️ XGBoost niedostępny. Powód importu: {XGB_IMPORT_REASON}")

# --- Miasto & Ulica ---
miasta_dostepne = sorted(train_df['miasto'].dropna().unique())
selected_city = st.selectbox("Wybierz miasto", miasta_dostepne)

ulice_city = []
if not ulice_df.empty and {"miasto","ulica"}.issubset(ulice_df.columns):
    ulice_city = sorted(ulice_df[ulice_df["miasto"] == selected_city]["ulica"].dropna().unique().tolist())
if not ulice_city and "ulica" in train_df.columns:
    ulice_city = sorted(train_df.loc[train_df["miasto"] == selected_city, "ulica"].dropna().unique().tolist())

selected_street = st.selectbox("Wybierz ulicę", ulice_city) if ulice_city else ""

# --- Parametry ---
df_city = train_df[train_df["miasto"] == selected_city]
def _int_range_safe(s, default_min, default_max):
    if s is None or s.dropna().empty:
        return default_min, default_max
    return int(np.floor(s.min())), int(np.ceil(s.max()))

area_min, area_max   = _int_range_safe(df_city.get("powierzchnia_m2"), 20, 120)
rooms_min, rooms_max = _int_range_safe(df_city.get("liczba_pokoi"), 1, 5)
floor_min, floor_max = _int_range_safe(df_city.get("pietro"), 0, 10)
poi_min, poi_max     = _int_range_safe(df_city.get("liczba_punktow_poi"), 0, 200)

st.subheader("Parametry nieruchomości")
area   = st.slider("Powierzchnia (m²)", area_min, area_max, min(50, area_max))
rooms  = st.slider("Liczba pokoi", rooms_min, rooms_max, min(max(rooms_min, 2), rooms_max))
floor  = st.slider("Piętro", floor_min, floor_max, min(max(floor_min, 2), floor_max))

balkon  = 1 if st.radio("Balkon", ("Tak", "Nie")) == "Tak" else 0
parking = 1 if st.radio("Miejsce parkingowe", ("Tak", "Nie")) == "Tak" else 0
winda   = 1 if st.radio("Winda", ("Tak", "Nie")) == "Tak" else 0
ochrona = 1 if st.radio("Ochrona", ("Tak", "Nie")) == "Tak" else 0
komorka = 1 if st.radio("Komórka lokatorska", ("Tak", "Nie")) == "Tak" else 0

liczba_punktow_poi = st.slider("Liczba punktów POI", poi_min, poi_max, min(max(poi_min, 10), poi_max))

# --- Modele ---
st.subheader("Wybierz modele ML")
ml_models_all = ["Regresja liniowa", "Regresja wielomianowa", "Drzewa decyzyjne", "Lasy losowe", "K Najbliższych Sąsiadów (KNN)"]
if XGB_AVAILABLE:
    ml_models_all.append("XGBoost")
selected_models = st.multiselect("Modele ML", ml_models_all, default=ml_models_all)

# ===================== Dane/cechy =====================

target_col = "cena"
available_cols = set(train_df.columns)

numeric_cols_all = [c for c in [
    "powierzchnia_m2", "liczba_pokoi", "pietro", "liczba_punktow_poi",
    "balkon", "miejsce_parkingowe", "winda", "ochrona", "komorka_lokatorska"
] if c in available_cols]

CONTINUOUS_CAND = ["powierzchnia_m2", "liczba_pokoi", "pietro", "liczba_punktow_poi"]
BINARY_CAND     = ["balkon", "miejsce_parkingowe", "winda", "ochrona", "komorka_lokatorska"]

numeric_poly_base  = [c for c in CONTINUOUS_CAND if c in available_cols]
numeric_plain_base = [c for c in BINARY_CAND if c in available_cols]

feature_cols = numeric_cols_all.copy()
if USE_CITY_OHE:
    feature_cols.append("miasto")
if USE_STREET_HASH and "ulica" in available_cols:
    feature_cols.append("ulica")

needed = set(feature_cols + [target_col, "miasto"])
if "ulica" in available_cols:
    needed.add("ulica")
missing2 = needed - available_cols
if missing2:
    st.error(f"Brak kolumn: {missing2}.")
    st.stop()

train_df_model = train_df.dropna(subset=[target_col]).copy()
test_df_model  = test_df.dropna(subset=[target_col]).copy()

X_train = train_df_model[feature_cols].copy()
y_train = train_df_model[target_col].copy()
X_test  = test_df_model[feature_cols].copy()
y_test  = test_df_model[target_col].copy()

# Próbka testowa do wykresów (szybciej)
plot_idx = np.arange(len(X_test))
if len(X_test) > TEST_PLOT_SAMPLE:
    rng = np.random.default_rng(42)
    plot_idx = rng.choice(len(X_test), size=TEST_PLOT_SAMPLE, replace=False)
X_test_plot = X_test.iloc[plot_idx]
y_test_plot = y_test.iloc[plot_idx]

# ===================== Przycisk =====================

if st.button("Prognozuj cenę"):
    user_data = {
        "powierzchnia_m2": area,
        "liczba_pokoi": rooms,
        "pietro": floor,
        "balkon": balkon,
        "miejsce_parkingowe": parking,
        "winda": winda,
        "ochrona": ochrona,
        "komorka_lokatorska": komorka,
        "liczba_punktow_poi": liczba_punktow_poi,
    }
    if USE_CITY_OHE:
        user_data["miasto"] = selected_city
    if USE_STREET_HASH and "ulica" in available_cols:
        user_data["ulica"] = (selected_street or "")

    user_row = pd.DataFrame([user_data])[feature_cols]

    tabs = st.tabs(["📈 Podsumowanie modeli", "🎯 Predykcja vs. rzeczywista", "🧠 Ważność cech", "📉 Reszty"])
    results_rows, predictions_rows = [], []

    models = get_models_dict(rf_trees=RF_TREES, knn_k=KNN_K, xgb_estimators=XGB_EST)

    for name in selected_models:
        if name == "XGBoost" and not XGB_AVAILABLE:
            st.warning("XGBoost nie jest dostępny – pomijam.")
            continue

        use_poly = (name == "Regresja wielomianowa")

        if use_poly:
            numeric_poly_cols  = numeric_poly_base
            numeric_plain_cols = numeric_plain_base
        else:
            numeric_poly_cols  = []
            numeric_plain_cols = numeric_cols_all

        pre = build_preprocessor(
            numeric_poly_cols=numeric_poly_cols,
            numeric_plain_cols=numeric_plain_cols,
            use_poly=use_poly, poly_degree=POLY_DEGREE,
            use_city_ohe=USE_CITY_OHE,
            use_street_hash=USE_STREET_HASH and ("ulica" in feature_cols),
            street_hash_dim=STREET_HASH_DIM
        )

        est = models[name]

        # === UCZENIE (bez ES — maksymalna zgodność xgboost) ===
        with st.spinner(f"Uczę model: {name}…"):
            pipe = Pipeline(steps=[("pre", pre), ("model", est)])
            pipe.fit(X_train, y_train)

        # Ocena (pełny TEST – metryki)
        y_pred_test = pipe.predict(X_test)
        y_pred_test = np.maximum(y_pred_test, 0)  # defensywnie: ceny nieujemne
        metrics = compute_metrics(y_test, y_pred_test)
        results_rows.append({"Model": name, **metrics})

        # Prognoza dla użytkownika
        user_pred = float(pipe.predict(user_row)[0])
        user_pred = max(user_pred, 0.0)
        predictions_rows.append({"Model": name, "Prognoza (PLN)": user_pred})

        # Wspólne predykcje na próbce (wykorzystamy 2x)
        y_pred_plot = pipe.predict(X_test_plot)

        # === WIZ: Predykcja vs Rzeczywista (na próbie) ===
        with tabs[1]:
            import plotly.express as px
            dfp = pd.DataFrame({"y_true": y_test_plot, "y_pred": y_pred_plot})
            fig_sc = px.scatter(
                dfp, x="y_true", y="y_pred",
                title=f"Predykcja vs Rzeczywista – {name} (próbka {len(dfp)})",
                labels={"y_true": "Cena rzeczywista (PLN)", "y_pred": "Cena przewidziana (PLN)"},
                opacity=0.6
            )
            if len(dfp) > 0:
                a = float(min(dfp["y_true"].min(), dfp["y_pred"].min()))
                b = float(max(dfp["y_true"].max(), dfp["y_pred"].max()))
                fig_sc.add_shape(type="line", x0=a, y0=a, x1=b, y1=b)
            fig_sc.update_layout(height=500)
            st.plotly_chart(fig_sc, use_container_width=True)

        # === WIZ: Ważność cech — tylko etykiety bazowe, bez Ulica (hash) ===
        with tabs[2]:
            import plotly.express as px

            # Nazwy cech po preprocesorze (lub rekonstrukcja)
            try:
                raw_names = pipe.named_steps["pre"].get_feature_names_out()
                raw_names = [n.replace("num__", "").replace("num_poly__", "").replace("num_plain__", "")
                               .replace("city__", "").replace("street__", "") for n in raw_names]
            except Exception:
                raw_names = []
                pre_fitted = pipe.named_steps["pre"]
                if use_poly and "num_poly" in pre_fitted.named_transformers_ and len(numeric_poly_cols) > 0:
                    try:
                        poly = pre_fitted.named_transformers_["num_poly"].named_steps["poly"]
                        raw_names += list(poly.get_feature_names_out(numeric_poly_cols))
                    except Exception:
                        raw_names += list(numeric_poly_cols)
                if "num_plain" in pre_fitted.named_transformers_ and len(numeric_plain_cols) > 0:
                    raw_names += list(numeric_plain_cols)
                if USE_CITY_OHE and "city" in pre_fitted.named_transformers_:
                    try:
                        ohe = pre_fitted.named_transformers_["city"].named_steps["ohe"]
                        raw_names += list(ohe.get_feature_names_out(["miasto"]))
                    except Exception:
                        raw_names += ["miasto"]
                if USE_STREET_HASH and "street" in pre_fitted.named_transformers_:
                    raw_names += [f"ulica_hash_{i}" for i in range(STREET_HASH_DIM)]

            # Rozpakuj model z ewentualnego TTR
            model_body = pipe.named_steps["model"]
            model_core = (model_body.regressor_ if isinstance(model_body, TransformedTargetRegressor) and hasattr(model_body, "regressor_")
                          else (model_body.regressor if isinstance(model_body, TransformedTargetRegressor) else model_body))

            imp_df = None
            try:
                if hasattr(model_core, "feature_importances_"):
                    importances = np.asarray(model_core.feature_importances_)
                    imp_df = aggregate_importances_to_base(raw_names, importances)
                elif hasattr(model_core, "coef_"):
                    coef = np.ravel(model_core.coef_)
                    imp_df = aggregate_importances_to_base(raw_names, coef)
                elif USE_PERM_IMPORTANCE:
                    Xp = pipe.named_steps["pre"].transform(X_test_plot)
                    r = permutation_importance(model_core, Xp, y_test_plot, n_repeats=5, random_state=42, n_jobs=-1)
                    imp_df = aggregate_importances_to_base(raw_names, r.importances_mean)
            except Exception:
                imp_df = None

            if imp_df is not None and not imp_df.empty:
                top_k = imp_df.sort_values("importance", ascending=False).head(20)
                fig_imp = px.bar(
                    top_k.sort_values("importance", ascending=True),
                    x="importance", y="feature", orientation="h",
                    title=f"Ważność cech – top 20 – {name}",
                    labels={"feature": "Cecha", "importance": "Ważność (zagregowana)"}
                )
                fig_imp.update_layout(height=600, yaxis=dict(tickfont=dict(size=11)))
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info(f"Brak (lub wyłączona) ważność cech dla modelu {name} w tym trybie.")

        # === WIZ: Reszty (na próbie) ===
        with tabs[3]:
            import plotly.express as px
            dfp_res = pd.DataFrame({"residual": y_test_plot - y_pred_plot})
            fig_hist = px.histogram(dfp_res, x="residual", nbins=40, title=f"Histogram reszt – {name} (próbka)",
                                    labels={"residual": "Reszta (y_true - y_pred)"})
            st.plotly_chart(fig_hist, use_container_width=True)

    # === Podsumowanie ===
    with tabs[0]:
        if predictions_rows:
            preds_df = pd.DataFrame(predictions_rows)
            preds_df = preds_df.sort_values("Prognoza (PLN)").reset_index(drop=True)
            st.subheader("🎯 Prognozy dla wybranych parametrów")
            st.dataframe(style_pl(preds_df), use_container_width=True)

        if results_rows:
            results_df = pd.DataFrame(results_rows)
            cols_order = ["Model", "MAPE [%]", "MAE", "RMSE", "R²", "MedAE"]
            results_df = results_df[cols_order].sort_values("RMSE").reset_index(drop=True)
            st.subheader("Metryki jakości")
            st.dataframe(style_pl(results_df), use_container_width=True)

            if predictions_rows and not results_df.empty:
                best_model = results_df.iloc[0]["Model"]
                best_val = preds_df.loc[preds_df["Model"] == best_model, "Prognoza (PLN)"].values
                if best_val.size:
                    best_txt = f"{best_val[0]:,.2f}".replace(",", "X").replace(".", ",").replace("X", " ")
                    st.success(f"**Rekomendacja (najniższy RMSE):** {best_model} → {best_txt} PLN")

else:
    st.caption("Ustaw parametry (miasto + ulica + cechy), wybierz modele i kliknij **Prognozuj cenę**.")
