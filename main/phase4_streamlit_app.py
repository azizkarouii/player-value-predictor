"""
=============================================================================
STREAMLIT APP — Phase 3 ML Pipeline
Football Player Market Value Prediction
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, warnings, time
from pathlib import Path
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
tf.get_logger().setLevel("ERROR")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phase 3 — Market Value Prediction",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 20px; }
    .big-font { font-size: 20px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CACHE & FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _years_between(start_series, end_timestamp):
    delta = (end_timestamp - start_series).dt.days
    return delta / 365.25

@st.cache_data
def load_real_training_data(base_dir="."):
    """Construit une table d'entraînement réelle à partir des CSV du projet."""
    root = Path(base_dir).resolve()
    today = pd.Timestamp.today().normalize()

    latest = pd.read_csv(root / "player_latest_market_value/player_latest_market_value.csv")
    latest = latest.groupby("player_id", as_index=False)["value"].max()
    latest = latest.rename(columns={"value": "market_value"})

    profiles = pd.read_csv(
        root / "player_profiles/player_profiles.csv",
        usecols=[
            "player_id", "date_of_birth", "height", "is_eu", "position",
            "main_position", "foot", "joined", "contract_expires"
        ],
    )
    profiles["date_of_birth"] = pd.to_datetime(profiles["date_of_birth"], errors="coerce")
    profiles["joined"] = pd.to_datetime(profiles["joined"], errors="coerce")
    profiles["contract_expires"] = pd.to_datetime(profiles["contract_expires"], errors="coerce")
    profiles["age"] = _years_between(profiles["date_of_birth"], today)
    profiles["years_since_joined"] = _years_between(profiles["joined"], today)
    profiles["contract_years_left"] = (profiles["contract_expires"] - today).dt.days / 365.25
    profiles["foot"] = profiles["foot"].fillna("Unknown")
    profiles["position"] = profiles["position"].fillna("Unknown")
    profiles["main_position"] = profiles["main_position"].fillna("Unknown")
    profiles["height"] = profiles["height"].fillna(profiles["height"].median())
    profiles["age"] = profiles["age"].fillna(profiles["age"].median())
    profiles["years_since_joined"] = profiles["years_since_joined"].fillna(profiles["years_since_joined"].median())
    profiles["contract_years_left"] = profiles["contract_years_left"].fillna(0)

    perf = pd.read_csv(
        root / "player_performances/player_performances.csv",
        usecols=[
            "player_id", "season_name", "competition_id", "goals", "assists",
            "subed_in", "subed_out", "yellow_cards",
            "second_yellow_cards", "direct_red_cards", "penalty_goals",
            "minutes_played"
        ],
    )
    perf_agg = perf.groupby("player_id", as_index=False).agg(
        performance_rows=("season_name", "size"),
        seasons_played=("season_name", "nunique"),
        competitions_played=("competition_id", "nunique"),
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        subed_in=("subed_in", "sum"),
        subed_out=("subed_out", "sum"),
        yellow_cards=("yellow_cards", "sum"),
        second_yellow_cards=("second_yellow_cards", "sum"),
        direct_red_cards=("direct_red_cards", "sum"),
        penalty_goals=("penalty_goals", "sum"),
        minutes_played=("minutes_played", "sum"),
    )

    injuries = pd.read_csv(
        root / "player_injuries/player_injuries.csv",
        usecols=["player_id", "days_missed", "games_missed"],
    )
    injury_agg = injuries.groupby("player_id", as_index=False).agg(
        injury_records=("days_missed", "size"),
        days_missed=("days_missed", "sum"),
        games_missed=("games_missed", "sum"),
        max_days_missed=("days_missed", "max"),
    )

    data = (
        latest.merge(profiles, on="player_id", how="left")
        .merge(perf_agg, on="player_id", how="left")
        .merge(injury_agg, on="player_id", how="left")
    )

    data = data[data["market_value"].notna()].copy()
    data = data[data["market_value"] > 0].copy()

    data["goals_per90"] = data["goals"].fillna(0) / (data["minutes_played"].fillna(0) / 90 + 1e-6)
    data["assists_per90"] = data["assists"].fillna(0) / (data["minutes_played"].fillna(0) / 90 + 1e-6)
    data["yellow_cards_per90"] = data["yellow_cards"].fillna(0) / (data["minutes_played"].fillna(0) / 90 + 1e-6)

    numeric_cols = [
        "age", "height", "is_eu", "years_since_joined", "contract_years_left",
        "performance_rows", "seasons_played", "competitions_played", "goals", "assists",
        "subed_in", "subed_out", "yellow_cards", "second_yellow_cards",
        "direct_red_cards", "penalty_goals", "minutes_played", "injury_records", "days_missed", "games_missed",
        "max_days_missed", "goals_per90", "assists_per90", "yellow_cards_per90",
    ]
    categorical_cols = ["position", "main_position", "foot"]

    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].fillna(data[col].median())

    data["foot"] = data["foot"].fillna("Unknown")
    data["position"] = data["position"].fillna("Unknown")
    data["main_position"] = data["main_position"].fillna("Unknown")

    return data, numeric_cols, categorical_cols

@st.cache_data
def load_dataset_inventory(base_dir="."):
    """Charge l'inventaire des CSV réels du projet."""
    root = Path(base_dir).resolve()
    rows = []

    for csv_path in sorted(root.glob("**/*.csv")):
        if ".venv" in csv_path.parts:
            continue
        if csv_path.parent == root:
            continue
        try:
            header = pd.read_csv(csv_path, nrows=0)
            with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
                row_count = max(sum(1 for _ in handle) - 1, 0)

            rows.append({
                "Fichier": str(csv_path.relative_to(root)).replace("\\", "/"),
                "Lignes": row_count,
                "Colonnes": len(header.columns),
                "Taille (Mo)": round(csv_path.stat().st_size / (1024 * 1024), 1),
            })
        except Exception:
            continue

    inventory = pd.DataFrame(rows)
    if inventory.empty:
        return inventory, {"files": 0, "rows": 0, "cols": 0, "size_mb": 0.0}

    summary = {
        "files": int(len(inventory)),
        "rows": int(inventory["Lignes"].sum()),
        "cols": int(inventory["Colonnes"].sum()),
        "size_mb": float(inventory["Taille (Mo)"].sum()),
    }
    return inventory, summary

@st.cache_resource
def prepare_real_data(df, numeric_cols, categorical_cols):
    """Prépare les features réelles."""
    target = np.log1p(df["market_value"].astype(np.float32).values)
    features = df[numeric_cols + categorical_cols].copy()
    features = pd.get_dummies(features, columns=categorical_cols, dummy_na=False)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(features, target, test_size=0.30, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)

    sc = StandardScaler()
    X_tr_scaled = sc.fit_transform(X_tr).astype(np.float32)
    X_val_scaled = sc.transform(X_val).astype(np.float32)
    X_te_scaled = sc.transform(X_te).astype(np.float32)

    return X_tr_scaled, X_val_scaled, X_te_scaled, y_tr.astype(np.float32), y_val.astype(np.float32), y_te.astype(np.float32), sc, list(features.columns)

def metrics(y_log, p_log):
    """Calcule métriques en euros"""
    y = np.expm1(y_log);  p = np.expm1(p_log)
    return {
        "RMSE": int(np.sqrt(mean_squared_error(y, p))),
        "MAE" : int(mean_absolute_error(y, p)),
        "R2"  : round(r2_score(y, p), 4),
        "MAPE": round(np.mean(np.abs((y - p) / (y + 1e-8))) * 100, 2),
    }

def evaluate_model(model, Xte, yte):
    """Évalue un modèle"""
    if isinstance(model, keras.Model):
        p = model.predict(Xte, verbose=0)
    else:
        p = model.predict(Xte)
    if hasattr(p, "flatten"): p = p.flatten()
    return metrics(yte, p)

def clip_for_plot(real_values, predicted_values, low_q=0.01, high_q=0.99):
    """Casse les extrêmes uniquement pour l'affichage afin de garder une vue lisible."""
    combined = np.concatenate([real_values, predicted_values])
    low = np.quantile(combined, low_q)
    high = np.quantile(combined, high_q)
    low = max(low, 1.0)
    return np.clip(real_values, low, high), np.clip(predicted_values, low, high), low, high

def get_keras_model_overview(model):
    """Construit un tableau lisible des couches Keras."""
    rows = []
    for layer in model.layers:
        try:
            output_shape = getattr(layer, "output_shape", None)
            if output_shape is None and hasattr(layer, "output"):
                output_shape = tuple(layer.output.shape)
        except Exception:
            output_shape = "N/A"

        rows.append({
            "Couche": layer.name,
            "Type": layer.__class__.__name__,
            "Output shape": str(output_shape),
            "Paramètres": int(layer.count_params()),
        })

    return pd.DataFrame(rows)

@st.cache_resource
def train_models(X_tr, X_val, X_te, y_tr, y_val, y_te):
    """Entraîne les 6 modèles"""
    results = {}
    
    # Régression Linéaire
    m_lr = LinearRegression().fit(X_tr, y_tr)
    results["Linear Regression"] = (m_lr, evaluate_model(m_lr, X_te, y_te))
    
    # Random Forest
    m_rf = RandomForestRegressor(n_estimators=300, max_depth=20,
                                 min_samples_split=5, min_samples_leaf=2,
                                 max_features="sqrt", n_jobs=-1, random_state=42)
    m_rf.fit(X_tr, y_tr)
    results["Random Forest"] = (m_rf, evaluate_model(m_rf, X_te, y_te))
    
    # XGBoost
    m_xgb = xgb.XGBRegressor(
        learning_rate=0.05, n_estimators=500, max_depth=8,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3,
        random_state=42, verbosity=0,
        eval_metric="rmse", early_stopping_rounds=20,
    )
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    results["XGBoost"] = (m_xgb, evaluate_model(m_xgb, X_te, y_te))
    
    # ANN1: MLP Shallow
    inp = keras.Input(shape=(X_tr.shape[1],))
    x   = layers.BatchNormalization()(inp)
    x   = layers.Dense(128, activation="relu", kernel_initializer="he_uniform")(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(64, activation="relu", kernel_initializer="he_uniform")(x)
    x   = layers.Dropout(0.1)(x)
    out = layers.Dense(1)(x)
    m_ann1 = keras.Model(inp, out)
    m_ann1.compile(optimizer=keras.optimizers.Adam(3e-4), loss="mse")
    m_ann1.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=80, batch_size=256, 
              callbacks=[EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0)],
              verbose=0)
    results["ANN1 MLP"] = (m_ann1, evaluate_model(m_ann1, X_te, y_te))
    
    # ANN2: Deep + BatchNorm
    inp = keras.Input(shape=(X_tr.shape[1],))
    x   = layers.BatchNormalization()(inp)
    for u in [512, 256, 128]:
        x = layers.Dense(u, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        if u in [512, 256]:
            x = layers.Dropout(0.3 if u==512 else 0.2)(x)
        else:
            x = layers.Dropout(0.1)(x)
    out = layers.Dense(1)(x)
    m_ann2 = keras.Model(inp, out)
    m_ann2.compile(optimizer=keras.optimizers.Adam(3e-4),
                  loss=keras.losses.Huber(delta=1.0))
    m_ann2.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=100, batch_size=256,
              callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0)],
              verbose=0)
    results["ANN2 Deep+BN"] = (m_ann2, evaluate_model(m_ann2, X_te, y_te))
    
    # ANN3: Residual MLP
    def res_block(x, u):
        sc = x
        x  = layers.Dense(u, activation="relu")(x)
        x  = layers.BatchNormalization()(x)
        x  = layers.Dense(u)(x)
        x  = layers.BatchNormalization()(x)
        if sc.shape[-1] != u: sc = layers.Dense(u)(sc)
        return layers.Activation("relu")(layers.Add()([x, sc]))
    
    inp = keras.Input(shape=(X_tr.shape[1],))
    x   = layers.BatchNormalization()(inp)
    x   = layers.Dense(256)(x)
    x   = res_block(x, 256)
    x   = res_block(x, 256)
    x   = layers.Dense(128)(x)
    x   = res_block(x, 128)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1)(x)
    m_ann3 = keras.Model(inp, out)
    m_ann3.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-3),
        loss=keras.losses.LogCosh(),
    )
    m_ann3.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=120, batch_size=256,
              callbacks=[EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=0)],
              verbose=0)
    results["ANN3 Residual MLP"] = (m_ann3, evaluate_model(m_ann3, X_te, y_te))
    
    return results

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")
page = st.sidebar.radio("Navigation", 
    ["📊 Overview", "🎯 Prédictions", "📈 Comparaison modèles", "🔍 Détails hyperparamètres"])

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("⏳ Initialisation du pipeline..."):
    dataset_inventory, dataset_summary = load_dataset_inventory(".")
    df, numeric_cols, categorical_cols = load_real_training_data(".")
    X_tr, X_val, X_te, y_tr, y_val, y_te, scaler, feature_columns = prepare_real_data(df, numeric_cols, categorical_cols)
    all_models = train_models(X_tr, X_val, X_te, y_tr, y_val, y_te)
    
    summary_df = pd.DataFrame(
        [{"Modèle": n, **r} for n,(_, r) in all_models.items()]
    ).sort_values("R2", ascending=False).reset_index(drop=True)

    train_target = df["market_value"].astype(float)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("⚽ Phase 3 — Football Player Market Value Prediction")
    st.info("Les modèles sont entraînés sur les CSV réels du projet, avec `player_latest_market_value` comme cible et des variables agrégées depuis les autres tables.")
    
    st.markdown("---")
    st.subheader("📋 Inventaire des CSV sources")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fichiers CSV", f"{dataset_summary['files']:,}")
    with col2:
        st.metric("Lignes totales", f"{dataset_summary['rows']:,}")
    with col3:
        st.metric("Colonnes totales", f"{dataset_summary['cols']:,}")
    with col4:
        st.metric("Taille totale", f"{dataset_summary['size_mb']:.1f} Mo")

    if not dataset_inventory.empty:
        st.markdown("### Composition du dataset")
        st.dataframe(dataset_inventory, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("📈 Dataset d'entraînement réel")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Joueurs utilisés", f"{len(df):,}")
        st.metric("Valeur médiane", f"€{train_target.median()/1e6:.2f}M")
        st.metric("Valeur max", f"€{train_target.max()/1e6:.1f}M")
        st.markdown(f"**Distribution des positions**")
        pos_dist = df['position'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(pos_dist.values, labels=pos_dist.index, autopct='%1.1f%%')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown(f"**Distribution des valeurs marchandes réelles**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(train_target/1e6, bins=50, color='#3498db', edgecolor='white', alpha=0.7)
        ax.set_xlabel("Valeur (M€)")
        ax.set_ylabel("Nombre de joueurs")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col3:
        st.markdown(f"**Distribution des âges**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df['age'], bins=30, color='#9b59b6', edgecolor='white', alpha=0.7)
        ax.set_xlabel("Age")
        ax.set_ylabel("Nombre")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("🏆 Meilleurs modèles (Test set)")
    st.dataframe(summary_df.head(3), use_container_width=True)
    
    # Visualisation comparaison
    st.markdown("---")
    st.subheader("📊 Comparaison des 6 modèles")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Comparaison des 6 modèles — Test set", fontsize=13, fontweight="bold")
    
    colors = ["#e74c3c", "#2ecc71", "#27ae60", "#3498db", "#8e44ad", "#16a085"]
    labels = summary_df["Modèle"].tolist()
    
    for ax, col, lbl in zip(axes, ["R2","MAE","MAPE"],
                            ["R² (↑)","MAE en € (↓)","MAPE % (↓)"]):
        vals = summary_df[col].tolist()
        bars = ax.barh(labels, vals, color=colors, edgecolor="white")
        fmt  = "%.4f" if col=="R2" else ("%.0f" if col=="MAE" else "%.1f%%")
        ax.bar_label(bars, fmt=fmt, padding=4, fontsize=8)
        ax.set_xlabel(lbl); ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3, ls="--")
        ax.spines[["top","right"]].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2: PRÉDICTIONS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Prédictions":
    st.title("🎯 Prédiction personnalisée")
    st.markdown("Entrez les caractéristiques d'un joueur pour prédire sa valeur marchande à partir des vrais CSV.")
    
    st.markdown("---")
    
    numeric_defaults = df[numeric_cols].median().to_dict()
    categorical_defaults = {
        "position": sorted(df["position"].dropna().unique().tolist())[0],
        "main_position": sorted(df["main_position"].dropna().unique().tolist())[0],
        "foot": sorted(df["foot"].dropna().unique().tolist())[0],
    }

    cat_options = {
        "position": sorted(df["position"].dropna().unique().tolist()),
        "main_position": sorted(df["main_position"].dropna().unique().tolist()),
        "foot": sorted(df["foot"].dropna().unique().tolist()),
    }

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 15.0, 45.0, float(round(numeric_defaults["age"])), step=0.1)
        height = st.slider("Taille (cm)", 140.0, 220.0, float(round(numeric_defaults["height"])), step=1.0)
        is_eu = st.checkbox("Joueur UE", value=bool(round(numeric_defaults["is_eu"])))
        contract_years_left = st.slider("Années de contrat restantes", -2.0, 10.0, float(round(numeric_defaults["contract_years_left"])), step=0.1)
    with col2:
        position = st.selectbox("Position", cat_options["position"], index=cat_options["position"].index(categorical_defaults["position"]))
        main_position = st.selectbox("Main position", cat_options["main_position"], index=cat_options["main_position"].index(categorical_defaults["main_position"]))
        foot = st.selectbox("Pied", cat_options["foot"], index=cat_options["foot"].index(categorical_defaults["foot"]))
        years_since_joined = st.slider("Années depuis l'arrivée", 0.0, 25.0, float(round(numeric_defaults["years_since_joined"])), step=0.1)
    with col3:
        goals = st.number_input("Buts", min_value=0.0, value=0.0, step=1.0)
        assists = st.number_input("Passes décisives", min_value=0.0, value=0.0, step=1.0)
        minutes_played = st.slider("Minutes jouées", 0.0, 50000.0, float(round(numeric_defaults["minutes_played"])), step=100.0)
        yellow_cards = st.number_input("Cartons jaunes", min_value=0.0, value=0.0, step=1.0)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        direct_red_cards = st.number_input("Cartons rouges directs", min_value=0.0, value=0.0, step=1.0)
        penalty_goals = st.number_input("Buts sur penalty", min_value=0.0, value=0.0, step=1.0)
    with col5:
        injury_records = st.number_input("Nombre de blessures", min_value=0.0, value=0.0, step=1.0)
        days_missed = st.number_input("Jours manqués", min_value=0.0, value=0.0, step=1.0)
        games_missed = st.number_input("Matchs manqués", min_value=0.0, value=0.0, step=1.0)
    with col6:
        st.write(" ")
        st.write(" ")
        st.write(" ")
    
    goals_per90 = goals / (minutes_played / 90 + 1e-6)
    assists_per90 = assists / (minutes_played / 90 + 1e-6)
    yellow_cards_per90 = yellow_cards / (minutes_played / 90 + 1e-6)
    
    feature_row = pd.DataFrame([{
        "age": age,
        "height": height,
        "is_eu": int(is_eu),
        "years_since_joined": years_since_joined,
        "contract_years_left": contract_years_left,
        "performance_rows": numeric_defaults["performance_rows"],
        "seasons_played": numeric_defaults["seasons_played"],
        "competitions_played": numeric_defaults["competitions_played"],
        "goals": goals,
        "assists": assists,
        "subed_in": numeric_defaults["subed_in"],
        "subed_out": numeric_defaults["subed_out"],
        "yellow_cards": yellow_cards,
        "second_yellow_cards": numeric_defaults["second_yellow_cards"],
        "direct_red_cards": direct_red_cards,
        "penalty_goals": penalty_goals,
        "minutes_played": minutes_played,
        "injury_records": injury_records,
        "days_missed": days_missed,
        "games_missed": games_missed,
        "max_days_missed": numeric_defaults["max_days_missed"],
        "goals_per90": goals_per90,
        "assists_per90": assists_per90,
        "yellow_cards_per90": yellow_cards_per90,
        "position": position,
        "main_position": main_position,
        "foot": foot,
    }])
    feature_row = pd.get_dummies(feature_row, columns=categorical_cols, dummy_na=False)
    feature_row = feature_row.reindex(columns=feature_columns, fill_value=0)
    feature_vector_scaled = scaler.transform(feature_row).astype(np.float32)
    
    st.markdown("---")
    st.subheader("🔮 Prédictions par modèle")
    
    predictions = {}
    for name, (model, _) in all_models.items():
        if isinstance(model, keras.Model):
            pred_log = model.predict(feature_vector_scaled, verbose=0)
        else:
            pred_log = model.predict(feature_vector_scaled)
        if hasattr(pred_log, 'flatten'):
            pred_log = pred_log.flatten()[0]
        else:
            pred_log = pred_log[0]
        pred_value = np.expm1(pred_log)
        predictions[name] = pred_value
    
    # Afficher les prédictions
    col1, col2, col3 = st.columns(3)
    colors_grad = ["#e74c3c", "#2ecc71", "#27ae60", "#3498db", "#8e44ad", "#16a085"]
    
    for idx, (name, value) in enumerate(sorted(predictions.items(), key=lambda x: x[1], reverse=True)):
        if idx % 3 == 0:
            col1, col2, col3 = st.columns(3)
        
        with [col1, col2, col3][idx % 3]:
            st.metric(name, f"€{value/1e6:.2f}M", delta=None)
    
    # Moyenne et intervalle
    st.markdown("---")
    pred_mean = np.mean(list(predictions.values()))
    pred_std = np.std(list(predictions.values()))
    pred_min = np.min(list(predictions.values()))
    pred_max = np.max(list(predictions.values()))
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Moyenne prédite", f"€{pred_mean/1e6:.2f}M")
    with col2:
        st.metric("Écart-type", f"€{pred_std/1e6:.2f}M")
    with col3:
        st.metric("Min", f"€{pred_min/1e6:.2f}M")
    with col4:
        st.metric("Max", f"€{pred_max/1e6:.2f}M")
    
    # Graphique
    fig, ax = plt.subplots(figsize=(12, 5))
    names = list(predictions.keys())
    values = [predictions[n]/1e6 for n in names]
    bars = ax.barh(names, values, color=colors_grad, edgecolor="white")
    ax.bar_label(bars, fmt="€%.2fM", padding=5, fontsize=10)
    ax.set_xlabel("Valeur prédite (M€)", fontsize=11)
    ax.set_title("Prédictions par tous les modèles", fontweight="bold", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3: COMPARAISON MODÈLES
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📈 Comparaison modèles":
    st.title("📈 Comparaison des modèles sur le test set")
    
    st.markdown("---")
    st.subheader("📊 Tableau résumé (classé par R²)")
    st.dataframe(summary_df, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🎯 Prédictions vs Réel (scatter plot)")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    colors = ["#e74c3c", "#2ecc71", "#27ae60", "#3498db", "#8e44ad", "#16a085"]
    
    for ax, ((name,(model,_)), color) in zip(axes, zip(all_models.items(), colors)):
        if isinstance(model, keras.Model):
            p = model.predict(X_te, verbose=0)
        else:
            p = model.predict(X_te)
        if hasattr(p,"flatten"): p = p.flatten()
        yt = np.expm1(y_te) / 1e6
        yp = np.expm1(p) / 1e6
        yt_plot, yp_plot, low, high = clip_for_plot(yt, yp)
        ax.scatter(yt_plot, yp_plot, alpha=0.25, s=8, color=color)
        lim = max(yt_plot.max(), yp_plot.max()) * 1.05
        ax.plot([low, lim], [low, lim], "k--", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(low, lim)
        ax.set_ylim(low, lim)
        ax.set_xlabel("Réel (M€)",fontsize=9)
        ax.set_ylabel("Prédit (M€)",fontsize=9)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.text(0.05,0.92,f"R²={r2_score(yt,yp):.3f}",transform=ax.transAxes,
                fontsize=9,bbox=dict(boxstyle="round",facecolor="white",alpha=0.7))
        ax.grid(alpha=0.3)
    
    fig.suptitle("Réel vs Prédit — test set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4: DÉTAILS HYPERPARAMÈTRES
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Détails hyperparamètres":
    st.title("🔍 Détails des modèles")
    
    model_choice = st.selectbox("Sélectionnez un modèle", list(all_models.keys()))
    
    model, metrics_dict = all_models[model_choice]
    
    st.markdown("---")
    st.subheader(f"Résultats: {model_choice}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R²", f"{metrics_dict['R2']:.4f}")
    with col2:
        st.metric("RMSE", f"€{metrics_dict['RMSE']:,}")
    with col3:
        st.metric("MAE", f"€{metrics_dict['MAE']:,}")
    with col4:
        st.metric("MAPE", f"{metrics_dict['MAPE']:.2f}%")
    
    st.markdown("---")
    st.subheader("📋 Détails du modèle")
    
    if hasattr(model, 'get_params'):
        params = model.get_params()
        st.write(f"**Type:** {type(model).__name__}")
        st.json(params)
    elif isinstance(model, keras.Model):
        st.write("**Type:** Keras Neural Network")
        overview = get_keras_model_overview(model)
        total_params = model.count_params()
        trainable_params = int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
        non_trainable_params = int(np.sum([np.prod(w.shape) for w in model.non_trainable_weights]))
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        with stats_col1:
            st.metric("Total params", f"{total_params:,}")
        with stats_col2:
            st.metric("Trainable", f"{trainable_params:,}")
        with stats_col3:
            st.metric("Non-trainable", f"{non_trainable_params:,}")
        st.dataframe(overview, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("📈 Distribution des erreurs")
    
    if isinstance(model, keras.Model):
        p = model.predict(X_te, verbose=0)
    else:
        p = model.predict(X_te)
    if hasattr(p,"flatten"): p = p.flatten()
    yt = np.expm1(y_te);  yp = np.expm1(p)
    errors = np.abs(yt - yp) / 1e6
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(errors, bins=50, color='#3498db', edgecolor='white', alpha=0.7)
    axes[0].set_xlabel("Erreur absolue (M€)")
    axes[0].set_ylabel("Fréquence")
    axes[0].set_title("Distribution des erreurs")
    axes[0].axvline(errors.mean(), color='red', linestyle='--', label=f"Moyenne: €{errors.mean():.2f}M")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].boxplot([errors], labels=[model_choice])
    axes[1].set_ylabel("Erreur absolue (M€)")
    axes[1].set_title("Boxplot des erreurs")
    axes[1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")
st.caption("Phase 3 Pipeline — ML Models for Football Player Market Value Prediction | 2026")
