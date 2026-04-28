"""
=============================================================================
Phase 3 – Pipeline CRISP-DM : Prédiction valeur marchande footballeurs
3 modèles ML  : Régression Linéaire, Random Forest, XGBoost
3 modèles ANN : MLP Shallow, Deep+BatchNorm, ResNet tabulaire
=============================================================================
Durée estimée : ~6-10 min (CPU)
=============================================================================
"""

import os, warnings, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
tf.get_logger().setLevel("ERROR")
tf.random.set_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATASET SYNTHÉTIQUE
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(n=5000, seed=42):
    rng = np.random.default_rng(seed)

    POS  = ["ATT","MIL","DEF","GK"]
    pos  = rng.choice(POS, p=[0.307, 0.347, 0.301, 0.045], size=n)
    ages = np.clip(rng.normal(25.8, 4.2, n), 16, 43).astype(int)

    big5   = ["PL","LL","BL","SA","L1"]
    others = ["ED","LP","ST","ML","SB"]
    probs  = [0.14]*5 + [0.06]*5
    leagues = rng.choice(big5 + others, p=probs, size=n)
    in_big5 = np.isin(leagues, big5).astype(int)

    MG = {"ATT":22,"MIL":7,"DEF":2,"GK":0}
    SG = {"ATT":12,"MIL":6,"DEF":3,"GK":1}
    MA = {"ATT":12,"MIL":10,"DEF":4,"GK":1}
    SA = {"ATT":8, "MIL":7, "DEF":4,"GK":2}

    goals   = np.array([max(0, int(rng.normal(MG[p], SG[p]))) for p in pos])
    assists = np.array([max(0, int(rng.normal(MA[p], SA[p]))) for p in pos])
    minutes = np.clip(rng.normal(6500, 2000, n), 0, 10000).astype(int)
    injury  = np.clip(rng.exponential(100, n),   0, 1200).astype(int)
    intl    = (rng.random(n) > 0.5).astype(int)
    ycards  = np.clip(rng.poisson(4, n), 0, 25)

    PC = {"ATT":0.5,"MIL":0.3,"DEF":0.1,"GK":-0.1}
    log_val = np.clip(
        13.5
        + np.array([PC[p] for p in pos])
        - 0.04 * (ages - 26) ** 2
        + 0.025 * goals
        + 0.020 * assists
        + 0.0001 * minutes
        - 0.003 * injury
        + 0.60  * in_big5
        + 0.30  * intl
        + rng.normal(0, 0.5, n),
        10.1, 19.0
    )

    return pd.DataFrame({
        "pos": pos, "age": ages, "league": leagues, "in_big5": in_big5,
        "goals": goals, "assists": assists, "minutes": minutes,
        "injury": injury, "intl": intl, "ycards": ycards,
        "log_val": log_val,
        "market_value": np.exp(log_val).astype(int),
    })


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURES & SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def prepare(df):
    d = df.copy()
    d["pos_enc"]     = LabelEncoder().fit_transform(d["pos"])
    d["league_enc"]  = LabelEncoder().fit_transform(d["league"])
    d["age2"]        = d["age"] ** 2
    d["g90"]         = d["goals"] / (d["minutes"] / 90 + 1e-5)
    d["log_inj"]     = np.log1p(d["injury"])

    FEATS = ["pos_enc","age","age2","league_enc","in_big5",
             "goals","assists","minutes","injury","log_inj",
             "intl","ycards","g90"]

    X = d[FEATS].values.astype(np.float32)
    y = d["log_val"].values.astype(np.float32)     # cible = log(valeur €)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)

    sc   = StandardScaler()
    X_tr  = sc.fit_transform(X_tr).astype(np.float32)
    X_val = sc.transform(X_val).astype(np.float32)
    X_te  = sc.transform(X_te).astype(np.float32)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# ─────────────────────────────────────────────────────────────────────────────
# 3. MÉTRIQUES  (espace euros après exp)
# ─────────────────────────────────────────────────────────────────────────────

def mets(y_log, p_log):
    y = np.exp(y_log);  p = np.exp(p_log)
    return {
        "RMSE": int(np.sqrt(mean_squared_error(y, p))),
        "MAE" : int(mean_absolute_error(y, p)),
        "R2"  : round(r2_score(y, p), 4),
        "MAPE": round(np.mean(np.abs((y - p) / (y + 1e-8))) * 100, 2),
    }

def ev(model, Xte, yte):
    p = model.predict(Xte)
    if hasattr(p, "flatten"): p = p.flatten()
    return mets(yte, p)

CB = lambda p=12: [
    EarlyStopping(monitor="val_loss", patience=p, restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(4, p//3), verbose=0),
]


# ─────────────────────────────────────────────────────────────────────────────
# 4. ML CLASSIQUES
# ─────────────────────────────────────────────────────────────────────────────

def run_ml(Xtr, ytr, Xval, yval, Xte, yte):
    RES, HP = {}, {}

    # ── Régression Linéaire ──────────────────────────────────────────────────
    print("\n[ML1] Régression Linéaire (baseline)...")
    t0 = time.time()
    m  = LinearRegression().fit(Xtr, ytr)
    r  = ev(m, Xte, yte); r["t"] = round(time.time()-t0, 2)
    print(f"  R2={r['R2']}  MAE={r['MAE']:,}  MAPE={r['MAPE']}%  ({r['t']}s)")
    RES["Reg Lineaire"] = (m, r)

    # ── Random Forest ────────────────────────────────────────────────────────
    print("\n[ML2] Random Forest (10 configs)...")
    RF_CFGS = [
        (100, 10,   "RF-1"),  (100, 20,   "RF-2"),  (100, None, "RF-3"),
        (200, 10,   "RF-4"),  (200, 20,   "RF-5"),  (200, None, "RF-6"),
        (300, 20,   "RF-7 ✓"),(300, None, "RF-8"),
        (500, 20,   "RF-9"),  (500, None, "RF-10"),
    ]
    rows, bR2, bM = [], -9, None
    for ne, md, lbl in RF_CFGS:
        t0 = time.time()
        m  = RandomForestRegressor(n_estimators=ne, max_depth=md,
                                    min_samples_split=5, min_samples_leaf=2,
                                    max_features="sqrt", n_jobs=-1, random_state=42)
        m.fit(Xtr, ytr)
        r = ev(m, Xte, yte); r["t"] = round(time.time()-t0, 2)
        rows.append({"Config":lbl,"n_est":ne,"max_depth":str(md),**r})
        print(f"  {lbl:10s}  n={ne:3d}  depth={str(md):4s}  R2={r['R2']}  MAPE={r['MAPE']}%  ({r['t']}s)")
        if r["R2"] > bR2: bR2, bM = r["R2"], m
    HP["RF"] = pd.DataFrame(rows)
    RES["Random Forest"] = (bM, ev(bM, Xte, yte))

    # ── XGBoost ──────────────────────────────────────────────────────────────
    print("\n[ML3] XGBoost (10 configs)...")
    XG_CFGS = [
        (0.30,200,6,"XG-1"),(0.30,200,8,"XG-2"),
        (0.10,300,6,"XG-3"),(0.10,300,8,"XG-4"),(0.10,300,10,"XG-5"),
        (0.05,400,6,"XG-6"),(0.05,400,8,"XG-7"),
        (0.05,500,8,"XG-8 ✓"),(0.05,500,10,"XG-9"),
        (0.01,800,8,"XG-10"),
    ]
    rows, bR2, bM = [], -9, None
    for lr, ne, md, lbl in XG_CFGS:
        t0 = time.time()
        m  = xgb.XGBRegressor(
            learning_rate=lr, n_estimators=ne, max_depth=md,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3,
            random_state=42, verbosity=0,
            eval_metric="rmse", early_stopping_rounds=20,
        )
        m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
        r = ev(m, Xte, yte); r["t"] = round(time.time()-t0, 2)
        rows.append({"Config":lbl,"lr":lr,"n_est":ne,"max_depth":md,**r})
        print(f"  {lbl:10s}  lr={lr:.2f}  n={ne:3d}  d={md}  "
              f"R2={r['R2']}  MAPE={r['MAPE']:.2f}%  ({r['t']}s)")
        if r["R2"] > bR2: bR2, bM = r["R2"], m
    HP["XGB"] = pd.DataFrame(rows)
    RES["XGBoost"] = (bM, ev(bM, Xte, yte))

    return RES, HP


# ─────────────────────────────────────────────────────────────────────────────
# 5. ANN  (BN en 1ère couche, lr=3e-4, MSE sur cible log)
# ─────────────────────────────────────────────────────────────────────────────

def run_ann(Xtr, ytr, Xval, yval, Xte, yte):
    D = Xtr.shape[1]
    RES, HP = {}, {}

    # ── ANN1 : MLP Shallow ───────────────────────────────────────────────────
    print("\n[ANN1] MLP Shallow (6 configs)...")
    A1_CFGS = [
        (64,  32,  0.1, 0.0, "A1-1"),
        (128, 64,  0.1, 0.0, "A1-2"),
        (128, 64,  0.2, 0.1, "A1-3 ✓"),
        (256, 128, 0.2, 0.1, "A1-4"),
        (256, 128, 0.3, 0.2, "A1-5"),
        (512, 256, 0.3, 0.2, "A1-6"),
    ]
    rows, bR2, bM = [], -9, None
    for u1, u2, d1, d2, lbl in A1_CFGS:
        t0  = time.time()
        inp = keras.Input(shape=(D,))
        x   = layers.BatchNormalization()(inp)   # stabilise l'entrée
        x   = layers.Dense(u1, activation="relu", kernel_initializer="he_uniform")(x)
        if d1 > 0: x = layers.Dropout(d1)(x)
        x   = layers.Dense(u2, activation="relu", kernel_initializer="he_uniform")(x)
        if d2 > 0: x = layers.Dropout(d2)(x)
        out = layers.Dense(1)(x)
        m   = keras.Model(inp, out)
        m.compile(optimizer=keras.optimizers.Adam(3e-4), loss="mse")
        m.fit(Xtr, ytr, validation_data=(Xval, yval),
              epochs=80, batch_size=256, callbacks=CB(8), verbose=0)
        r = ev(m, Xte, yte); r["t"] = round(time.time()-t0, 2)
        rows.append({"Config":lbl,"Units":f"[{u1},{u2}]","Drop":f"[{d1},{d2}]",**r})
        print(f"  {lbl:10s}  [{u1},{u2}]  drop=[{d1},{d2}]  "
              f"R2={r['R2']}  MAPE={r['MAPE']:.2f}%  ({r['t']}s)")
        if r["R2"] > bR2: bR2, bM = r["R2"], m
    HP["ANN1"] = pd.DataFrame(rows)
    RES["ANN1 MLP"] = (bM, ev(bM, Xte, yte))

    # ── ANN2 : Deep + BatchNorm ───────────────────────────────────────────────
    print("\n[ANN2] Deep + BatchNorm (5 configs)...")
    A2_CFGS = [
        ((256,128,64),  (0.2,0.1,0.0), 3e-4, "A2-1"),
        ((256,128,64),  (0.2,0.1,0.0), 1e-4, "A2-2"),
        ((512,256,128), (0.3,0.2,0.1), 3e-4, "A2-3 ✓"),
        ((512,256,128), (0.3,0.2,0.1), 1e-4, "A2-4"),
        ((512,256,128), (0.2,0.1,0.0), 3e-4, "A2-5"),
    ]
    rows, bR2, bM = [], -9, None
    for arch, drops, lr, lbl in A2_CFGS:
        t0  = time.time()
        inp = keras.Input(shape=(D,))
        x   = layers.BatchNormalization()(inp)
        for u, d in zip(arch, drops):
            x = layers.Dense(u, activation="relu",
                             kernel_regularizer=regularizers.l2(1e-4))(x)
            x = layers.BatchNormalization()(x)
            if d > 0: x = layers.Dropout(d)(x)
        out = layers.Dense(1)(x)
        m   = keras.Model(inp, out)
        m.compile(optimizer=keras.optimizers.Adam(lr),
                  loss=keras.losses.Huber(delta=1.0))
        m.fit(Xtr, ytr, validation_data=(Xval, yval),
              epochs=100, batch_size=256, callbacks=CB(10), verbose=0)
        r = ev(m, Xte, yte); r["t"] = round(time.time()-t0, 2)
        rows.append({"Config":lbl,"arch":str(arch),"lr":lr,**r})
        print(f"  {lbl:10s}  {str(arch):18s}  lr={lr}  "
              f"R2={r['R2']}  MAPE={r['MAPE']:.2f}%  ({r['t']}s)")
        if r["R2"] > bR2: bR2, bM = r["R2"], m
    HP["ANN2"] = pd.DataFrame(rows)
    RES["ANN2 Deep+BN"] = (bM, ev(bM, Xte, yte))

    # ── ANN3 : ResNet tabulaire ───────────────────────────────────────────────
    print("\n[ANN3] ResNet tabulaire (6 configs)...")

    def res_block(x, u):
        sc = x
        x  = layers.Dense(u, activation="relu")(x)
        x  = layers.BatchNormalization()(x)
        x  = layers.Dense(u)(x)
        x  = layers.BatchNormalization()(x)
        if sc.shape[-1] != u: sc = layers.Dense(u)(sc)
        return layers.Activation("relu")(layers.Add()([x, sc]))

    A3_CFGS = [
        (2, 128, 64,  0.2, 3e-4, "A3-1"),
        (2, 256, 128, 0.2, 3e-4, "A3-2"),
        (3, 256, 128, 0.2, 3e-4, "A3-3 ✓"),
        (3, 256, 128, 0.2, 1e-4, "A3-4"),
        (4, 256, 128, 0.2, 3e-4, "A3-5"),
        (3, 512, 256, 0.2, 3e-4, "A3-6"),
    ]
    rows, bR2, bM = [], -9, None
    for nb, d1, d2, drop, lr, lbl in A3_CFGS:
        t0  = time.time()
        inp = keras.Input(shape=(D,))
        x   = layers.BatchNormalization()(inp)
        x   = layers.Dense(d1)(x)
        for _ in range(min(2, nb)):
            x = res_block(x, d1)
        if nb >= 3:
            x = layers.Dense(d2)(x)
            x = res_block(x, d2)
        if nb >= 4:
            x = res_block(x, d2)
        x   = layers.Dropout(drop)(x)
        x   = layers.Dense(64, activation="relu")(x)
        out = layers.Dense(1)(x)
        m   = keras.Model(inp, out)
        m.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-3),
            loss=keras.losses.LogCosh(),
        )
        m.fit(Xtr, ytr, validation_data=(Xval, yval),
              epochs=120, batch_size=256, callbacks=CB(12), verbose=0)
        r = ev(m, Xte, yte); r["t"] = round(time.time()-t0, 2)
        rows.append({"Config":lbl,"n_blocs":nb,"dim":f"{d1}→{d2}","lr":lr,**r})
        print(f"  {lbl:10s}  blocs={nb}  {d1}→{d2}  lr={lr}  "
              f"R2={r['R2']}  MAPE={r['MAPE']:.2f}%  ({r['t']}s)")
        if r["R2"] > bR2: bR2, bM = r["R2"], m
    HP["ANN3"] = pd.DataFrame(rows)
    RES["ANN3 ResNet"] = (bM, ev(bM, Xte, yte))

    return RES, HP


# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

C6 = ["#c0392b","#27ae60","#1e8449","#2980b9","#8e44ad","#16a085"]

def plot_comparison(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Comparaison des 6 modèles — Test set", fontsize=13, fontweight="bold")
    labels = df["Modele"].tolist()
    for ax, col, lbl in zip(axes, ["R2","MAE","MAPE"],
                            ["R² (↑)","MAE en € (↓)","MAPE % (↓)"]):
        vals = df[col].tolist()
        bars = ax.barh(labels, vals, color=C6, edgecolor="white")
        fmt  = "%.4f" if col=="R2" else ("%.0f" if col=="MAE" else "%.1f%%")
        ax.bar_label(bars, fmt=fmt, padding=4, fontsize=8)
        ax.set_xlabel(lbl); ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3, ls="--")
        ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("results_comparison.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  -> results_comparison.png")

def plot_bar(df, title, cy, cn, fname):
    fig, ax = plt.subplots(figsize=(10, 4))
    cols = [cy if "✓" in c else cn for c in df["Config"]]
    bars = ax.bar(range(len(df)), df["R2"], color=cols, edgecolor="white")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["Config"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("R²"); ax.set_title(title, fontweight="bold")
    ax.bar_label(bars, fmt="%.4f", padding=2, fontsize=8)
    ax.axhline(df["R2"].max(), color="red", ls="--", lw=1,
               label=f"Best={df['R2'].max():.4f}")
    ax.legend(fontsize=8)
    ax.set_ylim(max(0, df["R2"].min()*0.94), df["R2"].max()*1.04)
    ax.grid(axis="y", alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  -> {fname}")

def plot_xgb_lines(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("XGBoost — Impact des hyperparamètres", fontsize=12, fontweight="bold")
    cols = ["#e67e22" if "✓" in c else "#fad7a0" for c in df["Config"]]
    axes[0].bar(range(len(df)), df["R2"], color=cols, edgecolor="white")
    axes[0].set_xticks(range(len(df)))
    axes[0].set_xticklabels(df["Config"], rotation=35, ha="right", fontsize=8)
    axes[0].set_ylabel("R²"); axes[0].set_title("R² par configuration")
    axes[0].axhline(df["R2"].max(), color="red", ls="--", lw=1)
    axes[0].set_ylim(max(0, df["R2"].min()*0.94), df["R2"].max()*1.04)
    axes[0].grid(axis="y", alpha=0.3)
    for lr_v, grp in df.groupby("lr"):
        axes[1].plot(grp["n_est"], grp["MAPE"], marker="o", label=f"lr={lr_v}")
    axes[1].set_xlabel("n_estimators"); axes[1].set_ylabel("MAPE (%)")
    axes[1].set_title("MAPE vs n_estimators par learning_rate")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("hyperparam_xgb.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  -> hyperparam_xgb.png")

def plot_scatter(all_res, Xte, yte):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for ax, ((name,(m,_)), color) in zip(axes, zip(all_res.items(), C6)):
        p = m.predict(Xte)
        if hasattr(p,"flatten"): p = p.flatten()
        yt = np.exp(yte)/1e6;  yp = np.exp(p)/1e6
        ax.scatter(yt, yp, alpha=0.25, s=8, color=color)
        lim = max(yt.max(), yp.max())*1.05
        ax.plot([0,lim],[0,lim],"k--",lw=0.8)
        ax.set_xlabel("Réel (M€)",fontsize=9); ax.set_ylabel("Prédit (M€)",fontsize=9)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.text(0.05,0.92,f"R²={r2_score(yt,yp):.3f}",transform=ax.transAxes,
                fontsize=9,bbox=dict(boxstyle="round",facecolor="white",alpha=0.7))
        ax.grid(alpha=0.3)
    fig.suptitle("Réel vs Prédit — test set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("predictions_scatter.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  -> predictions_scatter.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    T0 = time.time()
    SEP = "=" * 65

    print(SEP)
    print("  PHASE 3 — Pipeline CRISP-DM  Football Market Value")
    print(SEP)

    print("\n[DATA] Génération du dataset (n=5000)...")
    df = generate_dataset(5000)
    print(f"  {df.shape[0]} joueurs  |  "
          f"med={df['market_value'].median()/1e6:.2f}M€  "
          f"max={df['market_value'].max()/1e6:.1f}M€  "
          f"min={df['market_value'].min()/1e6:.3f}M€")

    Xtr, Xval, Xte, ytr, yval, yte = prepare(df)
    print(f"  Train={len(Xtr)}  Val={len(Xval)}  Test={len(Xte)}  Features={Xtr.shape[1]}")

    ml_res,  ml_hp  = run_ml(Xtr, ytr, Xval, yval, Xte, yte)
    ann_res, ann_hp = run_ann(Xtr, ytr, Xval, yval, Xte, yte)

    all_res = {**ml_res, **ann_res}

    # ── Tableau récapitulatif ─────────────────────────────────────────────────
    summary = pd.DataFrame(
        [{"Modele": n, **r} for n,(_, r) in all_res.items()]
    ).sort_values("R2", ascending=False).reset_index(drop=True)
    summary.index += 1

    print(f"\n{SEP}")
    print("  RÉSULTATS FINAUX — test set (meilleures configs)")
    print(SEP)
    print(summary.to_string())

    # ── Export CSV ────────────────────────────────────────────────────────────
    print(f"\n[CSV] Export...")
    summary.to_csv("results_summary.csv", index=False)
    for k, dfh in {**ml_hp, **ann_hp}.items():
        fname = f"hyperparam_{k.lower()}.csv"
        dfh.to_csv(fname, index=False)
        print(f"  -> {fname}")
    print("  -> results_summary.csv")

    # ── Graphiques ────────────────────────────────────────────────────────────
    print("\n[VIZ] Génération des graphiques...")
    plot_comparison(summary)
    plot_bar(ml_hp["RF"],   "Random Forest — R² par config", "#27ae60","#aed6f1","hyperparam_rf.png")
    plot_xgb_lines(ml_hp["XGB"])
    plot_bar(ann_hp["ANN1"],"ANN1 MLP Shallow — R²",        "#2980b9","#d6eaf8","hyperparam_ann1.png")
    plot_bar(ann_hp["ANN2"],"ANN2 Deep+BatchNorm — R²",     "#8e44ad","#e8daef","hyperparam_ann2.png")
    plot_bar(ann_hp["ANN3"],"ANN3 ResNet tabulaire — R²",   "#16a085","#d1f2eb","hyperparam_ann3.png")
    plot_scatter(all_res, Xte, yte)

    print(f"\n{SEP}")
    print(f"  Pipeline terminé en {round(time.time()-T0, 1)}s")
    print(f"  Fichiers générés :")
    for f in [
        "results_summary.csv",
        "hyperparam_rf.csv","hyperparam_xgb.csv",
        "hyperparam_ann1.csv","hyperparam_ann2.csv","hyperparam_ann3.csv",
        "results_comparison.png","hyperparam_rf.png","hyperparam_xgb.png",
        "hyperparam_ann1.png","hyperparam_ann2.png","hyperparam_ann3.png",
        "predictions_scatter.png",
    ]:
        print(f"    {f}")
    print(SEP)
