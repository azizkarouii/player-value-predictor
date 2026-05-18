# 🎯 Phase 3 — Streamlit App

Application interactive Streamlit pour tester et déployer le pipeline ML de prédiction de valeur marchande des joueurs de football.

## 📋 Fonctionnalités

### 1. 📊 Overview
- Visualisation générale du dataset réel (CSV du projet)
- Distribution des positions, valeurs marchandes, âges
- Top 3 meilleurs modèles
- Graphiques comparatifs des 6 modèles

### 2. 🎯 Prédictions personnalisées
- Interface interactive pour entrer les caractéristiques d'un joueur
- Prédictions simultanées par les 6 modèles
- Statistiques agrégées (moyenne, écart-type, min, max)
- Graphique comparatif des prédictions
- Champs simplifiés (variables secondaires retirées)

### 3. 📈 Comparaison des modèles
- Tableau résumé des performances (classé par R²)
- Scatter plots Réel vs Prédit pour chaque modèle
- Visualisation de la qualité des prédictions avec:
	- échelle logarithmique
	- clipping des valeurs extrêmes pour améliorer la lisibilité

### 4. 🔍 Détails des modèles
- Sélection d'un modèle pour analyser ses détails
- Affichage des métriques (R², RMSE, MAE, MAPE)
- Distribution des erreurs
- Détails des hyperparamètres
- Tableau lisible des couches pour les modèles Deep Learning (sans bloc ASCII Keras)

## 🗂️ Données utilisées

L'application entraîne les modèles sur les **données réelles** du projet:

- `player_latest_market_value/player_latest_market_value.csv` (cible)
- `player_profiles/player_profiles.csv`
- `player_performances/player_performances.csv`
- `player_injuries/player_injuries.csv`

Le pipeline construit des features agrégées par joueur puis entraîne les modèles sur la cible `market_value` (transformée en log1p pendant l'apprentissage).

## 🚀 Installation & Lancement

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Lancer l'application

```bash
streamlit run streamlit_app.py
```

L'app s'ouvrira automatiquement dans votre navigateur (http://localhost:8501)

## 📊 Modèles inclus

### ML Classiques (3)
- **Régression Linéaire** — Baseline
- **Random Forest** — 300 estimateurs, max_depth=20
- **XGBoost** — 500 estimateurs, learning_rate=0.05

### Deep Learning (3)
- **ANN1 MLP Shallow** — 2 couches, BatchNormalization
- **ANN2 Deep+BatchNorm** — 3 couches + BatchNorm + L2 regularization
- **ANN3 Residual MLP** — Architecture résiduelle tabulaire

## 📈 Métriques utilisées

- **R²** — Coefficient de détermination (plus haut = meilleur)
- **RMSE** — Racine de l'erreur quadratique moyenne
- **MAE** — Erreur absolue moyenne
- **MAPE** — Pourcentage d'erreur absolue moyenne

## 🎨 Navigation

Utilisez la barre latérale gauche pour naviguer entre les 4 sections :
- 📊 Overview
- 🎯 Prédictions
- 📈 Comparaison modèles
- 🔍 Détails hyperparamètres

## ⚙️ Configuration

Le fichier `requirements.txt` contient toutes les dépendances nécessaires. 
Si vous rencontrez des problèmes avec TensorFlow/Keras, essayez :

```bash
pip install --upgrade tensorflow
```

## 📝 Notes

- Les modèles sont entraînés au démarrage (premier lancement potentiellement plus long, car CSV réels)
- Streamlit cache les résultats pour les lancements ultérieurs
- Le dataset synthétique n'est plus utilisé dans la version actuelle

## 🐛 Troubleshooting

**L'app ne se lance pas ?**
- Vérifiez que vous êtes dans le bon répertoire
- Activez votre environnement virtuel
- Assurez-vous que Streamlit est installé : `pip install streamlit`

**Les modèles sont trop lents ?**
- C'est normal sur CPU (entraînement potentiellement long sur les données réelles)
- Les lancements suivants sont instantanés (cache Streamlit)

**Les graphiques ne s'affichent pas ?**
- Actualisez la page (F5)
- Vérifiez que matplotlib est installé

## 📞 Support

Pour toute question sur le pipeline Phase 3, consultez :
- `phase3_pipeline.py` — Code du pipeline original
- `phase3_pipeline.ipynb` — Notebook Jupyter avec explications
