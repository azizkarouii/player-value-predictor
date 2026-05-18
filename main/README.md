# player-value-predictor

Application de prediction de valeur marchande des joueurs de football avec pipeline ML + Deep Learning et interface Streamlit.

## Apercu

Ce projet entraine 6 modeles sur les donnees reelles du projet (agrégation par joueur):
- 3 modeles ML classiques: Regression Lineaire, Random Forest, XGBoost
- 3 modeles Deep Learning: ANN1 MLP, ANN2 Deep+BN, ANN3 ResNet

L'application Streamlit permet:
- visualisation des donnees et resultats
- prediction personnalisee d'un joueur
- comparaison des modeles
- analyse des erreurs et details du modele

## Lancement rapide

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Structure utile

- `streamlit_app.py`: application principale
- `STREAMLIT_README.md`: documentation detaillee de l'app
- `phase3_pipeline.py`: pipeline initial de la phase 3

## Notes

- Les modeles sont entraines au demarrage de l'app
- Les donnees volumineuses et artefacts sont ignores par `.gitignore`
