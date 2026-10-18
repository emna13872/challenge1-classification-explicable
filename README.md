---
title: Classification explicable
sdk: docker
app_port: 7860
---

# Challenge 1 – Application web interactive de classification explicable

Data Mining · II3-IF · ENSI · 2026-2027 — cas métier n° 13 : **Télécom – attrition client** (Telco Customer Churn, IBM).

Une application web (React) appuyée sur une API FastAPI : elle télécharge les données, les prépare, entraîne et compare cinq modèles, puis explique leurs décisions globalement et individu par individu (SHAP et LIME).

| Livrable | Où |
|---|---|
| Application web | `frontend/` (React + Plotly), servie par FastAPI en production |
| API FastAPI documentée | `backend/app/`, documentation interactive sur `/docs` |
| Notebook | `notebooks/analyse_telco.ipynb` (exécuté, avec l'analyse critique) |
| Déploiement | `Dockerfile` (Hugging Face Spaces, Docker) |
| Liens de déploiement et captures d'écran | **à compléter** (voir plus bas) |
| One-pager PDF | `docs/one-pager.pdf` |

## Les six modules

| Module | Onglet | Contenu |
|---|---|---|
| 1 – Données | Données | Téléchargement automatique (GitHub IBM, repli OpenML ; UCI via `ucimlrepo`) ou import CSV avec cible ; aperçu, statistiques, valeurs manquantes, classes ; distributions, taux de la classe positive par variable, corrélations |
| 2 – Préparation | Préparation | Choix des variables, imputation, encodage one-hot, normalisation, découpage paramétrable (stratifié) |
| 3 – Modélisation | Modèles | Régression logistique, arbre, k-NN, Random Forest, XGBoost ; hyperparamètres modifiables ; optimisation par recherche en grille (AUC, 3 plis) |
| 4 – Évaluation | Évaluation | AUC-ROC, PR-AUC, précision, rappel, F1, Gini, KS ; ROC et PR superposées ; matrice de confusion ; curseur de seuil en direct |
| 5 – XAI | Explications | Importance native, importance par permutation (comparée entre modèles), SHAP beeswarm et dépendance ; SHAP waterfall, LIME, comparaison et phrase de synthèse pour un individu du test ou fictif ; audit des variables sensibles |
| 6 – API | API | Toute l'interface passe par l'API FastAPI (`/docs`) |

## Lancer en local

```bash
# 1. API
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend && uvicorn app.main:app --port 8000

# 2. Interface (autre terminal)
cd frontend && npm install && npm run dev      # http://localhost:5173 (les appels /api sont relayés vers le port 8000)
```

Pour tout servir depuis FastAPI : `cd frontend && npm run build`, puis ouvrir `http://localhost:8000`.

## Tests

```bash
pip install -r backend/requirements-dev.txt
cd backend && pytest -q                  # 16 tests de bout en bout sur le vrai jeu Telco (accès Internet requis)
cd frontend && npm test                  # parcours des 6 onglets contre l'API lancée sur le port 8000
```

Ces tests vérifient notamment l'additivité de SHAP (base + contributions = sortie du modèle) pour les cinq modèles.

## Déployer sur Hugging Face Spaces

1. Créer un Space de type **Docker**, y pousser ce dépôt (le bloc YAML en tête de ce fichier configure le port 7860).
2. Le `Dockerfile` construit l'interface, installe l'API et sert le tout sur un seul port.

## Choix de conception

- **React + FastAPI** plutôt que Streamlit : l'énoncé accepte React ; l'API est ainsi un vrai service séparé de l'interface.
- **Une session serveur par onglet de navigateur** (en-tête `X-Session-Id`), conservée en mémoire : un redémarrage du serveur efface les données et modèles.
- **Imputation et normalisation ajustées sur l'entraînement uniquement.** Les explications LIME et les valeurs affichées utilisent l'espace non normalisé, plus lisible ; SHAP porte sur l'espace vu par le modèle.
- **Importance par permutation groupée** par variable d'origine (toutes les colonnes one-hot d'une variable sont permutées ensemble).
- **Échelles SHAP** : probabilité pour arbre, Random Forest et k-NN ; log-odds pour régression logistique et XGBoost. Le graphique l'indique.
- **k-NN** : SHAP par méthode Kernel sur 60 individus seulement (le calcul est lent).

## Limites connues

- Seul le jeu **Telco** a été exécuté de bout en bout. Les jeux `bank_marketing` et `online_shoppers` (UCI via `ucimlrepo`) sont branchés mais n'ont pas pu être testés dans l'environnement de développement (accès à UCI bloqué) : à vérifier au premier lancement.
- Les captures d'écran et les liens de déploiement restent à ajouter.
- Le `Dockerfile` n'a pas été construit dans l'environnement de développement (pas de Docker) ; l'interface compile et l'API est testée hors conteneur.
- Les métriques proviennent d'un seul découpage train/test, sans intervalle de confiance.

## Résultats de référence (notebook, graine 42)

AUC-ROC sur le test : XGBoost 0,848 · régression logistique 0,846 · Random Forest 0,846 · k-NN 0,834 · arbre 0,831. Détails et analyse critique : `notebooks/analyse_telco.ipynb`.
