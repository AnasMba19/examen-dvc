# Examen DVC - DataScientest

**Nom et Prénom** : Anas Mbarki  
**Email** : anasmbarki.amk@gmail.com  
**Lien vers le dépôt DagsHub** : [https://dagshub.com/AnasMba19/examen-dvc]

## Description du projet
Ce projet met en place un pipeline DVC permettant de gérer un workflow de machine learning structuré et automatisé. Il comprend les étapes suivantes :
- **Prétraitement des données** : Séparation en jeu d'entraînement et de test, normalisation
- **Optimisation des hyperparamètres** : Recherche des meilleurs paramètres via GridSearchCV
- **Entraînement du modèle** : Modèle RandomForestRegressor entraîné sur les données traitées
- **Évaluation du modèle** : Calcul des métriques de performance (MSE, R²)
- **Gestion des données et du modèle avec DVC et DagsHub**

## Structure du projet
Le projet est organisé comme suit :
- **`src/data/`** : Scripts de prétraitement des données (`split_data.py`, `normalize_data.py`)
- **`src/models/`** : Scripts d'entraînement et d'évaluation du modèle (`gridsearch_model.py`, `train_model.py`, `evaluate_model.py`)
- **`data/`** : Données brutes et transformées
- **`models/`** : Modèle entraîné (`model.pkl`)
- **`metrics/`** : Résultats des métriques (`metrics.json`)
- **`dvc.yaml`** : Définition du pipeline DVC
- **`dvc.lock`** : Fichier de verrouillage DVC pour garantir la reproductibilité

---

## Instructions d'exécution
Pour exécuter le pipeline DVC sur un nouvel environnement, suivez les étapes suivantes :

```bash
git clone https://github.com/AnasMba19/examen-dvc.git
cd examen-dvc
dvc repro
