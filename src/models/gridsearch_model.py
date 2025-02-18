import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import os
import joblib

def main():
    # 1. Charger les données
    X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()  # Convertir en 1D

    # 2. Définir le modèle et la grille d'hyperparamètres
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    # 3. Exécuter GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # 4. Afficher les meilleurs paramètres
    best_params = grid_search.best_params_
    print(f"✔️ Meilleurs hyperparamètres : {best_params}")

    # 5. Sauvegarder les meilleurs hyperparamètres
    output_dir = "models/"
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(best_params, os.path.join(output_dir, "best_params.pkl"))

if __name__ == "__main__":
    main()
