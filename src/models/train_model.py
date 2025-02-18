import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import os

def main():
    # 1. Charger les données
    X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()  # Convertir en 1D

    # 2. Charger les meilleurs hyperparamètres
    best_params = joblib.load("models/best_params.pkl")
    print(f"✔️ Chargement des hyperparamètres : {best_params}")

    # 3. Entraîner le modèle avec les meilleurs hyperparamètres
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    print("✔️ Modèle entraîné avec succès.")

    # 4. Sauvegarder le modèle
    output_dir = "models/"
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "model.pkl"))
    print("✔️ Modèle sauvegardé dans 'models/model.pkl'.")

if __name__ == "__main__":
    main()
