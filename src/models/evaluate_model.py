import pandas as pd
import joblib
import json
from sklearn.metrics import mean_squared_error, r2_score
import os

def main():
    # 1. Charger le modèle entraîné
    model = joblib.load("models/model.pkl")
    print("✔️ Modèle chargé avec succès.")

    # 2. Charger les données de test
    X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
    y_test = pd.read_csv("data/processed_data/y_test.csv").values.ravel()  # Convertir en 1D

    # 3. Faire des prédictions
    y_pred = model.predict(X_test)

    # 4. Calculer les métriques d'évaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {"Mean Squared Error": mse, "R2 Score": r2}

    # 5. Afficher les résultats
    print(f"✔️ Mean Squared Error: {mse:.4f}")
    print(f"✔️ R2 Score: {r2:.4f}")

    # 6. Sauvegarder les métriques dans un fichier JSON
    output_dir = "metrics/"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    
    print("✔️ Métriques sauvegardées dans 'metrics/metrics.json'.")

if __name__ == "__main__":
    main()
