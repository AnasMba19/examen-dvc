import pandas as pd
import os
from sklearn.model_selection import train_test_split

def main():
    # Vérifier que les données brutes existent
    raw_data_path = "data/raw/raw.csv"
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"❌ Le fichier '{raw_data_path}' est introuvable.")

    # Charger les données brutes
    df = pd.read_csv(raw_data_path)

    # Séparer les features (X) et la cible (y)
    X = df.iloc[:, :-1]  # Toutes les colonnes sauf la dernière
    y = df.iloc[:, -1]   # La dernière colonne comme target

    # Séparer en train et test (80%-20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vérifier et créer le dossier `processed_data` si nécessaire
    output_dir = "data/processed_data/"
    os.makedirs(output_dir, exist_ok=True)

    # Sauvegarder les datasets
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print("✔️ Données séparées et enregistrées dans 'data/processed_data/'.")

if __name__ == "__main__":
    main()
