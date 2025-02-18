import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def main():
    # 1. Charger les fichiers train et test
    X_train = pd.read_csv("data/processed_data/X_train.csv")
    X_test = pd.read_csv("data/processed_data/X_test.csv")

    # 2. Exclure la colonne 'date' et sélectionner uniquement les colonnes numériques
    X_train_numeric = X_train.drop(columns=["date"])
    X_test_numeric = X_test.drop(columns=["date"])

    # 3. Appliquer la normalisation StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_numeric)
    X_test_scaled = scaler.transform(X_test_numeric)

    # 4. Convertir en DataFrame avec les mêmes colonnes
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_numeric.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_numeric.columns)

    # 5. Vérifier et créer le dossier `processed_data` si nécessaire
    output_dir = "data/processed_data/"
    os.makedirs(output_dir, exist_ok=True)

    # 6. Sauvegarder les fichiers normalisés
    X_train_scaled.to_csv(os.path.join(output_dir, "X_train_scaled.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(output_dir, "X_test_scaled.csv"), index=False)

    print("✔️ Données normalisées (sans la colonne 'date') et enregistrées dans 'data/processed_data/'.")

if __name__ == "__main__":
    main()
