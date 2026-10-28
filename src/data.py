"""Garde-Fil : chargement du jeu de données de base et métriques communes.

Partagé par l'entraînement initial et le réapprentissage, pour qu'ils utilisent
EXACTEMENT le même découpage : sinon, comparer deux modèles n'aurait aucun sens.
"""
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from src.config import ALERT_THRESHOLD, DATASET_NAME, DATASET_SPLIT


def load_base_split(test_size: float = 0.2, seed: int = 42):
    """Retourne X_train, X_test, y_train, y_test (découpage stratifié et reproductible)."""
    df = load_dataset(DATASET_NAME, split=DATASET_SPLIT).to_pandas()
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"])
    print(f"{DATASET_NAME} [{DATASET_SPLIT}] : {len(df)} textes | "
          f"répartition : {df['toxic'].value_counts().to_dict()}")
    # random_state fixe : le même jeu de test à chaque exécution
    return train_test_split(
        df["text"].tolist(), df["toxic"].tolist(),
        test_size=test_size, stratify=df["toxic"], random_state=seed,
    )


def binary_metrics(y_true, scores, threshold: float = ALERT_THRESHOLD) -> dict:
    """Précision, rappel, F1 et matrice de confusion au seuil donné."""
    y_pred = [int(s >= threshold) for s in scores]
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold, "precision": round(float(p), 3), "recall": round(float(r), 3),
        "f1": round(float(f1), 3), "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }
