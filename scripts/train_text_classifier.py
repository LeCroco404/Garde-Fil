"""Garde-Fil : entraîne et évalue le classifieur de texte sur le sujet voisin (toxicité FR).

Usage (depuis la racine du projet, venv actif) :
    python scripts/train_text_classifier.py
"""
import json
import sys
from pathlib import Path

# Rendre le paquet `src` importable quand on lance ce script directement
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.config import (ALERT_THRESHOLD, DATASET_NAME, DATASET_SPLIT,  # noqa: E402
                        TEXT_CLF_PATH, TEXT_METRICS_PATH)
from src.text_classifier import TextRiskClassifier  # noqa: E402


def main() -> None:
    # 1. Chargement du jeu public (mis en cache dans ~/.cache/huggingface, hors du dépôt)
    print(f"Chargement de {DATASET_NAME} [{DATASET_SPLIT}]...")
    df = load_dataset(DATASET_NAME, split=DATASET_SPLIT).to_pandas()
    df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"])
    print(f"{len(df)} textes | répartition : {df['toxic'].value_counts().to_dict()}")

    # 2. Découpage 80/20 stratifié : même proportion de toxiques dans les deux parts.
    #    Le jeu de test n'est JAMAIS vu à l'entraînement : c'est lui qui mesure la vraie performance.
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(), df["toxic"].tolist(),
        test_size=0.2, stratify=df["toxic"], random_state=42,
    )

    # 3. Entraînement (le calcul des embeddings prend l'essentiel du temps)
    print("Calcul des embeddings et entraînement...")
    model = TextRiskClassifier().fit(X_train, y_train)

    # 4. Évaluation au seuil d'alerte
    y_pred = (model.predict_proba(X_test) >= ALERT_THRESHOLD).astype(int)
    print("\n" + classification_report(y_test, y_pred, target_names=["non toxique", "toxique"], digits=3))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"Matrice de confusion : VP={tp}  FP={fp}  FN={fn}  VN={tn}")

    # 5. Sauvegarde du modèle et des métriques (dans models/, non versionné)
    model.save(TEXT_CLF_PATH)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    TEXT_METRICS_PATH.write_text(json.dumps({
        "dataset": f"{DATASET_NAME}:{DATASET_SPLIT}", "n_train": len(X_train), "n_test": len(X_test),
        "threshold": ALERT_THRESHOLD, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
    }, indent=2))
    print(f"\nModèle sauvegardé : {TEXT_CLF_PATH}")


if __name__ == "__main__":
    main()
