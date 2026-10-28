"""Garde-Fil : entraîne et évalue le classifieur de texte sur le sujet voisin (toxicité FR).

Usage (depuis la racine du projet, venv actif) :
    python scripts/train_text_classifier.py
"""
import json
import sys
from pathlib import Path

# Rendre le paquet `src` importable quand on lance ce script directement
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import classification_report  # noqa: E402

from src.config import (ALERT_THRESHOLD, DATASET_NAME, DATASET_SPLIT, EMB_CACHE_PATH,  # noqa: E402
                        TEXT_CLF_PATH, TEXT_METRICS_PATH)
from src.data import binary_metrics, load_base_split  # noqa: E402
from src.text_classifier import TextRiskClassifier  # noqa: E402


def main() -> None:
    # 1. Jeu public, découpé 80/20 de façon stratifiée et reproductible (voir src/data.py).
    #    Le jeu de test n'est JAMAIS vu à l'entraînement : c'est lui qui mesure la vraie performance.
    X_train, X_test, y_train, y_test = load_base_split()

    # 2. Entraînement. Les embeddings du jeu public sont mis en cache : le 1er lancement les
    #    calcule (quelques minutes), les suivants les relisent depuis le disque (quelques secondes).
    print("Calcul des embeddings (ou lecture du cache) et entraînement...")
    model = TextRiskClassifier()
    model.fit_embeddings(model.embed(X_train, cache_path=EMB_CACHE_PATH), y_train)

    # 3. Évaluation au seuil d'alerte
    scores = model.predict_proba_embeddings(model.embed(X_test, cache_path=EMB_CACHE_PATH))
    y_pred = (scores >= ALERT_THRESHOLD).astype(int)
    print("\n" + classification_report(y_test, y_pred, target_names=["non toxique", "toxique"], digits=3))
    metrics = binary_metrics(y_test, scores)
    c = metrics["confusion"]
    print(f"Matrice de confusion : VP={c['tp']}  FP={c['fp']}  FN={c['fn']}  VN={c['tn']}")

    # 4. Sauvegarde du modèle et des métriques (dans models/, non versionné)
    model.save(TEXT_CLF_PATH)
    TEXT_METRICS_PATH.write_text(json.dumps({
        "dataset": f"{DATASET_NAME}:{DATASET_SPLIT}", "n_train": len(X_train), "n_test": len(X_test),
        "feedback_examples": 0, **metrics,
    }, indent=2))
    print(f"\nModèle sauvegardé : {TEXT_CLF_PATH}")


if __name__ == "__main__":
    main()
