"""Garde-Fil : réentraîne le classifieur avec les revues des modérateurs.

Boucle de réapprentissage :
  jeu public (entraînement) + exemples annotés par les modérateurs (pondérés)
  -> nouveau modèle -> comparé à l'ancien sur le MÊME jeu de test
  -> adopté seulement s'il ne régresse pas (garde-fou), l'ancien est sauvegardé.

Usage (depuis la racine du projet, venv actif) :
    python scripts/retrain_with_feedback.py
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (ALERT_THRESHOLD, DATASET_NAME, DATASET_SPLIT, FEEDBACK_WEIGHT, MAX_F1_DROP,  # noqa: E402
                        TEXT_CLF_BACKUP_PATH, TEXT_CLF_PATH, TEXT_METRICS_PATH)
from src.data import binary_metrics, load_base_split  # noqa: E402
from src.feedback import feedback_examples  # noqa: E402
from src.text_classifier import TextRiskClassifier  # noqa: E402


def main() -> None:
    # 1. Exemples annotés par les modérateurs
    fb_texts, fb_labels, disagreements = feedback_examples()
    if not fb_texts:
        print("Aucune revue enregistrée : rien à apprendre. Fais des revues dans le tableau de bord.")
        return
    print(f"{len(fb_texts)} exemples annotés, dont {disagreements} correction(s) (désaccords avec le modèle).")

    # 2. Même découpage que l'entraînement initial
    X_train, X_test, y_train, y_test = load_base_split()

    # Fuite de données : un exemple du jeu de test ne doit jamais servir à l'entraînement
    test_set = set(X_test)
    fb = [(t, y) for t, y in zip(fb_texts, fb_labels) if t not in test_set]
    fb_texts, fb_labels = [t for t, _ in fb], [y for _, y in fb]

    # 3. Ancien modèle : score de référence
    old = TextRiskClassifier.load(TEXT_CLF_PATH)
    m_old = binary_metrics(y_test, old.predict_proba(X_test))

    # 4. Nouveau modèle : jeu public + corrections, pondérées plus fort (ce sont des cas difficiles et réels)
    print("Réentraînement...")
    weights = [1.0] * len(X_train) + [FEEDBACK_WEIGHT] * len(fb_texts)
    new = TextRiskClassifier().fit(X_train + fb_texts, y_train + fb_labels, sample_weight=weights)
    m_new = binary_metrics(y_test, new.predict_proba(X_test))

    # 5. Les corrections ont-elles été assimilées ? (contrôle de cohérence, pas une évaluation)
    if fb_texts:
        old_ok = sum(int(s >= ALERT_THRESHOLD) == y for s, y in zip(old.predict_proba(fb_texts), fb_labels))
        new_ok = sum(int(s >= ALERT_THRESHOLD) == y for s, y in zip(new.predict_proba(fb_texts), fb_labels))
        print(f"Exemples des modérateurs correctement classés : {old_ok}/{len(fb_texts)} -> {new_ok}/{len(fb_texts)}")

    # 6. Comparaison sur le jeu de test commun
    print(f"\n{'':10}{'ancien':>8}{'nouveau':>9}")
    for k in ("precision", "recall", "f1"):
        print(f"{k:10}{m_old[k]:>8.3f}{m_new[k]:>9.3f}")

    # 7. Garde-fou de non-régression
    if m_new["f1"] < m_old["f1"] - MAX_F1_DROP:
        print(f"\n❌ Modèle refusé : F1 en baisse de plus de {MAX_F1_DROP}. L'ancien modèle est conservé.")
        return
    shutil.copy(TEXT_CLF_PATH, TEXT_CLF_BACKUP_PATH)  # retour arrière possible
    new.save(TEXT_CLF_PATH)
    TEXT_METRICS_PATH.write_text(json.dumps({
        "dataset": f"{DATASET_NAME}:{DATASET_SPLIT}", "n_train": len(X_train), "n_test": len(X_test),
        "feedback_examples": len(fb_texts), **m_new,
    }, indent=2))
    print(f"\n✅ Nouveau modèle adopté : {TEXT_CLF_PATH}")
    print(f"   Ancien sauvegardé : {TEXT_CLF_BACKUP_PATH}")
    print("   Mets à jour les scores : bouton « Réanalyser toutes les vidéos » du tableau de bord.")


if __name__ == "__main__":
    main()
