"""Garde-Fil : journal des revues des modérateurs (boucle de réapprentissage).

Format JSONL : une ligne JSON par revue, ajoutée à la fin du fichier, jamais modifiée.
On garde ainsi l'historique complet (traçabilité des décisions humaines) ; en cas de
nouvelle revue d'une même vidéo, c'est la plus récente qui fait foi.

RGPD : le fichier contient des extraits de transcription -> il reste dans data/feedback/
(non versionné), le modérateur est identifié par un pseudonyme, et on ne stocke que
le strict nécessaire au réapprentissage (minimisation).
"""
import json

from src.config import REVIEWS_PATH


def save_review(record: dict) -> None:
    REVIEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REVIEWS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_reviews() -> list[dict]:
    if not REVIEWS_PATH.exists():
        return []
    with REVIEWS_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_reviews_by_video() -> dict[str, dict]:
    """{nom_vidéo: revue la plus récente}."""
    latest: dict[str, dict] = {}
    for review in load_reviews():  # le fichier est chronologique : la dernière écrase les précédentes
        latest[review["video"]] = review
    return latest


def feedback_examples() -> tuple[list[str], list[int], int]:
    """Exemples annotés par les modérateurs : (textes, étiquettes, nb de désaccords avec le modèle).

    Un même texte annoté plusieurs fois garde sa dernière étiquette.
    """
    labels: dict[str, int] = {}
    model_said: dict[str, int] = {}
    for review in latest_reviews_by_video().values():
        for seg in review["segments"]:
            # Seuls les passages TEXTE entraînent le classifieur de texte. Les étiquettes « image »
            # restent dans le journal : elles serviront à évaluer et ajuster les descriptions CLIP.
            if seg["source"] == "image":
                continue
            labels[seg["text"]] = seg["human_label"]
            model_said[seg["text"]] = int(seg["model_flagged"])
    disagreements = sum(labels[t] != model_said[t] for t in labels)
    return list(labels), list(labels.values()), disagreements
