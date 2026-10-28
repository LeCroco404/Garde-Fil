"""Garde-Fil : analyse d'image zero-shot avec CLIP (phase 5b, optionnelle).

CLIP projette images et textes dans le MÊME espace de vecteurs : une image est proche
des descriptions qui lui correspondent. « Zero-shot » = aucun entraînement spécifique :
on compare chaque image à une liste de descriptions (config.VISION_PROMPTS).

Score visuel d'une image = probabilité cumulée des descriptions « à risque »,
calculée par un softmax sur TOUTES les descriptions (risque + neutres).
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import VISION_ALERT_THRESHOLD, VISION_MODEL, VISION_PROMPTS
from src.ocr import select_frames_to_read

RISK_MASK = np.array([is_risk for _, _, is_risk in VISION_PROMPTS])


@lru_cache(maxsize=1)
def load_vision():
    """Charge CLIP une fois et pré-calcule les vecteurs des descriptions (identiques pour toutes les images)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(VISION_MODEL)
    prompts_emb = model.encode([p for p, _, _ in VISION_PROMPTS], normalize_embeddings=True)
    return model, prompts_emb


def score_images(paths: list[Path]) -> list[tuple[float, int]]:
    """Pour chaque image : (score de risque entre 0 et 1, indice de la description à risque la plus proche)."""
    model, prompts_emb = load_vision()
    images = [Image.open(p).convert("RGB") for p in paths]
    images_emb = model.encode(images, batch_size=16, normalize_embeddings=True)

    # Similarité cosinus (vecteurs normalisés) x 100 : l'échelle utilisée par CLIP lui-même
    logits = 100.0 * images_emb @ prompts_emb.T
    logits -= logits.max(axis=1, keepdims=True)          # stabilité numérique du softmax
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)

    risk = probs[:, RISK_MASK].sum(axis=1)
    risk_idx = np.where(RISK_MASK)[0]
    top_risk = risk_idx[probs[:, RISK_MASK].argmax(axis=1)]
    return [(float(r), int(i)) for r, i in zip(risk, top_risk)]


def analyze_frames(frames: list[tuple[float, Path]], fps: float) -> tuple[list[dict], list[dict], int]:
    """Analyse visuelle des images extraites.

    Retourne (segments signalés, score de chaque image, nombre d'images réellement analysées).
    Seules les images au-dessus de VISION_ALERT_THRESHOLD produisent un segment.
    """
    to_read = select_frames_to_read(frames)  # même logique que l'OCR : on saute les images identiques
    scored = dict(zip(to_read, score_images([frames[i][1] for i in to_read]))) if to_read else {}

    per_frame: list[dict] = []
    last = (0.0, None)
    for i, (ts, _path) in enumerate(frames):
        last = scored.get(i, last)
        per_frame.append({"t": ts, "score": round(last[0], 3),
                          "top": VISION_PROMPTS[last[1]][1] if last[1] is not None else None})

    # Regroupe les images consécutives signalées pour la même raison
    segments: list[dict] = []
    current = None
    for f in per_frame:
        if f["score"] < VISION_ALERT_THRESHOLD:
            current = None
            continue
        if current and current["text"].endswith(f["top"]):
            current["end"] = round(f["t"] + 1 / fps, 2)
            current["score"] = max(current["score"], f["score"])
        else:
            current = {"start": f["t"], "end": round(f["t"] + 1 / fps, 2),
                       "text": f"image évoquant : {f['top']}", "score": f["score"],
                       "flagged": True, "source": "image"}
            segments.append(current)
    return segments, per_frame, len(to_read)
