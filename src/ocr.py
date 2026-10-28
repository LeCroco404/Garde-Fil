"""Garde-Fil : lecture du texte incrusté dans les images (OCR Tesseract).

Une même incrustation reste affichée plusieurs secondes : on fusionne les images
consécutives au texte similaire en un seul segment horodaté [début, fin], au même
format que les segments Whisper. Le classifieur les traite ensuite de la même façon.
"""
import difflib
import re
from pathlib import Path

import pytesseract
from PIL import Image, ImageOps

from src.config import OCR_LANG, OCR_MIN_CONF, OCR_PSM, OCR_SIMILARITY


def ocr_frame(image_path: Path) -> tuple[str, float]:
    """Lit le texte d'une image. Retourne (texte, confiance moyenne en %)."""
    # Prétraitement : niveaux de gris + contraste maximal -> lettres plus nettes pour Tesseract
    img = ImageOps.autocontrast(ImageOps.grayscale(Image.open(image_path)))

    # image_to_data donne chaque mot avec sa confiance et sa position (bloc, paragraphe, ligne)
    data = pytesseract.image_to_data(
        img, lang=OCR_LANG, config=f"--psm {OCR_PSM}", output_type=pytesseract.Output.DICT
    )

    lines: dict[tuple, list[str]] = {}
    confs: list[float] = []
    for i, word in enumerate(data["text"]):
        word, conf = word.strip(), float(data["conf"][i])
        if not word or conf < OCR_MIN_CONF:  # filtre le bruit (textures, logos, faux caractères)
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confs.append(conf)

    # Reconstitution ligne par ligne, dans l'ordre de lecture détecté par Tesseract
    text = " ".join(" ".join(words) for _, words in sorted(lines.items()))
    return text, (sum(confs) / len(confs) if confs else 0.0)


def _is_meaningful(text: str) -> bool:
    """Au moins 3 lettres : en dessous, c'est presque toujours du bruit d'OCR."""
    return len(re.findall(r"[^\W\d_]", text)) >= 3


def _similar(a: str, b: str) -> bool:
    """Compare deux textes en ignorant casse et espaces (l'OCR varie un peu d'une image à l'autre)."""
    norm = lambda t: re.sub(r"\s+", " ", t.lower()).strip()  # noqa: E731
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio() >= OCR_SIMILARITY


def ocr_frames(frames: list[tuple[float, Path]], fps: float) -> list[dict]:
    """OCR de toutes les images + regroupement en segments [{"start", "end", "text", "ocr_conf"}]."""
    segments: list[dict] = []
    current = None
    for ts, path in frames:
        text, conf = ocr_frame(path)
        if not _is_meaningful(text):
            current = None  # plus de texte à l'écran : l'incrustation en cours est terminée
            continue
        if current and _similar(current["text"], text):
            # Même incrustation : on prolonge le segment et on garde la lecture la plus fiable
            current["end"] = round(ts + 1 / fps, 2)
            if conf > current["ocr_conf"]:
                current["text"], current["ocr_conf"] = text, conf
        else:
            current = {"start": ts, "end": round(ts + 1 / fps, 2), "text": text, "ocr_conf": conf}
            segments.append(current)

    for s in segments:
        s["ocr_conf"] = round(s["ocr_conf"], 1)
    return segments
