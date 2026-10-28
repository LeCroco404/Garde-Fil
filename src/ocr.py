"""Garde-Fil : lecture du texte incrusté dans les images (OCR Tesseract).

Une même incrustation reste affichée plusieurs secondes : on fusionne les images
consécutives au texte similaire en un seul segment horodaté [début, fin], au même
format que les segments Whisper. Le classifieur les traite ensuite de la même façon.

Optimisations (phase 5a) :
- une image quasi identique à la dernière image lue n'est pas relue (on réutilise le résultat) ;
- les images à lire sont traitées en parallèle (plusieurs processus Tesseract à la fois).
"""
import difflib
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image, ImageOps

from src.config import (FRAME_CHANGE_PIXEL, FRAME_CHANGE_RATIO, OCR_LANG, OCR_MIN_CONF, OCR_PSM,
                        OCR_SIMILARITY, OCR_WORKERS)


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
    """Au moins un mot de 3 lettres ou plus : sinon, c'est presque toujours du bruit d'OCR
    (textures, cordes, logos lus comme « À à | BE À »)."""
    return re.search(r"[^\W\d_]{3,}", text) is not None


def _similar(a: str, b: str) -> bool:
    """Compare deux textes en ignorant casse et espaces (l'OCR varie un peu d'une image à l'autre)."""
    norm = lambda t: re.sub(r"\s+", " ", t.lower()).strip()  # noqa: E731
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio() >= OCR_SIMILARITY


def _thumbnail(image_path: Path) -> np.ndarray:
    """Miniature 72x128 en niveaux de gris : assez fine pour voir un texte apparaître, très rapide à comparer."""
    return np.asarray(ImageOps.grayscale(Image.open(image_path)).resize((72, 128)), dtype=np.int16)


def select_frames_to_read(frames: list[tuple[float, Path]]) -> list[int]:
    """Indices des images à passer à l'OCR : la première, puis chaque image qui diffère
    suffisamment de la dernière image retenue (voir FRAME_CHANGE_PIXEL / FRAME_CHANGE_RATIO)."""
    selected: list[int] = []
    ref = None
    for i, (_ts, path) in enumerate(frames):
        thumb = _thumbnail(path)
        # On compare à la dernière image LUE (pas à la précédente) : un fondu lent finit par être détecté
        if ref is None or (np.abs(thumb - ref) > FRAME_CHANGE_PIXEL).mean() > FRAME_CHANGE_RATIO:
            selected.append(i)
            ref = thumb
    return selected


def ocr_frames(frames: list[tuple[float, Path]], fps: float) -> tuple[list[dict], int]:
    """OCR des images + regroupement en segments [{"start", "end", "text", "ocr_conf"}].

    Retourne (segments, nombre d'images réellement lues par Tesseract).
    """
    to_read = select_frames_to_read(frames)
    # pytesseract lance un processus tesseract par image : des threads suffisent pour paralléliser
    with ThreadPoolExecutor(max_workers=OCR_WORKERS) as pool:
        readings = dict(zip(to_read, pool.map(ocr_frame, [frames[i][1] for i in to_read])))

    segments: list[dict] = []
    current = None
    last_reading = ("", 0.0)
    for i, (ts, _path) in enumerate(frames):
        # Image non relue = identique à la dernière lue : on réutilise sa lecture
        last_reading = readings.get(i, last_reading)
        text, conf = last_reading
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
    return segments, len(to_read)
