"""Garde-Fil : pipeline complet (parole + écran -> fusion -> décision expliquée, chronométré).

vidéo -> audio (ffmpeg) -> segments parlés (Whisper)  ┐
      -> images 1/s (ffmpeg) -> segments écran (OCR)  ┴-> score par segment -> fusion -> rapport

Usage (depuis la racine du projet, venv actif) :
    python -m src.pipeline data/videos/test_01.mp4
"""
import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

from src.config import ALERT_THRESHOLD, FRAME_FPS, PROCESSED_DIR, TEXT_CLF_PATH, WHISPER_MODEL
from src.extract import extract_audio, extract_frames
from src.fusion import fmt_ts, fuse
from src.ocr import ocr_frames
from src.text_classifier import TextRiskClassifier
from src.transcribe import _load_model as load_whisper
from src.transcribe import transcribe

SOURCE_LABELS = {"parole": "parole", "ecran": "écran "}


def warm_up(model: TextRiskClassifier) -> float:
    """Charge les modèles lourds (embeddings ~470 Mo, Whisper ~500 Mo) une fois, AVANT les analyses.

    Sans cela, la 1re vidéo paie seule ces chargements et son chronométrage est faussé.
    Retourne la durée du chargement en secondes.
    """
    start = time.perf_counter()
    _ = model.embedder           # la propriété déclenche le chargement du modèle d'embeddings
    load_whisper(WHISPER_MODEL)  # mis en cache par lru_cache pour les appels suivants
    return round(time.perf_counter() - start, 2)


def analyze_video(video_path: Path, model: TextRiskClassifier) -> dict:
    video_path = Path(video_path)
    work_dir = PROCESSED_DIR / video_path.stem   # un dossier de travail par vidéo

    # Chronométrage de chaque étape : on optimise ce qu'on a mesuré, pas ce qu'on suppose
    timings: dict[str, float] = {}

    @contextmanager
    def timed(step: str):
        start = time.perf_counter()
        yield
        timings[step] = round(time.perf_counter() - start, 2)

    # 1. Parole : extraction audio + transcription
    with timed("audio"):
        audio = extract_audio(video_path, work_dir)
    with timed("transcription"):
        speech = transcribe(audio) if audio else []
    for s in speech:
        s["source"] = "parole"

    # 2. Texte à l'écran : extraction d'images + OCR
    with timed("images"):
        frames = extract_frames(video_path, work_dir, FRAME_FPS)
    with timed("ocr"):
        screen, n_ocr = ocr_frames(frames, FRAME_FPS)
    for s in screen:
        s["source"] = "ecran"

    # 3. Score de chaque segment, toutes sources confondues, dans l'ordre chronologique
    segments = sorted(speech + screen, key=lambda s: (s["start"], s["source"]))
    with timed("classification"):
        scores = model.predict_proba([s["text"] for s in segments])
    for seg, score in zip(segments, scores):
        seg["score"] = round(float(score), 3)
        seg["flagged"] = bool(score >= ALERT_THRESHOLD)

    # 4. Fusion -> score global, niveau, raisons, action recommandée
    with timed("fusion"):
        risk = fuse(segments)
    timings["total"] = round(sum(timings.values()), 2)
    result = {
        "video": video_path.name,
        "has_audio": audio is not None,
        "n_frames": len(frames),
        "n_ocr": n_ocr,
        "risk": risk,
        "segments": segments,
        "timings": timings,
    }

    # 5. Sauvegarde : transcription et OCR sont des données personnelles -> data/ (non versionné)
    (work_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def print_report(result: dict) -> None:
    risk = result["risk"]
    print(f"\n=== {result['video']} ===")
    if not result["has_audio"]:
        print("(aucune piste audio)")
    if not result["segments"]:
        print("Aucun texte détecté (ni parole, ni écran).")

    # Détail des segments (vue technique)
    for s in result["segments"]:
        mark = "⚠️ " if s["flagged"] else "   "
        print(f"{mark}[{fmt_ts(s['start'])}-{fmt_ts(s['end'])}] {SOURCE_LABELS[s['source']]} "
              f"{s['score']:.2f}  {s['text']}")

    # Décision expliquée (vue modérateur)
    print(f"\nScore global : {risk['score']:.2f} | niveau : {risk['level'].upper()}")
    if risk["reasons"]:
        print("Pourquoi :")
        for reason in risk["reasons"]:
            print(f"  - {reason}")
    print(f"Action recommandée : {risk['action']}")
    # Le système signale et explique ; seul un humain décide
    print("Décision finale : modérateur humain." if risk["flagged"] else "Non transmise au modérateur.")

    # Performance (vue technique)
    t = result["timings"]
    steps = " | ".join(f"{k} {v:.2f}s" for k, v in t.items() if k != "total")
    print(f"Temps : {steps} | TOTAL {t['total']:.2f}s "
          f"(OCR sur {result['n_ocr']}/{result['n_frames']} images)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Garde-Fil : analyse et score de risque expliqué")
    parser.add_argument("videos", nargs="+", type=Path)
    args = parser.parse_args()

    model = TextRiskClassifier.load(TEXT_CLF_PATH)  # chargé une fois pour toutes les vidéos
    print(f"Chargement des modèles : {warm_up(model):.2f}s (une seule fois, hors chronométrage par vidéo)")
    for video in args.videos:
        print_report(analyze_video(video, model))


if __name__ == "__main__":
    main()
