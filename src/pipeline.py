"""Garde-Fil : pipeline phase 1 (texte seul).

vidéo -> audio (ffmpeg) -> segments horodatés (Whisper) -> score par segment (classifieur)

Usage (depuis la racine du projet, venv actif) :
    python -m src.pipeline data/videos/test_01.mp4
"""
import argparse
import json
from pathlib import Path

from src.config import ALERT_THRESHOLD, PROCESSED_DIR, TEXT_CLF_PATH
from src.extract import extract_audio
from src.text_classifier import TextRiskClassifier
from src.transcribe import transcribe


def fmt_ts(seconds: float) -> str:
    """12.4 -> '0:12' (format lisible par le modérateur)."""
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def analyze_video(video_path: Path, model: TextRiskClassifier) -> dict:
    video_path = Path(video_path)
    work_dir = PROCESSED_DIR / video_path.stem   # un dossier de travail par vidéo

    # 1. Extraction + transcription
    audio = extract_audio(video_path, work_dir)
    segments = transcribe(audio) if audio else []

    # 2. Score de chaque segment
    scores = model.predict_proba([s["text"] for s in segments])
    for seg, score in zip(segments, scores):
        seg["score"] = round(float(score), 3)
        seg["flagged"] = bool(score >= ALERT_THRESHOLD)

    # 3. Agrégation par le MAXIMUM : un seul passage préoccupant suffit à justifier
    #    une revue humaine. Une moyenne « diluerait » ce passage dans le reste de la vidéo.
    top = max(segments, key=lambda s: s["score"], default=None)
    result = {
        "video": video_path.name,
        "has_audio": audio is not None,
        "text_score": top["score"] if top else 0.0,
        "flagged": bool(top and top["flagged"]),
        "top_segment": top,
        "segments": segments,
    }

    # 4. Sauvegarde : la transcription est une donnée personnelle -> elle reste dans data/ (non versionné)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def print_report(result: dict) -> None:
    print(f"\n=== {result['video']} ===")
    if not result["has_audio"]:
        print("Aucune piste audio.")
    elif not result["segments"]:
        print("Aucune parole détectée.")
    for s in result["segments"]:
        mark = "⚠️ " if s["flagged"] else "   "
        print(f"{mark}[{fmt_ts(s['start'])}-{fmt_ts(s['end'])}] {s['score']:.2f}  {s['text']}")
    verdict = "À REVOIR par un modérateur" if result["flagged"] else "Rien à signaler"
    print(f"\nScore texte : {result['text_score']:.2f} -> {verdict}")
    if result["flagged"]:
        print(f"Passage détecté à {fmt_ts(result['top_segment']['start'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Garde-Fil : analyse texte d'une ou plusieurs vidéos")
    parser.add_argument("videos", nargs="+", type=Path)
    args = parser.parse_args()

    model = TextRiskClassifier.load(TEXT_CLF_PATH)  # chargé une fois pour toutes les vidéos
    for video in args.videos:
        print_report(analyze_video(video, model))


if __name__ == "__main__":
    main()
