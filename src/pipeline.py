"""Garde-Fil : pipeline phase 2 (parole + texte à l'écran).

vidéo -> audio (ffmpeg) -> segments parlés (Whisper)  ┐
      -> images 1/s (ffmpeg) -> segments écran (OCR)  ┴-> score par segment (classifieur)

Usage (depuis la racine du projet, venv actif) :
    python -m src.pipeline data/videos/test_01.mp4
"""
import argparse
import json
from pathlib import Path

from src.config import ALERT_THRESHOLD, FRAME_FPS, PROCESSED_DIR, TEXT_CLF_PATH
from src.extract import extract_audio, extract_frames
from src.ocr import ocr_frames
from src.text_classifier import TextRiskClassifier
from src.transcribe import transcribe

SOURCE_LABELS = {"parole": "parole", "ecran": "écran "}


def fmt_ts(seconds: float) -> str:
    """12.4 -> '0:12' (format lisible par le modérateur)."""
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def analyze_video(video_path: Path, model: TextRiskClassifier) -> dict:
    video_path = Path(video_path)
    work_dir = PROCESSED_DIR / video_path.stem   # un dossier de travail par vidéo

    # 1. Parole : extraction audio + transcription
    audio = extract_audio(video_path, work_dir)
    speech = transcribe(audio) if audio else []
    for s in speech:
        s["source"] = "parole"

    # 2. Texte à l'écran : extraction d'images + OCR
    frames = extract_frames(video_path, work_dir, FRAME_FPS)
    screen = ocr_frames(frames, FRAME_FPS)
    for s in screen:
        s["source"] = "ecran"

    # 3. Les deux sources ont le même format : on les classe ensemble, dans l'ordre chronologique
    segments = sorted(speech + screen, key=lambda s: (s["start"], s["source"]))
    scores = model.predict_proba([s["text"] for s in segments])
    for seg, score in zip(segments, scores):
        seg["score"] = round(float(score), 3)
        seg["flagged"] = bool(score >= ALERT_THRESHOLD)

    # 4. Agrégation par le MAXIMUM (la fusion pondérée arrivera en phase 3)
    top = max(segments, key=lambda s: s["score"], default=None)
    result = {
        "video": video_path.name,
        "has_audio": audio is not None,
        "n_frames": len(frames),
        "text_score": top["score"] if top else 0.0,
        "flagged": bool(top and top["flagged"]),
        "top_segment": top,
        "segments": segments,
    }

    # 5. Sauvegarde : transcription et OCR sont des données personnelles -> data/ (non versionné)
    (work_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def print_report(result: dict) -> None:
    print(f"\n=== {result['video']} ({result['n_frames']} images analysées) ===")
    if not result["has_audio"]:
        print("(aucune piste audio)")
    if not result["segments"]:
        print("Aucun texte détecté (ni parole, ni écran).")
    for s in result["segments"]:
        mark = "⚠️ " if s["flagged"] else "   "
        print(f"{mark}[{fmt_ts(s['start'])}-{fmt_ts(s['end'])}] {SOURCE_LABELS[s['source']]} "
              f"{s['score']:.2f}  {s['text']}")
    verdict = "À REVOIR par un modérateur" if result["flagged"] else "Rien à signaler"
    print(f"\nScore texte : {result['text_score']:.2f} -> {verdict}")
    if result["flagged"]:
        top = result["top_segment"]
        where = "dit" if top["source"] == "parole" else "affiché à l'écran"
        print(f"Passage {where} à {fmt_ts(top['start'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Garde-Fil : analyse parole + texte à l'écran")
    parser.add_argument("videos", nargs="+", type=Path)
    args = parser.parse_args()

    model = TextRiskClassifier.load(TEXT_CLF_PATH)  # chargé une fois pour toutes les vidéos
    for video in args.videos:
        print_report(analyze_video(video, model))


if __name__ == "__main__":
    main()
