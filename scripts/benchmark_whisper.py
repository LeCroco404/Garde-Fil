"""Garde-Fil : compare plusieurs tailles de modèles Whisper (vitesse ET qualité).

Pour chaque modèle : temps de chargement, temps de transcription, RTF et texte produit,
pour juger à l'œil le compromis vitesse / justesse.

RTF (real-time factor) = temps de traitement / durée de l'audio.
RTF 0.5 = 2x plus rapide que le temps réel ; RTF > 1 = plus lent que la vidéo elle-même.

Usage (depuis la racine du projet, venv actif) :
    python scripts/benchmark_whisper.py data/videos/*.mp4
    python scripts/benchmark_whisper.py --models base small medium data/videos/toxique.mp4
"""
import argparse
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR  # noqa: E402
from src.extract import extract_audio  # noqa: E402
from src.transcribe import _load_model, transcribe  # noqa: E402


def wav_duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--models", nargs="+", default=["base", "small"],
                        help="tailles à comparer (medium : ~1,5 Go à télécharger, ~2 Go de RAM)")
    args = parser.parse_args()

    # Audio extrait une seule fois, réutilisé pour tous les modèles
    audios = []
    for video in args.videos:
        wav = extract_audio(video, PROCESSED_DIR / video.stem)
        if wav:
            audios.append((video.name, wav, wav_duration(wav)))
    total_audio = sum(d for _, _, d in audios)

    summary = []
    for name in args.models:
        print(f"\n===== Whisper {name} =====")
        t0 = time.perf_counter()
        _load_model(name)  # chargement séparé : il n'a lieu qu'une fois par session
        load_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        for video_name, wav, _ in audios:
            text = " ".join(s["text"] for s in transcribe(wav, model_name=name))
            print(f"  {video_name:22} {text}")
        run_s = time.perf_counter() - t0
        summary.append((name, load_s, run_s, run_s / total_audio))
        _load_model.cache_clear()  # libère la mémoire avant le modèle suivant (8 Go de RAM)

    print(f"\n{'modèle':8}{'chargement':>12}{'transcription':>15}{'RTF':>7}   ({total_audio:.1f} s d'audio)")
    for name, load_s, run_s, rtf in summary:
        print(f"{name:8}{load_s:>11.1f}s{run_s:>14.1f}s{rtf:>7.2f}")


if __name__ == "__main__":
    main()
