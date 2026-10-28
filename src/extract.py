"""Garde-Fil : extraction des médias d'une vidéo avec ffmpeg.

Phase 1 : piste audio uniquement. L'extraction d'images arrivera en phase 2.
"""
import subprocess
from pathlib import Path


def has_audio(video_path: Path) -> bool:
    """Indique si la vidéo contient au moins une piste audio (via ffprobe)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return "audio" in out.stdout


def extract_audio(video_path: Path, out_dir: Path) -> Path | None:
    """Extrait l'audio en WAV mono 16 kHz, le format attendu par Whisper.

    Retourne le chemin du WAV, ou None si la vidéo est muette.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not has_audio(video_path):
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(video_path),
         "-vn",                   # ignorer la vidéo
         "-ac", "1",              # mono
         "-ar", "16000",          # 16 kHz : fréquence native de Whisper
         "-c:a", "pcm_s16le",     # WAV non compressé
         str(wav_path)],
        check=True,
    )
    return wav_path
