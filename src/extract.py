"""Garde-Fil : extraction des médias d'une vidéo avec ffmpeg.

Audio (phase 1) pour Whisper, images (phase 2) pour l'OCR.
"""
import shutil
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


def extract_frames(video_path: Path, out_dir: Path, fps: float = 1.0) -> list[tuple[float, Path]]:
    """Extrait `fps` image(s) par seconde en PNG (sans perte, meilleur pour l'OCR).

    Retourne [(horodatage_en_secondes, chemin_image), ...] dans l'ordre chronologique.
    """
    frames_dir = out_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)  # évite de mélanger avec les images d'une analyse précédente
    frames_dir.mkdir(parents=True)

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(video_path),
         "-vf", f"fps={fps}",     # filtre fps : une image toutes les 1/fps secondes, à partir de t=0
         str(frames_dir / "frame_%04d.png")],
        check=True,
    )
    # L'image n°i (à partir de 0) correspond à l'instant i / fps
    return [(round(i / fps, 2), p) for i, p in enumerate(sorted(frames_dir.glob("frame_*.png")))]
