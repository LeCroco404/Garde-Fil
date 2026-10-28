"""Garde-Fil : transcription de la parole avec faster-whisper.

Whisper découpe la parole en segments horodatés : ce sont eux qu'on classera,
ce qui permet ensuite de dire « passage détecté à 0:12 ».
"""
from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel

from src.config import LANGUAGE, WHISPER_COMPUTE_TYPE, WHISPER_MODEL


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> WhisperModel:
    """Charge un modèle une seule fois (le 1er appel le télécharge : base ~150 Mo, small ~500 Mo)."""
    # faster-whisper ne gère pas le GPU Apple (MPS) : le CPU en int8 est le bon choix sur Mac.
    return WhisperModel(model_name, device="cpu", compute_type=WHISPER_COMPUTE_TYPE)


def transcribe(audio_path: Path, model_name: str = WHISPER_MODEL) -> list[dict]:
    """Retourne la liste des segments : [{"start", "end", "text"}, ...] (temps en secondes)."""
    segments, _info = _load_model(model_name).transcribe(
        str(audio_path),
        language=LANGUAGE,
        vad_filter=True,   # VAD : ignore les silences et la musique seule, limite les « hallucinations »
        beam_size=5,
    )
    # `segments` est un générateur : la transcription s'exécute pendant cette boucle
    return [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments
        if s.text.strip()
    ]
