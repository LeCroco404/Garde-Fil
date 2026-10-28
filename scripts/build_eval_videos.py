"""Garde-Fil : génère les vidéos du jeu de test à partir de eval/cases.yaml.

Seules les vidéos manquantes sont générées (relancer le script est sans risque).
Les cas « image » sont ignorés si la photo n'existe pas dans data/images/.

Usage (depuis la racine du projet, venv actif) :
    python scripts/build_eval_videos.py
"""
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, EVAL_CASES_PATH, EVAL_DIR  # noqa: E402

VIDEOS_DIR = EVAL_DIR / "videos"


def main() -> None:
    cases = yaml.safe_load(EVAL_CASES_PATH.read_text(encoding="utf-8"))["cases"]
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    built = skipped = existing = 0

    for case in cases:
        out = VIDEOS_DIR / f"{case['id']}.mp4"
        if out.exists():
            existing += 1
            continue
        if "image" in case:
            image = DATA_DIR / "images" / case["image"]
            if not image.exists():
                print(f"  ⏭  {case['id']} : {image.name} absente de data/images/, cas ignoré")
                skipped += 1
                continue
            cmd = [str(ROOT / "scripts/make_image_video.sh"), str(image), str(out)]
        else:
            # Un espace évite un texte incrusté vide (refusé par ffmpeg) ; l'OCR l'ignore ensuite
            cmd = [str(ROOT / "scripts/make_test_video.sh"), case["speech"], case.get("overlay") or " ",
                   str(out), str(case.get("overlay_start", 0))]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"  ✅ {case['id']} ({case['category']})")
        built += 1

    print(f"\n{built} vidéo(s) générée(s), {existing} déjà présente(s), {skipped} ignorée(s) -> {VIDEOS_DIR}")


if __name__ == "__main__":
    main()
