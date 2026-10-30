"""Garde-Fil : prépare les données de la démo statique (dossier site/, publié sur GitHub Pages).

site/index.html est écrit à la main (versionné). Ce script génère à côté :
    site/data/examples.json     exemples pré-analysés (copiés de demo/, champs utiles seulement)
    site/data/text_model.json   coefficients du modèle de texte
    site/data/reference.json    scores de référence calculés ici en pleine précision, pour vérifier
                                que le modèle compressé exécuté dans le navigateur donne les mêmes résultats
    site/videos/*.mp4           vidéos synthétiques

À lancer après scripts/build_demo.py (venv actif) :
    python scripts/build_site.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEMO = ROOT / "demo"
SITE = ROOT / "site"

# Phrases supplémentaires pour la vérification (en plus des passages des exemples)
EXTRA_PHRASES = [
    "J'adore ce film, il est vraiment réussi.",
    "Personne ne veut de toi ici, dégage.",
    "Bonjour à tous, bienvenue sur ma chaîne.",
    "Tu es vraiment nul, ferme-la.",
    "On se retrouve demain pour réviser ensemble ?",
]

SEGMENT_KEYS = ("start", "end", "source", "score", "flagged", "text")
RISK_KEYS = ("level", "score", "reasons", "action")


def main() -> None:
    if not (DEMO / "examples.json").exists():
        sys.exit("demo/ absent : lance d'abord python scripts/build_demo.py")

    (SITE / "data").mkdir(parents=True, exist_ok=True)
    if (SITE / "videos").exists():
        shutil.rmtree(SITE / "videos")
    shutil.copytree(DEMO / "videos", SITE / "videos")

    # Exemples : on ne garde que ce que la page affiche (pas de chronométrage, pas de détails internes)
    examples = json.loads((DEMO / "examples.json").read_text(encoding="utf-8"))
    slim = [{
        "id": ex["id"], "title": ex["title"], "note": ex["note"], "video": ex["video"],
        "risk": {k: ex["result"]["risk"][k] for k in RISK_KEYS},
        "segments": [{k: s[k] for k in SEGMENT_KEYS} for s in ex["result"]["segments"]],
    } for ex in examples]
    (SITE / "data" / "examples.json").write_text(json.dumps(slim, ensure_ascii=False, indent=1), encoding="utf-8")

    params = json.loads((DEMO / "text_model.json").read_text())
    shutil.copy(DEMO / "text_model.json", SITE / "data" / "text_model.json")

    # Scores de référence en pleine précision (fp32), avec exactement le calcul de la page
    from sentence_transformers import SentenceTransformer
    phrases = list(dict.fromkeys([s["text"] for ex in slim for s in ex["segments"]] + EXTRA_PHRASES))
    emb = SentenceTransformer(params["embedder_name"]).encode(phrases, normalize_embeddings=True)
    scores = 1 / (1 + np.exp(-(emb @ np.array(params["coef"]) + params["intercept"])))
    (SITE / "data" / "reference.json").write_text(json.dumps(
        [{"text": t, "score": round(float(s), 4)} for t, s in zip(phrases, scores)],
        ensure_ascii=False, indent=1), encoding="utf-8")

    size = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file())
    print(f"✅ site/ prêt : {len(slim)} exemples, {len(phrases)} phrases de référence, {size / 1024:.0f} Ko")
    print("Test local : python -m http.server 8000 -d site  puis http://localhost:8000")


if __name__ == "__main__":
    main()
