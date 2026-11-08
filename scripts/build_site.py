"""Garde-Fil : prépare les données de la démo statique (dossier site/, publié sur GitHub Pages).

Les pages (site/*.html), les styles (site/css/), les scripts (site/js/) et Transformers.js (site/vendor/)
sont écrits à la main et versionnés. Ce script génère à côté :
    site/data/examples.json     exemples pré-analysés (copiés de demo/, champs utiles seulement)
    site/data/text_model.json   coefficients du modèle de texte
    site/data/reference.json    scores de référence calculés ici en pleine précision, pour vérifier
                                que le modèle compressé exécuté dans le navigateur donne les mêmes résultats
    site/videos/*.mp4           vidéos synthétiques
    site/videos/*.vtt           sous-titres (texte d'origine, pas la transcription automatique)

À lancer après scripts/build_demo.py (venv actif) :
    python scripts/build_site.py
"""
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEMO = ROOT / "demo"
SITE = ROOT / "site"
NBSP = " "

# Phrases supplémentaires pour la vérification (en plus des passages des exemples)
EXTRA_PHRASES = [
    "J'adore ce film, il est vraiment réussi.",
    "Personne ne veut de toi ici, dégage.",
    "Bonjour à tous, bienvenue sur ma chaîne.",
    "Tu es vraiment nul, ferme-la.",
    "On se retrouve demain pour réviser ensemble ?",
]

# Texte d'origine des vidéos synthétiques (voix et incrustation passées à scripts/make_test_video.sh).
# Sert aux sous-titres et à la description : la transcription automatique contient des erreurs
# (« C'était » au lieu de « T'es »), qui restent visibles dans le détail de l'analyse.
# id : (texte dit, texte affiché à l'écran, seconde d'apparition du texte)
MEDIA = {
    "cuisine": ("Aujourd'hui je vous montre ma recette de crêpes préférée, c'est vraiment simple.",
                "Recette du jour", 0),
    "insulte": ("T'es vraiment un gros nul, ferme-la et dégage de cette vidéo.", "Test toxique", 0),
    "ecran": ("Voici un petit résumé de ma journée, rien de spécial.",
              "Personne ne te supporte ici dégage, t'es de trop", 2),
    "convergence": ("T'es vraiment un gros nul, ferme-la.", "Personne ne veut de toi ici", 0),
    "prevention": ("Le harcèlement en ligne, ça suffit. Si tu en es victime, parles-en à un adulte "
                   "ou appelle le 3018.", "Prévention", 0),
    "temoignage": ("Quand j'étais au collège, on se moquait de moi. Aujourd'hui je vais bien et je veux "
                   "vous dire qu'on peut s'en sortir.", "Mon témoignage", 0),
    "critique": ("Ce film était vraiment nul, je ne vous le conseille pas.", "Critique ciné", 0),
    "exclusion": ("On a tous décidé de ne plus te parler, tu n'es plus invité nulle part.", "Bonne soirée", 0),
}

SEGMENT_KEYS = ("start", "end", "source", "score", "flagged", "text")
RISK_KEYS = ("level", "score", "reasons", "action")


def typo(texte: str) -> str:
    """Typographie française : espaces insécables avant : ; ! ? et à l'intérieur des « »."""
    texte = re.sub(r" ([:;!?»])", NBSP + r"\1", texte)
    return texte.replace("« ", "«" + NBSP)


def vtt_time(seconds: float) -> str:
    return f"00:{int(seconds) // 60:02d}:{seconds % 60:06.3f}"


def write_captions(ex_id: str, segments: list[dict]) -> str | None:
    """Un sous-titre couvrant la parole détectée, avec le texte d'origine."""
    speech = [s for s in segments if s["source"] == "parole"]
    if ex_id not in MEDIA or not speech:
        return None
    start, end = min(s["start"] for s in speech), max(s["end"] for s in speech)
    (SITE / "videos" / f"{ex_id}.vtt").write_text(
        f"WEBVTT\n\n{vtt_time(start)} --> {vtt_time(end)}\n{typo(MEDIA[ex_id][0])}\n", encoding="utf-8")
    return f"videos/{ex_id}.vtt"


def description(ex_id: str) -> str | None:
    """Équivalent textuel du contenu visuel (accessibilité) : fond, voix, texte incrusté."""
    if ex_id not in MEDIA:
        return None
    _, overlay, overlay_start = MEDIA[ex_id]
    when = f" à partir de 0:{overlay_start:02d}" if overlay_start else ""
    return typo(f"Vidéo synthétique : fond bleu uni, voix de synthèse. Texte affiché à l'écran{when} : « {overlay} ».")


def main() -> None:
    if not (DEMO / "examples.json").exists():
        sys.exit("demo/ absent : lance d'abord python scripts/build_demo.py")

    (SITE / "data").mkdir(parents=True, exist_ok=True)
    if (SITE / "videos").exists():
        shutil.rmtree(SITE / "videos")
    shutil.copytree(DEMO / "videos", SITE / "videos")

    # Exemples : on ne garde que ce que la page affiche (pas de chronométrage, pas de détails internes)
    examples = json.loads((DEMO / "examples.json").read_text(encoding="utf-8"))
    slim = []
    for ex in examples:
        segments = [{k: s[k] for k in SEGMENT_KEYS} for s in ex["result"]["segments"]]
        slim.append({
            "id": ex["id"], "title": ex["title"], "note": ex["note"], "video": ex["video"],
            "captions": write_captions(ex["id"], segments), "description": description(ex["id"]),
            "risk": {k: ex["result"]["risk"][k] for k in RISK_KEYS},
            "segments": segments,
        })
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

    n_captions = sum(1 for ex in slim if ex["captions"])
    print(f"✅ Données de la démo prêtes : {len(slim)} exemples, {n_captions} sous-titres, "
          f"{len(phrases)} phrases de référence")
    print("Test local : python -m http.server 8000 -d site  puis http://localhost:8000")


if __name__ == "__main__":
    main()
