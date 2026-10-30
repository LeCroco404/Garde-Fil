"""Garde-Fil : prépare le contenu de la démo publique (dossier demo/, versionné).

1. Copie une sélection de vidéos SYNTHÉTIQUES dans demo/videos/.
2. Les analyse avec le pipeline actuel et ne garde que le résultat (pas d'audio, pas d'images).
3. Exporte le classifieur de texte en JSON (coefficients) pour le champ « Tester une phrase ».
   Format texte plutôt que pickle : charger un pickle revient à exécuter du code,
   ce qu'une application publique doit éviter.

À lancer sur la machine qui a les modèles (venv actif) :
    python scripts/build_demo.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import ALERT_THRESHOLD, DATA_DIR, HIGH_THRESHOLD, MIN_WORDS_TO_FLAG, TEXT_CLF_PATH  # noqa: E402
from src.text_classifier import TextRiskClassifier  # noqa: E402

DEMO_DIR = ROOT / "demo"

# Sélection : (identifiant, vidéo source dans data/, titre affiché, ce qu'il faut remarquer)
# Uniquement des vidéos synthétiques (voix de synthèse, fond uni) : aucune photo, aucune personne réelle.
EXAMPLES = [
    ("cuisine", "videos/neutre.mp4", "Une recette de cuisine",
     "Un contenu ordinaire : rien n'est signalé, la vidéo ne serait même pas montrée au modérateur."),
    ("insulte", "videos/toxique.mp4", "Une insulte dite à l'oral",
     "Le message est repéré dans la parole. Même quand la transcription fait une petite erreur, "
     "le sens agressif de la phrase est reconnu."),
    ("ecran", "videos/ecran.mp4", "Un message écrit à l'écran, voix neutre",
     "La voix ne dit rien de grave : c'est le texte incrusté qui est repéré, à la seconde où il apparaît. "
     "Sans lecture de l'écran, cette vidéo passerait inaperçue."),
    ("convergence", "videos/convergence.mp4", "Un message dit ET écrit en même temps",
     "Les deux sources se confirment au même instant : le score monte et la vidéo passe en revue prioritaire."),
    ("prevention", "videos/prevention.mp4", "Une vidéo de prévention",
     "Elle parle de harcèlement, mais pour aider. Elle n'est pas signalée : c'est le cas le plus important "
     "du projet, car un système qui bloquerait la prévention ferait plus de mal que de bien."),
    ("temoignage", "eval/videos/p3.mp4", "Un témoignage : « on peut s'en sortir »",
     "Quelqu'un raconte ce qu'il a vécu pour encourager les autres. Ce témoignage ne doit pas être signalé, "
     "et il ne l'est pas."),
    ("critique", "eval/videos/h1.mp4", "Une critique de film (« nul »)",
     "Un mot négatif ne suffit pas : personne n'est visé. Le système juge le sens, pas les mots."),
    ("exclusion", "videos/exclusion.mp4", "Une exclusion formulée calmement (cas difficile)",
     "Sans insulte ni agressivité, le système a du mal à reconnaître ce type de harcèlement. "
     "C'est une limite connue, et c'est pour ça qu'un humain garde toujours la décision."),
]


def export_text_model() -> None:
    model = TextRiskClassifier.load(TEXT_CLF_PATH)
    coef = model.clf.coef_[0].astype(float)
    intercept = float(model.clf.intercept_[0])

    # Vérification : le calcul manuel doit redonner exactement predict_proba
    probe = ["Bonjour à tous", "Ferme-la, personne ne veut de toi ici"]
    manual = 1 / (1 + np.exp(-(model.embed(probe) @ coef + intercept)))
    assert np.allclose(manual, model.predict_proba(probe), atol=1e-6), "export du modèle incohérent"

    (DEMO_DIR / "text_model.json").write_text(json.dumps({
        "embedder_name": model.embedder_name,
        "coef": [round(c, 8) for c in coef.tolist()],
        "intercept": intercept,
        "alert_threshold": ALERT_THRESHOLD,
        "high_threshold": HIGH_THRESHOLD,
        "min_words_to_flag": MIN_WORDS_TO_FLAG,
    }))
    print("✅ Modèle de texte exporté : demo/text_model.json")


def main() -> None:
    from src.pipeline import analyze_video, warm_up  # import tardif : modèles lourds

    (DEMO_DIR / "videos").mkdir(parents=True, exist_ok=True)
    model = TextRiskClassifier.load(TEXT_CLF_PATH)
    print(f"Chargement des modèles : {warm_up(model, use_vision=False):.1f}s")

    examples = []
    with tempfile.TemporaryDirectory() as work:  # audio et images extraits : supprimés à la fin
        for ex_id, src, title, note in EXAMPLES:
            source = DATA_DIR / src
            if not source.exists():
                print(f"  ⏭  {ex_id} : {src} introuvable, exemple ignoré")
                continue
            target = DEMO_DIR / "videos" / f"{ex_id}.mp4"
            shutil.copy(source, target)
            # Vidéos synthétiques sans image réelle : l'analyse d'image n'apporte rien ici
            result = analyze_video(target, model, use_vision=False, out_dir=Path(work))
            for key in ("vision_frames", "n_vision"):
                result.pop(key, None)
            examples.append({"id": ex_id, "title": title, "note": note,
                             "video": f"videos/{ex_id}.mp4", "result": result})
            print(f"  ✅ {ex_id:12} {result['risk']['level']:7} score {result['risk']['score']:.2f}")

    (DEMO_DIR / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2))
    export_text_model()
    print(f"\n{len(examples)} exemples prêts dans demo/. Lancement : streamlit run app/demo.py")


if __name__ == "__main__":
    main()
