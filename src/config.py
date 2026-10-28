"""Garde-Fil : configuration centrale (chemins, modèles, seuils).

Tout réglage partagé entre modules est ici, pour ne jamais le dupliquer.
"""
from pathlib import Path

# --- Chemins (calculés depuis ce fichier : le projet marche quel que soit le dossier courant) ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"   # audio extrait + résultats d'analyse (jamais versionnés)
MODELS_DIR = ROOT / "models"             # classifieurs entraînés (jamais versionnés)

# --- Transcription (faster-whisper) ---
WHISPER_MODEL = "small"        # bon compromis qualité/vitesse sur CPU avec 8 Go de RAM
WHISPER_COMPUTE_TYPE = "int8"  # quantification : ~2x moins de mémoire, perte de qualité négligeable
LANGUAGE = "fr"                # forcer la langue évite les erreurs de détection sur des clips courts

# --- Classification du texte ---
# Modèle d'embeddings multilingue compact (~120 M paramètres, vecteurs de 384 dimensions)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEXT_CLF_PATH = MODELS_DIR / "text_clf.joblib"
TEXT_METRICS_PATH = MODELS_DIR / "text_metrics.json"

# Sujet voisin pour la phase de développement : toxicité en français (licence openrail++)
DATASET_NAME = "textdetox/multilingual_toxicity_dataset"
DATASET_SPLIT = "fr"

# Score à partir duquel un passage est signalé au modérateur (affiné en phase 6)
ALERT_THRESHOLD = 0.5

# --- Images & OCR (texte incrusté) ---
FRAME_FPS = 1          # 1 image par seconde : suffit pour un texte qui reste affiché ~1 s ou plus
OCR_LANG = "fra"       # pack de langue Tesseract
OCR_PSM = 11           # mode « texte épars » : adapté aux incrustations dispersées sur l'image
OCR_MIN_CONF = 60      # on ignore les mots reconnus avec une confiance < 60 % (bruit, faux texte)
OCR_SIMILARITY = 0.8   # deux images consécutives au texte similaire à 80 % = même incrustation

# --- Fusion & décision (phase 3) ---
# Valeurs fixées à la main pour l'instant (règles explicables) ; calibrées en phase 6.
MOMENT_GAP = 1.0           # deux passages signalés à moins d'1 s d'écart = un même « moment »
CONVERGENCE_BONUS = 0.10   # le même moment est signalé dans la parole ET à l'écran
RECURRENCE_BONUS = 0.05    # par moment signalé supplémentaire dans la vidéo...
RECURRENCE_MAX = 0.10      # ...plafonné à +0.10
HIGH_THRESHOLD = 0.75      # au-delà : revue prioritaire

# Ressource d'aide vers laquelle orienter (sujet voisin : cyberharcèlement).
# Lors de la transposition au vrai sujet, ce sera le 3114 (prévention du suicide).
HELP_RESOURCE = "3018 (cyberharcèlement)"

# Action recommandée au modérateur selon le niveau. Le système ne supprime jamais rien.
ACTIONS = {
    "élevé": "Revue prioritaire. Si le risque est confirmé, privilégier un message d'orientation "
             f"vers le {HELP_RESOURCE} plutôt qu'un simple blocage.",
    "modéré": f"Revue standard. Si le risque est confirmé, proposer les ressources d'aide : {HELP_RESOURCE}.",
    "faible": "Aucune action recommandée.",
}
