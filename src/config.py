"""Garde-Fil : configuration centrale (chemins, modèles, seuils).

Tout réglage partagé entre modules est ici, pour ne jamais le dupliquer.
"""
from pathlib import Path

# --- Chemins (calculés depuis ce fichier : le projet marche quel que soit le dossier courant) ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"         # vidéos à analyser
PROCESSED_DIR = DATA_DIR / "processed"   # audio extrait + résultats d'analyse (jamais versionnés)
FEEDBACK_DIR = DATA_DIR / "feedback"     # revues des modérateurs (jamais versionnées)
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

# Score à partir duquel un passage est signalé au modérateur.
# Phase 6a : 0,5 conservé. Monter le seuil retirait des fausses alertes au prix d'un vrai cas manqué ;
# on a préféré corriger la cause des fausses alertes (textes trop courts, voir ci-dessous).
ALERT_THRESHOLD = 0.5

# Un texte de moins de 3 mots ne peut pas déclencher une alerte à lui seul : sans contexte,
# le classifieur (entraîné sur des phrases) répond de façon peu fiable (ex. « Chiffres clés » : 0,61).
MIN_WORDS_TO_FLAG = 3

# --- Images & OCR (texte incrusté) ---
FRAME_FPS = 1          # 1 image par seconde : suffit pour un texte qui reste affiché ~1 s ou plus
OCR_LANG = "fra"       # pack de langue Tesseract
OCR_PSM = 11           # mode « texte épars » : adapté aux incrustations dispersées sur l'image
OCR_MIN_CONF = 60      # on ignore les mots reconnus avec une confiance < 60 % (bruit, faux texte)
OCR_SIMILARITY = 0.8   # deux images consécutives au texte similaire à 80 % = même incrustation

# --- Fusion & décision (phase 3, bonus rendus proportionnels en phase 6a) ---
# Un bonus comble une PART de la marge restante jusqu'à 1 (0,94 -> 0,97) : plus de saturation à 1,00.
MOMENT_GAP = 1.0             # deux passages signalés à moins d'1 s d'écart = un même « moment »
CONVERGENCE_SHARE = 0.5      # plusieurs sources signalent le même moment : comble 50 % de la marge
RECURRENCE_SHARE = 0.15      # par moment signalé supplémentaire dans la vidéo...
RECURRENCE_MAX_SHARE = 0.30  # ...plafonné à 30 % de la marge
HIGH_THRESHOLD = 0.75        # au-delà : revue prioritaire

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

# --- Revue humaine & réapprentissage (phase 4) ---
REVIEWS_PATH = FEEDBACK_DIR / "reviews.jsonl"   # journal des revues : une ligne JSON par revue
FEEDBACK_WEIGHT = 3.0      # un exemple corrigé par un modérateur pèse 3x un exemple du jeu public
MAX_F1_DROP = 0.01         # garde-fou : on refuse un nouveau modèle qui perd plus de 0.01 de F1
TEXT_CLF_BACKUP_PATH = MODELS_DIR / "text_clf.prev.joblib"

# --- Performance (phase 5a) ---
OCR_WORKERS = 4              # images lues en parallèle par Tesseract (ton Mac a 8 cœurs)
# Une image est relue si plus de 0,05 % de ses pixels ont changé de plus de 30 niveaux (sur 255).
# Un texte qui apparaît change peu de pixels mais fortement ; le bruit de compression, l'inverse.
FRAME_CHANGE_PIXEL = 30
FRAME_CHANGE_RATIO = 0.0005
EMB_CACHE_PATH = MODELS_DIR / "embeddings_cache.joblib"  # cache des embeddings du jeu PUBLIC uniquement

# --- Analyse d'image (phase 5b, optionnelle) ---
ENABLE_VISION = True               # désactivable : python -m src.pipeline --no-vision ...
VISION_MODEL = "clip-ViT-B-32"     # CLIP via sentence-transformers (~600 Mo)
VISION_ALERT_THRESHOLD = 0.6       # plus strict que le texte : le zero-shot n'est pas calibré
# Une source non calibrée ne doit pas déclencher seule une revue prioritaire :
# si SEULE l'image signale un risque (aucune parole ni texte à l'écran), le niveau est plafonné.
IMAGE_ONLY_MAX_LEVEL = "modéré"

# Descriptions comparées à chaque image (zero-shot) : (texte pour CLIP, libellé pour l'explication, à risque ?)
# CLIP a été entraîné en anglais : les descriptions sont en anglais, les libellés en français.
# Sujet voisin : agressivité visuelle, décrite de façon générique. Les descriptions NEUTRES comptent
# autant que les autres : le score est relatif (une image est « à risque » par rapport aux alternatives).
# Pour le vrai sujet, ces descriptions seraient définies avec des cliniciens, pas rédigées ici.
VISION_PROMPTS = [
    ("a photo of people fighting", "une bagarre", True),
    ("a photo of a person making a threatening gesture", "un geste menaçant", True),
    ("a photo of a person shouting aggressively", "une personne qui crie avec agressivité", True),
    ("a photo of food", "de la nourriture", False),
    ("a photo of a person talking calmly to the camera", "une personne qui parle calmement", False),
    ("a photo of people smiling together", "des personnes souriantes", False),
    ("a photo of a landscape", "un paysage", False),
    ("a photo of an animal", "un animal", False),
    ("a plain colored background with text", "un fond uni avec du texte", False),
    ("a screenshot of a video game", "une capture de jeu vidéo", False),
]

# --- Évaluation (phase 6a) ---
EVAL_CASES_PATH = ROOT / "eval" / "cases.yaml"   # jeu de test synthétique, versionné (aucune donnée réelle)
EVAL_DIR = DATA_DIR / "eval"                     # vidéos générées + résultats (non versionnés)
EVAL_REPORT_PATH = ROOT / "docs" / "evaluation.md"
