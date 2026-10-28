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
