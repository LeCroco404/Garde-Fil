"""Garde-Fil : classifieur de texte = embeddings sémantiques + régression logistique.

Principe : un modèle pré-entraîné transforme chaque phrase en vecteur qui encode
son SENS (deux phrases proches en sens ont des vecteurs proches, même sans mot
commun). Un classifieur linéaire apprend ensuite à séparer les vecteurs « à risque »
des autres. Aucune liste de mots n'intervient.
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.config import EMBEDDING_MODEL


class TextRiskClassifier:
    def __init__(self, embedder_name: str = EMBEDDING_MODEL):
        self.embedder_name = embedder_name
        self._embedder = None  # chargé à la demande (lourd)
        # class_weight="balanced" : compense un éventuel déséquilibre entre classes
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    @property
    def embedder(self):
        """Charge le modèle d'embeddings au premier usage (téléchargé une fois, ~470 Mo)."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedder_name)
        return self._embedder

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.embedder.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,          # vecteurs de norme 1 : comparaison par le sens, pas la longueur
            show_progress_bar=len(texts) > 100,
        )

    def embed(self, texts: list[str], cache_path: Path | None = None) -> np.ndarray:
        """Transforme une liste de textes en matrice (n_textes x 384).

        cache_path (optionnel) : cache disque {texte: vecteur}. Un texte déjà encodé n'est plus
        recalculé. À réserver au jeu PUBLIC : on n'y met jamais de transcriptions (RGPD).
        """
        if cache_path is None:
            return self._encode(texts)
        cache = joblib.load(cache_path) if Path(cache_path).exists() else {}
        store = cache.setdefault(self.embedder_name, {})  # un cache par modèle d'embeddings
        missing = [t for t in dict.fromkeys(texts) if t not in store]  # dict.fromkeys : dédoublonne
        if missing:
            store.update(zip(missing, self._encode(missing)))
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(cache, cache_path)
        return np.vstack([store[t] for t in texts])

    # Variantes « à partir d'embeddings déjà calculés » : évitent de recalculer les mêmes vecteurs
    def fit_embeddings(self, X: np.ndarray, labels, sample_weight=None) -> "TextRiskClassifier":
        """sample_weight (optionnel) : importance de chaque exemple (ex. corrections des modérateurs)."""
        self.clf.fit(X, labels, sample_weight=sample_weight)
        return self

    def predict_proba_embeddings(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]

    def fit(self, texts: list[str], labels: list[int], sample_weight=None) -> "TextRiskClassifier":
        return self.fit_embeddings(self.embed(texts), labels, sample_weight)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Probabilité d'appartenir à la classe « à risque » (1) pour chaque texte."""
        if not texts:
            return np.array([])
        return self.predict_proba_embeddings(self.embed(texts))

    def save(self, path: Path) -> None:
        """Sauvegarde le classifieur + le nom du modèle d'embeddings (pas le modèle lui-même)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"embedder_name": self.embedder_name, "clf": self.clf}, path)

    @classmethod
    def load(cls, path: Path) -> "TextRiskClassifier":
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} introuvable : lance d'abord python scripts/train_text_classifier.py"
            )
        state = joblib.load(path)
        obj = cls(state["embedder_name"])
        obj.clf = state["clf"]
        return obj
