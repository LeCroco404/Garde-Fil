"""Garde-Fil : fusion des scores en un score global + explication pour le modérateur.

Principe : des règles simples et lisibles plutôt qu'une « boîte noire ».
Le modérateur doit pouvoir comprendre POURQUOI une vidéo lui est présentée.

1. Moments : les passages signalés (parole, écran ou image) proches dans le temps sont regroupés.
2. Score d'un moment : le pire passage, + bonus si plusieurs sources (parole, écran, image) convergent.
3. Score global : le pire moment, + bonus si plusieurs moments distincts (récurrence).
4. Niveau (faible / modéré / élevé) -> action recommandée + raisons horodatées.
5. Si seule l'image (non calibrée) signale un risque, le niveau est plafonné (IMAGE_ONLY_MAX_LEVEL).

Bonus PROPORTIONNELS (phase 6a) : un bonus comble une part de la marge restante jusqu'à 1,
au lieu d'ajouter une valeur fixe. 0,94 -> 0,97 au lieu de saturer à 1,00 : les cas graves
restent classables entre eux, et le score ne dépasse jamais 1 par construction.
"""
import re

from src.config import (ACTIONS, ALERT_THRESHOLD, CONVERGENCE_SHARE, HIGH_THRESHOLD,
                        IMAGE_ONLY_MAX_LEVEL, MIN_WORDS_TO_FLAG, MOMENT_GAP, RECURRENCE_MAX_SHARE,
                        RECURRENCE_SHARE)

LEVELS = ["faible", "modéré", "élevé"]  # du moins au plus urgent
SOURCE_ORDER = ["parole", "ecran", "image"]
SOURCE_WORDS = {"parole": "dit", "ecran": "affiché à l'écran", "image": "visible à l'image"}
SOURCE_NAMES = {"parole": "parole", "ecran": "écran", "image": "image"}


def fmt_ts(seconds: float) -> str:
    """12.4 -> '0:12' (format lisible par le modérateur)."""
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def _excerpt(text: str, max_len: int = 80) -> str:
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def text_alert(text: str, score: float, alert_threshold: float = ALERT_THRESHOLD) -> bool:
    """Un passage TEXTE est signalé si son score atteint le seuil ET s'il compte au moins
    MIN_WORDS_TO_FLAG mots (un fragment trop court n'est pas jugeable sans contexte)."""
    return score >= alert_threshold and len(re.findall(r"\w+", text)) >= MIN_WORDS_TO_FLAG


def boost(score: float, share: float) -> float:
    """Comble `share` de la marge restante : boost(0.94, 0.5) = 0.97 ; boost(0.6, 0.5) = 0.8."""
    return score + share * (1.0 - score)


def group_moments(flagged: list[dict]) -> list[list[dict]]:
    """Regroupe les segments signalés qui se chevauchent ou sont à moins de MOMENT_GAP s."""
    moments: list[list[dict]] = []
    for seg in sorted(flagged, key=lambda s: s["start"]):
        if moments and seg["start"] <= max(s["end"] for s in moments[-1]) + MOMENT_GAP:
            moments[-1].append(seg)
        else:
            moments.append([seg])
    return moments


def score_moment(segs: list[dict], convergence_share: float = CONVERGENCE_SHARE) -> dict:
    """Score d'un moment = pire passage (+ bonus proportionnel si plusieurs sources convergent)."""
    top = max(segs, key=lambda s: s["score"])
    sources = sorted({s["source"] for s in segs}, key=SOURCE_ORDER.index)
    converging = len(sources) > 1
    score = boost(top["score"], convergence_share) if converging else top["score"]
    return {
        "start": min(s["start"] for s in segs),
        "end": max(s["end"] for s in segs),
        "score": round(score, 3),
        "bonus": round(score - top["score"], 3),
        "sources": sources,
        "converging": converging,
        "top": top,
    }


def level_of(score: float, alert_threshold: float = ALERT_THRESHOLD,
             high_threshold: float = HIGH_THRESHOLD) -> str:
    if score < alert_threshold:  # testé en premier : reste vrai même si le seuil d'alerte dépasse HIGH_THRESHOLD
        return "faible"
    return "élevé" if score >= high_threshold else "modéré"


def _describe(moment: dict) -> str:
    """Une raison lisible : quand, où (parole / écran / image), quel score, quel extrait."""
    where = " ET ".join(SOURCE_WORDS[s] for s in moment["sources"])
    line = (f"{fmt_ts(moment['start'])}-{fmt_ts(moment['end'])} : passage {where} "
            f"(score {moment['score']:.2f}) « {_excerpt(moment['top']['text'])} »")
    if moment["converging"]:
        names = " + ".join(SOURCE_NAMES[s] for s in moment["sources"])
        line += f" [convergence {names} : +{moment['bonus']:.2f}]"
    return line


def fuse(segments: list[dict], alert_threshold: float = ALERT_THRESHOLD,
         high_threshold: float = HIGH_THRESHOLD, convergence_share: float = CONVERGENCE_SHARE) -> dict:
    """Produit la décision finale à partir des segments scorés (parole, écran, image).

    Les paramètres ont pour valeur par défaut ceux de config.py ; l'évaluation (phase 6a)
    les fait varier pour calibrer le système sans relancer toute l'analyse des vidéos.
    """
    flagged = [s for s in segments if s["flagged"]]
    moments = [score_moment(m, convergence_share) for m in group_moments(flagged)]

    if moments:
        worst = max(moments, key=lambda m: m["score"])
        share = min(RECURRENCE_MAX_SHARE, RECURRENCE_SHARE * (len(moments) - 1))
        score = boost(worst["score"], share)
        reasons = [_describe(m) for m in sorted(moments, key=lambda m: -m["score"])]
        if share:
            reasons.append(f"{len(moments)} moments distincts signalés dans la vidéo "
                           f"[récurrence : +{score - worst['score']:.2f}]")
    else:
        # Aucun passage signalé : on garde le meilleur score pour information, sans raison d'alerte.
        # Ce score peut dépasser le seuil (ex. texte trop court) : le niveau reste donc FAIBLE.
        score = max((s["score"] for s in segments), default=0.0)
        reasons = []

    # Seul un passage signalé peut faire monter le niveau (sinon la règle des 3 mots serait contournée)
    level = level_of(score, alert_threshold, high_threshold) if moments else "faible"

    # Plafond « image seule » : le score est conservé (transparence), seul le niveau est limité
    image_only = bool(moments) and all(m["sources"] == ["image"] for m in moments)
    if image_only and LEVELS.index(level) > LEVELS.index(IMAGE_ONLY_MAX_LEVEL):
        level = IMAGE_ONLY_MAX_LEVEL
        reasons.append(f"Niveau plafonné à {level.upper()} : seule l'image signale un risque "
                       "(source non calibrée, aucune parole ni texte à l'écran ne le confirme)")

    return {
        "score": round(score, 3),
        "level": level,
        "flagged": level != "faible",
        "first_alert": fmt_ts(min(m["start"] for m in moments)) if moments else None,
        "reasons": reasons,
        "action": ACTIONS[level],
        "moments": moments,
    }
