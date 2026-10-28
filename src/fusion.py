"""Garde-Fil : fusion des scores en un score global + explication pour le modérateur.

Principe : des règles simples et lisibles plutôt qu'une « boîte noire ».
Le modérateur doit pouvoir comprendre POURQUOI une vidéo lui est présentée.

1. Moments : les passages signalés (parole, écran ou image) proches dans le temps sont regroupés.
2. Score d'un moment : le pire passage, + bonus si plusieurs sources (parole, écran, image) convergent.
3. Score global : le pire moment, + bonus si plusieurs moments distincts (récurrence).
4. Niveau (faible / modéré / élevé) -> action recommandée + raisons horodatées.
5. Si seule l'image (non calibrée) signale un risque, le niveau est plafonné (IMAGE_ONLY_MAX_LEVEL).
"""
from src.config import (ACTIONS, ALERT_THRESHOLD, CONVERGENCE_BONUS, HIGH_THRESHOLD,
                        IMAGE_ONLY_MAX_LEVEL, MOMENT_GAP, RECURRENCE_BONUS, RECURRENCE_MAX)

LEVELS = ["faible", "modéré", "élevé"]  # du moins au plus urgent


def fmt_ts(seconds: float) -> str:
    """12.4 -> '0:12' (format lisible par le modérateur)."""
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def _excerpt(text: str, max_len: int = 80) -> str:
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def group_moments(flagged: list[dict]) -> list[list[dict]]:
    """Regroupe les segments signalés qui se chevauchent ou sont à moins de MOMENT_GAP s."""
    moments: list[list[dict]] = []
    for seg in sorted(flagged, key=lambda s: s["start"]):
        if moments and seg["start"] <= max(s["end"] for s in moments[-1]) + MOMENT_GAP:
            moments[-1].append(seg)
        else:
            moments.append([seg])
    return moments


def score_moment(segs: list[dict]) -> dict:
    """Score d'un moment = pire passage (+ bonus si plusieurs sources convergent)."""
    top = max(segs, key=lambda s: s["score"])
    sources = sorted({s["source"] for s in segs})
    converging = len(sources) > 1
    score = min(1.0, top["score"] + (CONVERGENCE_BONUS if converging else 0.0))
    return {
        "start": min(s["start"] for s in segs),
        "end": max(s["end"] for s in segs),
        "score": round(score, 3),
        "sources": sources,
        "converging": converging,
        "top": top,
    }


def level_of(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "élevé"
    return "modéré" if score >= ALERT_THRESHOLD else "faible"


SOURCE_ORDER = ["parole", "ecran", "image"]
SOURCE_WORDS = {"parole": "dit", "ecran": "affiché à l'écran", "image": "visible à l'image"}
SOURCE_NAMES = {"parole": "parole", "ecran": "écran", "image": "image"}


def _describe(moment: dict) -> str:
    """Une raison lisible : quand, où (parole / écran / image), quel score, quel extrait."""
    sources = sorted(moment["sources"], key=SOURCE_ORDER.index)
    where = " ET ".join(SOURCE_WORDS[s] for s in sources)
    line = (f"{fmt_ts(moment['start'])}-{fmt_ts(moment['end'])} : passage {where} "
            f"(score {moment['score']:.2f}) « {_excerpt(moment['top']['text'])} »")
    if moment["converging"]:
        names = " + ".join(SOURCE_NAMES[s] for s in sources)
        line += f" [convergence {names} : +{CONVERGENCE_BONUS:.2f}]"
    return line


def fuse(segments: list[dict]) -> dict:
    """Produit la décision finale à partir des segments scorés (parole, écran, image)."""
    flagged = [s for s in segments if s["flagged"]]
    moments = [score_moment(m) for m in group_moments(flagged)]

    if moments:
        worst = max(moments, key=lambda m: m["score"])
        recurrence = min(RECURRENCE_MAX, RECURRENCE_BONUS * (len(moments) - 1))
        score = min(1.0, worst["score"] + recurrence)
        reasons = [_describe(m) for m in sorted(moments, key=lambda m: -m["score"])]
        if recurrence:
            reasons.append(f"{len(moments)} moments distincts signalés dans la vidéo "
                           f"[récurrence : +{recurrence:.2f}]")
    else:
        # Rien au-dessus du seuil : on garde le meilleur score pour information, sans raison d'alerte
        score = max((s["score"] for s in segments), default=0.0)
        reasons = []

    level = level_of(score)

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
