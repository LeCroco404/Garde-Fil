"""Garde-Fil : démo publique en lecture seule.

Contrairement au tableau de bord (app/dashboard.py), cette application :
- n'accepte AUCUNE vidéo envoyée par les visiteurs (ni données personnelles, ni contenus dangereux) ;
- montre des vidéos synthétiques analysées à l'avance (demo/examples.json, créé par scripts/build_demo.py) ;
- permet de tester une phrase, sans rien enregistrer ;
- ne charge que le petit modèle de texte (pas de Whisper, Tesseract ni CLIP).

Lancement (depuis la racine du projet) :
    streamlit run app/demo.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fusion import fmt_ts  # noqa: E402  (module léger : aucune dépendance lourde)

DEMO_DIR = ROOT / "demo"
REPO_URL = "https://github.com/LeCroco404/Garde-Fil"
MAX_CHARS = 300

LEVEL_DISPLAY = {
    "élevé": ("🔴", "Risque élevé", "La vidéo serait montrée en priorité à un modérateur humain."),
    "modéré": ("🟠", "Risque modéré", "La vidéo serait montrée à un modérateur humain."),
    "faible": ("🟢", "Risque faible", "La vidéo ne serait pas montrée au modérateur."),
}
# Test de phrase : libellés prudents. Un score bas n'est jamais présenté comme une absence de danger.
PHRASE_DISPLAY = {
    "élevé": ("🔴", "Harcèlement probable",
              "Une vidéo contenant cette phrase serait montrée en priorité à un modérateur humain."),
    "modéré": ("🟠", "Harcèlement possible", "Une vidéo contenant cette phrase serait montrée à un modérateur humain."),
    "faible": ("", "Score bas", "Le modèle n'a pas reconnu de harcèlement dans cette phrase. Cela ne veut pas dire "
               "qu'elle est sans danger : ce prototype ne reconnaît pas de façon fiable les menaces, les incitations "
               "à se faire du mal ou les formulations indirectes."),
}
SOURCE_DISPLAY = {"parole": "🗣️ dit", "ecran": "🖥️ écrit à l'écran", "image": "🖼️ image"}

st.set_page_config(page_title="Garde-Fil, démo", page_icon="🛡️", layout="wide")


# ---------- Chargement ----------

@st.cache_data
def load_examples() -> list[dict]:
    return json.loads((DEMO_DIR / "examples.json").read_text(encoding="utf-8"))


@st.cache_resource
def load_text_model():
    """Modèle de texte = modèle d'embeddings public + coefficients exportés en JSON (pas de pickle)."""
    from sentence_transformers import SentenceTransformer
    params = json.loads((DEMO_DIR / "text_model.json").read_text())
    embedder = SentenceTransformer(params["embedder_name"])
    return embedder, np.array(params["coef"]), params


def score_sentence(text: str) -> tuple[float, str, bool]:
    """Score d'une phrase, niveau, et si elle est trop courte pour alerter seule. Rien n'est enregistré."""
    embedder, coef, p = load_text_model()
    emb = embedder.encode([text], normalize_embeddings=True)[0]   # même normalisation qu'à l'entraînement
    score = float(1 / (1 + np.exp(-(emb @ coef + p["intercept"]))))  # régression logistique
    too_short = len(re.findall(r"\w+", text)) < p["min_words_to_flag"]
    if score < p["alert_threshold"] or too_short:
        level = "faible"
    else:
        level = "élevé" if score >= p["high_threshold"] else "modéré"
    return score, level, too_short


def level_banner(level: str, score: float, display: dict = LEVEL_DISPLAY) -> None:
    icon, label, meaning = display[level]
    st.subheader(f"{icon} {label} · score {score:.2f}".strip())
    st.caption(meaning)


# ---------- En-tête ----------

st.title("🛡️ Garde-Fil, démo")
st.markdown(
    "Garde-Fil est un prototype d'**aide à la modération** de vidéos courtes. Il écoute ce qui est dit, "
    "lit ce qui est écrit à l'écran, puis indique à un modérateur humain **si une vidéo mérite d'être revue, "
    "et pourquoi**. Il ne supprime jamais rien : c'est toujours une personne qui décide."
)
st.info(
    "Cette démo porte sur le **harcèlement en ligne**. Toutes les vidéos sont **fabriquées pour l'exemple** "
    "(voix de synthèse, fond uni) : aucune personne réelle. Rien de ce que vous faites ici n'est enregistré."
)

if not (DEMO_DIR / "examples.json").exists():
    st.error("Contenu de démo absent : lancer d'abord `python scripts/build_demo.py`.")
    st.stop()

tab_videos, tab_phrase, tab_how = st.tabs(["🎬 Vidéos d'exemple", "✍️ Tester une phrase", "⚙️ Comment ça marche"])

# ---------- Onglet 1 : vidéos analysées à l'avance ----------

with tab_videos:
    examples = load_examples()
    choice = st.selectbox("Choisissez un exemple", range(len(examples)),
                          format_func=lambda i: examples[i]["title"])
    ex = examples[choice]
    risk = ex["result"]["risk"]

    col_video, col_result = st.columns([1, 1.6])
    with col_video:
        st.video(str(DEMO_DIR / ex["video"]))
    with col_result:
        level_banner(risk["level"], risk["score"])
        st.markdown(f"**À remarquer :** {ex['note']}")
        if risk["reasons"]:
            st.markdown("**Ce que le système explique au modérateur :**")
            for reason in risk["reasons"]:
                st.text(f"• {reason}")
        st.markdown(f"**Action recommandée :** {risk['action']}")

        with st.expander("Détail : ce que le système a entendu et lu"):
            for seg in ex["result"]["segments"]:
                st.markdown(f"`{fmt_ts(seg['start'])}` {SOURCE_DISPLAY[seg['source']]} · score "
                            f"**{seg['score']:.2f}**{' ⚠️' if seg['flagged'] else ''}  \n{seg['text']}")
            st.caption("Score : de 0 (rien d'inquiétant) à 1 (très probablement du harcèlement). "
                       "Un passage est signalé à partir de 0,50.")

        with st.expander("Ce que verrait le modérateur pour décider"):
            st.radio("Décision sur la vidéo", ["Risque confirmé", "Pas de risque", "Escalade (cas complexe)"],
                     index=None, horizontal=True, disabled=True, key=f"decision_{ex['id']}")
            st.caption("Dans l'outil réel, le modérateur choisit lui-même (aucune réponse n'est proposée par défaut) "
                       "et peut corriger le système. Désactivé dans la démo.")

# ---------- Onglet 2 : tester une phrase ----------

with tab_phrase:
    st.warning(
        "**Prototype limité.** Ce modèle a appris à partir d'environ 3 800 phrases, surtout des insultes. "
        "Il ne reconnaît pas de façon fiable les menaces, les incitations à se faire du mal ni les formulations "
        "indirectes. **Un score bas ne veut pas dire qu'un message est sans danger.**"
    )
    st.markdown(
        "Écrivez une phrase comme si elle était dite dans une vidéo. Le système donne un score de 0 à 1. "
        "**N'écrivez aucune information personnelle** (nom, adresse…). Rien n'est enregistré."
    )
    text = st.text_area("Votre phrase", max_chars=MAX_CHARS, height=100,
                        placeholder="Exemple : Personne ne veut de toi ici, dégage.")
    if st.button("Analyser", type="primary"):
        if not text.strip():
            st.warning("Écrivez une phrase d'abord.")
        else:
            with st.spinner("Analyse… (la première fois, le modèle se charge : quelques secondes)"):
                score, level, too_short = score_sentence(text.strip())
            level_banner(level, score, PHRASE_DISPLAY)
            if too_short:
                st.caption("Phrase de moins de 3 mots : trop courte pour déclencher une alerte à elle seule, "
                           "car sans contexte le système se trompe facilement.")
            st.caption("Si cette phrase vous concerne, ou concerne un proche : 3114 (gratuit, 24 h/24, 7 j/7). "
                       "En cas de danger immédiat : 15 ou 112.")
    st.caption("Une phrase isolée, sans le reste de la vidéo, peut être mal jugée : c'est pour ça qu'un humain décide.")

# ---------- Onglet 3 : explication ----------

with tab_how:
    st.markdown(
        "1. **Écouter et lire.** Le système transcrit la parole et lit le texte affiché à l'écran.\n"
        "2. **Comprendre le sens.** Chaque phrase est transformée en une représentation de son sens, "
        "puis comparée à des milliers d'exemples. Il n'y a pas de liste de mots interdits : "
        "« ce film est nul » n'est pas du harcèlement, une vidéo de prévention non plus.\n"
        "3. **Expliquer.** Le système indique quoi, où et quand : « passage dit et affiché à l'écran à 0:02 ».\n"
        "4. **Laisser décider un humain.** Le modérateur confirme ou corrige. Ses corrections "
        "aident le système à s'améliorer, avec des garde-fous.\n"
        "5. **Orienter vers l'aide.** Quand un risque est confirmé, l'action proposée est d'orienter la personne "
        "vers une aide (ici le 3018, numéro contre le harcèlement), plutôt que de simplement bloquer."
    )
    st.markdown(
        "**Ses limites.** Ce prototype a été entraîné sur un petit jeu de phrases, surtout des insultes. "
        "Il manque d'entraînement : il repère mal le harcèlement sans insulte, les menaces et les incitations "
        "à se faire du mal. Pour un vrai déploiement, il faudrait des données annotées par des professionnels, "
        "dans un cadre encadré. C'est aussi pour ça qu'il ne décide jamais seul."
    )
    st.markdown(f"Code, méthode, résultats et limites : [dépôt GitHub du projet]({REPO_URL}).")

# ---------- Pied de page ----------

st.divider()
st.caption(
    "Besoin d'aide ? Harcèlement en ligne : 3018 (gratuit). Idées suicidaires, pour soi ou un proche : "
    "3114 (gratuit, 24 h/24, 7 j/7). En cas d'urgence : 15 ou 112."
)
