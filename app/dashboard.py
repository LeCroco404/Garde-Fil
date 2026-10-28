"""Garde-Fil : tableau de bord du modérateur (Streamlit).

Le système signale et explique ; le modérateur décide et corrige.
Ses corrections alimentent la boucle de réapprentissage (scripts/retrain_with_feedback.py).

Lancement (depuis la racine du projet, venv actif) :
    streamlit run app/dashboard.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# Streamlit ajoute app/ au chemin d'import, pas la racine : on l'ajoute pour trouver `src`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR, TEXT_CLF_PATH, VIDEOS_DIR  # noqa: E402
from src.feedback import feedback_examples, latest_reviews_by_video, save_review  # noqa: E402
from src.fusion import fmt_ts  # noqa: E402

st.set_page_config(page_title="Garde-Fil", page_icon="🛡️", layout="wide")

LEVEL_ORDER = {"élevé": 0, "modéré": 1, "faible": 2}
LEVEL_BADGE = {"élevé": "🔴 ÉLEVÉ", "modéré": "🟠 MODÉRÉ", "faible": "🟢 FAIBLE"}
SOURCE_ICON = {"parole": "🗣️ parole", "ecran": "🖥️ écran"}
LABELS = ["Sans risque", "À risque"]
DECISIONS = ["Risque confirmé", "Pas de risque", "Escalade (cas complexe)"]


# ---------- Chargement des données ----------

def load_results() -> list[dict]:
    """Résultats d'analyse, triés du plus urgent au moins urgent."""
    results = []
    for f in PROCESSED_DIR.glob("*/result.json"):
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if "risk" in r:  # ignore les résultats au format des phases 1-2 (à réanalyser)
            results.append(r)
    return sorted(results, key=lambda r: (LEVEL_ORDER[r["risk"]["level"]], -r["risk"]["score"]))


@st.cache_resource
def get_model(model_mtime: float):
    """Charge le classifieur une seule fois. La date du fichier sert de clé de cache :
    après un réapprentissage, le nouveau modèle est rechargé automatiquement."""
    from src.text_classifier import TextRiskClassifier
    return TextRiskClassifier.load(TEXT_CLF_PATH)


def run_analysis(videos: list[Path]) -> None:
    from src.pipeline import analyze_video  # import tardif : charge Whisper seulement si besoin
    model = get_model(TEXT_CLF_PATH.stat().st_mtime)
    bar = st.progress(0.0, text="Analyse en cours…")
    for i, video in enumerate(videos):
        bar.progress(i / len(videos), text=f"Analyse de {video.name} ({i + 1}/{len(videos)})…")
        analyze_video(video, model)
    bar.progress(1.0, text="Analyse terminée")


# ---------- Barre latérale : file de revue ----------

with st.sidebar:
    st.title("🛡️ Garde-Fil")
    st.caption("Aide à la modération : le système signale et explique, l'humain décide.")
    moderator = st.text_input("Pseudonyme du modérateur", value="moderateur_1")

    results = load_results()
    analyzed = {r["video"] for r in results}
    all_videos = sorted(VIDEOS_DIR.glob("*.mp4"))
    new_videos = [v for v in all_videos if v.name not in analyzed]

    if not TEXT_CLF_PATH.exists():
        st.error("Aucun modèle : lance d'abord python scripts/train_text_classifier.py")
        st.stop()
    if new_videos and st.button(f"Analyser {len(new_videos)} nouvelle(s) vidéo(s)", type="primary"):
        run_analysis(new_videos)
        st.rerun()
    if all_videos and st.button("Réanalyser toutes les vidéos"):
        run_analysis(all_videos)  # utile après un réapprentissage
        st.rerun()

    # Audit : revoir aussi des vidéos NON signalées est le seul moyen de repérer les faux négatifs
    show_all = st.toggle("Inclure les vidéos non signalées (audit)", value=False)
    reviews = latest_reviews_by_video()
    queue = [r for r in results if show_all or r["risk"]["flagged"]]

    st.divider()
    fb_texts, _, disagreements = feedback_examples()
    c1, c2 = st.columns(2)
    c1.metric("Vidéos revues", len(reviews))
    c2.metric("Corrections", disagreements)
    st.caption(f"{len(fb_texts)} passages annotés. Réapprentissage : "
               "`python scripts/retrain_with_feedback.py`")

    if not queue:
        st.info("Aucune vidéo dans la file. Ajoute des .mp4 dans data/videos/ puis lance l'analyse.")
        st.stop()

    st.divider()
    idx = st.radio(
        "File de revue",
        range(len(queue)),
        format_func=lambda i: (f"{'✅' if queue[i]['video'] in reviews else '⏳'} "
                               f"{LEVEL_BADGE[queue[i]['risk']['level']]} "
                               f"{queue[i]['risk']['score']:.2f} · {queue[i]['video']}"),
    )

# ---------- Zone principale : une vidéo ----------

result = queue[idx]
risk = result["risk"]
video = result["video"]
previous = reviews.get(video)

if st.session_state.pop("saved", None) == video:
    st.success("Revue enregistrée.")

st.header(video)
col_video, col_risk = st.columns([1, 1.4])

with col_video:
    path = VIDEOS_DIR / video
    if path.exists():
        st.video(str(path), start_time=st.session_state.get(f"start_{video}", 0))
    else:
        st.warning("Fichier vidéo introuvable dans data/videos/.")
    # Boutons de saut : le modérateur va directement au passage signalé
    for i, m in enumerate(risk["moments"]):
        if st.button(f"▶ Aller à {fmt_ts(m['start'])}", key=f"jump_{video}_{i}"):
            st.session_state[f"start_{video}"] = int(m["start"])
            st.rerun()

with col_risk:
    st.subheader(f"{LEVEL_BADGE[risk['level']]} · score {risk['score']:.2f}")
    if risk["reasons"]:
        st.markdown("**Pourquoi ce signalement**")
        for reason in risk["reasons"]:
            st.text(f"• {reason}")
    else:
        st.caption("Aucun passage au-dessus du seuil d'alerte.")
    st.info(f"**Action recommandée** : {risk['action']}")
    if previous:
        st.caption(f"Déjà revue par {previous['moderator']} le {previous['reviewed_at'][:16].replace('T', ' ')} "
                   f"UTC : « {previous['decision']} ». Une nouvelle revue remplacera celle-ci.")

# ---------- Formulaire de revue ----------

st.divider()
st.subheader("Revue des passages")
st.caption("L'étiquette proposée est celle du modèle : corrige-la s'il s'est trompé. "
           "Ces corrections serviront au réapprentissage.")

prev_labels = {(s["source"], s["start"], s["text"]): s["human_label"]
               for s in previous["segments"]} if previous else {}

with st.form(key=f"form_{video}"):
    human_labels = []
    for i, seg in enumerate(result["segments"]):
        default = prev_labels.get((seg["source"], seg["start"], seg["text"]), int(seg["flagged"]))
        c_meta, c_text, c_label = st.columns([1.2, 4, 1.8])
        c_meta.markdown(f"`{fmt_ts(seg['start'])}-{fmt_ts(seg['end'])}`  \n"
                        f"{SOURCE_ICON[seg['source']]} · **{seg['score']:.2f}**")
        c_text.text(seg["text"])
        choice = c_label.radio("Étiquette", LABELS, index=default, horizontal=True,
                               key=f"label_{video}_{i}", label_visibility="collapsed")
        human_labels.append(LABELS.index(choice))

    # Pas de valeur par défaut : le modérateur doit choisir activement (limite le biais d'automatisation)
    decision = st.radio("Décision sur la vidéo", DECISIONS, index=None, horizontal=True)
    submitted = st.form_submit_button("Enregistrer la revue", type="primary")

if submitted:
    if decision is None:
        st.error("Choisis une décision avant d'enregistrer.")
    else:
        save_review({
            "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "moderator": moderator.strip() or "anonyme",
            "video": video,
            "decision": decision,
            "model_score": risk["score"],
            "model_level": risk["level"],
            "segments": [
                {"source": s["source"], "start": s["start"], "end": s["end"], "text": s["text"],
                 "model_score": s["score"], "model_flagged": s["flagged"], "human_label": label}
                for s, label in zip(result["segments"], human_labels)
            ],
        })
        st.session_state["saved"] = video
        st.rerun()
