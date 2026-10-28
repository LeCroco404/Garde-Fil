"""Garde-Fil : évaluation de bout en bout sur le jeu de test vidéo (eval/cases.yaml).

1. Analyse chaque vidéo du jeu de test UNE fois (résultats mis en cache dans data/eval/processed/).
2. Re-score les passages texte et refait la fusion sans relancer Whisper/OCR/CLIP : on peut donc
   tester beaucoup de réglages en quelques secondes.
3. Produit :
   A. les performances avec les réglages actuels (par catégorie + erreurs détaillées) ;
   B. un balayage du seuil d'alerte (calibrage) ;
   C. la comparaison modèle actuel / modèle précédent (généralisation des corrections du modérateur) ;
   D. un rapport docs/evaluation.md (versionnable : le jeu de test est synthétique).

Usage (depuis la racine du projet, venv actif) :
    python scripts/evaluate.py              # utilise le cache d'analyse s'il existe
    python scripts/evaluate.py --refresh    # réanalyse toutes les vidéos
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (ALERT_THRESHOLD, CONVERGENCE_SHARE, ENABLE_VISION, EVAL_CASES_PATH,  # noqa: E402
                        EVAL_DIR, EVAL_REPORT_PATH, TEXT_CLF_BACKUP_PATH, TEXT_CLF_PATH)
from src.fusion import fuse, text_alert  # noqa: E402
from src.text_classifier import TextRiskClassifier  # noqa: E402

VIDEOS_DIR = EVAL_DIR / "videos"
RESULTS_DIR = EVAL_DIR / "processed"
SWEEP = [round(0.30 + 0.05 * i, 2) for i in range(11)]  # 0.30 -> 0.80


# ---------- 1. Analyse (coûteuse, mise en cache) ----------

def analyze_all(cases: list[dict], refresh: bool, use_vision: bool) -> dict[str, dict]:
    results, todo = {}, []
    for case in cases:
        cached = RESULTS_DIR / case["id"] / "result.json"
        if cached.exists() and not refresh:
            results[case["id"]] = json.loads(cached.read_text())
        else:
            todo.append(case)
    if todo:
        from src.pipeline import analyze_video, warm_up  # import tardif : modèles lourds
        model = TextRiskClassifier.load(TEXT_CLF_PATH)
        print(f"Chargement des modèles : {warm_up(model, use_vision):.1f}s")
        for i, case in enumerate(todo, 1):
            print(f"  analyse {i}/{len(todo)} : {case['id']}", flush=True)
            results[case["id"]] = analyze_video(VIDEOS_DIR / f"{case['id']}.mp4", model,
                                                use_vision, out_dir=RESULTS_DIR)
    return results


# ---------- 2. Décision (rapide, rejouable) ----------

def decide(result: dict, text_scores: dict[str, float], alert: float, share: float,
           use_vision: bool = True) -> dict:
    """Refait la décision d'une vidéo avec d'autres scores texte et/ou d'autres réglages."""
    segments = [s for s in copy.deepcopy(result["segments"]) if use_vision or s["source"] != "image"]
    for seg in segments:
        if seg["source"] != "image":  # les segments image gardent leur propre seuil (VISION_ALERT_THRESHOLD)
            seg["score"] = text_scores[seg["text"]]
            seg["flagged"] = text_alert(seg["text"], seg["score"], alert)
    return fuse(segments, alert_threshold=alert, convergence_share=share)


def metrics(cases: list[dict], risks: dict[str, dict]) -> dict:
    tp = sum(1 for c in cases if c["label"] == 1 and risks[c["id"]]["flagged"])
    fn = sum(1 for c in cases if c["label"] == 1 and not risks[c["id"]]["flagged"])
    fp = sum(1 for c in cases if c["label"] == 0 and risks[c["id"]]["flagged"])
    tn = sum(1 for c in cases if c["label"] == 0 and not risks[c["id"]]["flagged"])
    prev = [c for c in cases if c["category"] == "prevention"]
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": round(p, 3), "recall": round(r, 3),
        "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        # Indicateur clé du projet : part des contenus de prévention signalés à tort
        "prevention_fp": sum(risks[c["id"]]["flagged"] for c in prev), "prevention_n": len(prev),
    }


def by_category(cases: list[dict], risks: dict[str, dict]) -> list[tuple]:
    rows = []
    for cat in dict.fromkeys(c["category"] for c in cases):  # ordre d'apparition
        group = [c for c in cases if c["category"] == cat]
        correct = sum(risks[c["id"]]["flagged"] == bool(c["label"]) for c in group)
        rows.append((cat, group[0]["label"], len(group), correct))
    return rows


# ---------- 3. Rapport ----------

def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="réanalyse toutes les vidéos")
    parser.add_argument("--no-vision", action="store_true",
                        help="ignore l'analyse d'image (si le cache a été créé sans vision, relancer avec --refresh)")
    args = parser.parse_args()
    use_vision = ENABLE_VISION and not args.no_vision

    all_cases = yaml.safe_load(EVAL_CASES_PATH.read_text(encoding="utf-8"))["cases"]
    cases = [c for c in all_cases if (VIDEOS_DIR / f"{c['id']}.mp4").exists()]
    if not cases:
        print("Aucune vidéo de test : lance d'abord python scripts/build_eval_videos.py")
        return
    print(f"{len(cases)}/{len(all_cases)} cas disponibles.")
    results = analyze_all(cases, args.refresh, use_vision)

    # Scores texte : chaque texte unique est encodé une seule fois, puis scoré par chaque modèle
    texts = sorted({s["text"] for r in results.values() for s in r["segments"] if s["source"] != "image"})
    current = TextRiskClassifier.load(TEXT_CLF_PATH)
    emb = current.embed(texts) if texts else None
    scores_cur = dict(zip(texts, current.predict_proba_embeddings(emb).tolist())) if texts else {}

    # A. Réglages actuels
    risks = {c["id"]: decide(results[c["id"]], scores_cur, ALERT_THRESHOLD, CONVERGENCE_SHARE, use_vision) for c in cases}
    m = metrics(cases, risks)
    print(f"\n=== A. Réglages actuels (seuil {ALERT_THRESHOLD}, convergence {CONVERGENCE_SHARE}) ===")
    print(f"Précision {m['precision']:.2f} | Rappel {m['recall']:.2f} | F1 {m['f1']:.2f} | "
          f"VP={m['tp']} FP={m['fp']} FN={m['fn']} VN={m['tn']}")
    print(f"Prévention signalée à tort : {m['prevention_fp']}/{m['prevention_n']}")
    print(f"\n{'catégorie':22}{'attendu':>9}{'corrects':>10}")
    for cat, label, n, ok in by_category(cases, risks):
        print(f"{cat:22}{'signalé' if label else 'ignoré':>9}{ok:>7}/{n}")
    errors = [c for c in cases if risks[c["id"]]["flagged"] != bool(c["label"])]
    print("\nErreurs :" if errors else "\nAucune erreur.")
    for c in errors:
        r = risks[c["id"]]
        kind = "faux positif" if r["flagged"] else "faux négatif"
        print(f"  {c['id']} [{c['category']}] {kind} : score {r['score']:.2f}, niveau {r['level']}"
              + (f" | {r['reasons'][0]}" if r["reasons"] else ""))

    # B. Calibrage : balayage du seuil d'alerte
    print("\n=== B. Balayage du seuil d'alerte (texte) ===")
    print(f"{'seuil':>6}{'précision':>11}{'rappel':>8}{'F1':>6}{'FP prévention':>15}")
    sweep = []
    for t in SWEEP:
        st = metrics(cases, {c["id"]: decide(results[c["id"]], scores_cur, t, CONVERGENCE_SHARE, use_vision) for c in cases})
        sweep.append((t, st))
    # Meilleur compromis : F1 maximal ; à égalité, le moins de prévention signalée, puis la meilleure précision
    best_t = max(sweep, key=lambda x: (x[1]["f1"], -x[1]["prevention_fp"], x[1]["precision"]))[0]
    for t, st in sweep:
        marks = (" <- actuel" if t == ALERT_THRESHOLD else "") + (" <- meilleur compromis" if t == best_t else "")
        print(f"{t:>6.2f}{st['precision']:>11.2f}{st['recall']:>8.2f}{st['f1']:>6.2f}"
              f"{st['prevention_fp']:>10}/{st['prevention_n']}{marks}")

    # C. Généralisation des corrections : modèle précédent vs actuel, sur des formulations jamais vues
    comparison = []
    if TEXT_CLF_BACKUP_PATH.exists() and texts:
        prev_model = TextRiskClassifier.load(TEXT_CLF_BACKUP_PATH)
        e = emb if prev_model.embedder_name == current.embedder_name else prev_model.embed(texts)
        scores_prev = dict(zip(texts, prev_model.predict_proba_embeddings(e).tolist()))
        print("\n=== C. Modèle précédent -> actuel (score texte max par vidéo) ===")
        for c in cases:
            if c["category"] not in ("exclusion", "prevention"):
                continue
            seg_texts = [s["text"] for s in results[c["id"]]["segments"] if s["source"] != "image"]
            before = max((scores_prev[t] for t in seg_texts), default=0.0)
            after = max((scores_cur[t] for t in seg_texts), default=0.0)
            comparison.append((c["id"], c["category"], round(before, 2), round(after, 2)))
            print(f"  {c['id']} [{c['category']:10}] {before:.2f} -> {after:.2f} ({after - before:+.2f})")
    else:
        print("\n(C. pas de modèle précédent : comparaison ignorée)")

    # D. Rapport Markdown (jeu synthétique : versionnable) + JSON détaillé (data/, non versionné)
    case_rows = [[c["id"], c["category"], "signalé" if c["label"] else "ignoré",
                  risks[c["id"]]["level"], f"{risks[c['id']]['score']:.2f}",
                  "✅" if risks[c["id"]]["flagged"] == bool(c["label"]) else "❌"] for c in cases]
    report = "\n\n".join([
        "# Évaluation de Garde-Fil (sujet voisin : harcèlement)",
        f"Jeu de test synthétique : {len(cases)} vidéos (`eval/cases.yaml`). Généré par `scripts/evaluate.py`.",
        "## A. Réglages actuels",
        f"Seuil d'alerte {ALERT_THRESHOLD}, part de convergence {CONVERGENCE_SHARE}.",
        md_table(["Précision", "Rappel", "F1", "VP", "FP", "FN", "VN", "Prévention signalée à tort"],
                 [[m["precision"], m["recall"], m["f1"], m["tp"], m["fp"], m["fn"], m["tn"],
                   f"{m['prevention_fp']}/{m['prevention_n']}"]]),
        md_table(["Catégorie", "Attendu", "Corrects"],
                 [[cat, "signalé" if lbl else "ignoré", f"{ok}/{n}"] for cat, lbl, n, ok in by_category(cases, risks)]),
        md_table(["Cas", "Catégorie", "Attendu", "Niveau", "Score", "OK"], case_rows),
        "## B. Balayage du seuil d'alerte",
        f"Meilleur compromis (F1, puis le moins de prévention signalée, puis la précision) : seuil {best_t}.",
        md_table(["Seuil", "Précision", "Rappel", "F1", "Prévention signalée à tort"],
                 [[t, st["precision"], st["recall"], st["f1"], f"{st['prevention_fp']}/{st['prevention_n']}"]
                  for t, st in sweep]),
        "## C. Modèle précédent → actuel (score texte max)",
        md_table(["Cas", "Catégorie", "Avant", "Après"], [list(row) for row in comparison])
        if comparison else "Pas de modèle précédent.",
        "## Limites",
        "Jeu de test petit et synthétique (voix de synthèse, fonds unis) : les chiffres sont indicatifs. "
        "Calibrer et mesurer sur le même jeu surestime les performances ; un usage réel exigerait "
        "un jeu de validation et un jeu de test distincts, issus de données réelles encadrées.",
    ]) + "\n"
    EVAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_REPORT_PATH.write_text(report, encoding="utf-8")
    (EVAL_DIR / "report.json").write_text(json.dumps(
        {"metrics": m, "sweep": [{"threshold": t, **st} for t, st in sweep],
         "comparison": comparison}, ensure_ascii=False, indent=2))
    print(f"\nRapport écrit : {EVAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
