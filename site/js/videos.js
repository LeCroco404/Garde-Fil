// Garde-Fil : page « Vidéos d'exemple ». Les vidéos sont analysées à l'avance (data/examples.json).
import { chargerJSON, el, formatScore, formatTemps, typo, verdict } from "./outils.js";

const LIBELLES = {
  "élevé": { tampon: "Risque élevé", sens: "La vidéo serait montrée en priorité à un modérateur humain." },
  "modéré": { tampon: "Risque modéré", sens: "La vidéo serait montrée à un modérateur humain." },
  "faible": { tampon: "Risque faible", sens: "La vidéo ne serait pas montrée au modérateur." },
};
const SOURCES = { parole: "Dit", ecran: "Écrit", image: "Image" };
const SOURCES_LONGUES = { parole: "dit dans la vidéo", ecran: "affiché à l'écran", image: "visible à l'image" };

const $ = (id) => document.getElementById(id);
const exemples = await chargerJSON("data/examples.json");

for (const [i, ex] of exemples.entries()) {
  $("exemple").append(el("option", { value: String(i) }, ex.title));
}

function afficher(i) {
  const ex = exemples[i];
  const risque = ex.risk;

  // Vidéo, sous-titres (parole d'origine) et description du contenu visuel
  const video = $("video");
  video.replaceChildren();
  video.src = ex.video;
  if (ex.captions) {
    video.append(el("track", { kind: "captions", srclang: "fr", label: "Français", src: ex.captions }));
  }
  $("description").textContent = ex.description ? typo(ex.description) : "";

  // Verdict : un nouveau tampon est posé à chaque changement d'exemple
  $("verdict").replaceChildren(verdict(risque.level, risque.score, LIBELLES));

  $("note").textContent = typo(ex.note);
  $("raisons").replaceChildren(...risque.reasons.map((r) => el("li", {}, r)));
  $("raisons").hidden = risque.reasons.length === 0;
  $("sans-raison").hidden = risque.reasons.length > 0;
  $("action").textContent = typo(risque.action);

  // Détail : chaque passage entendu ou lu, avec son score
  $("segments").replaceChildren(...ex.segments.map((s) => el("li", { class: "signal" },
    el("span", { class: `badge ${s.flagged ? "badge--grave" : "badge--info"}` }, SOURCES[s.source]),
    el("div", {},
      el("p", {}, `« ${s.text} »`),
      el("p", { class: "preuves" },
        `${SOURCES_LONGUES[s.source]} à ${formatTemps(s.start)} · score ${formatScore(s.score)} · `
        + (s.flagged ? "passage signalé" : "non signalé")),
    ),
  )));
}

// Lien direct vers un exemple : videos.html?exemple=prevention
const demande = new URLSearchParams(location.search).get("exemple");
const depart = Math.max(0, exemples.findIndex((ex) => ex.id === demande));
$("exemple").value = String(depart);
afficher(depart);

$("exemple").addEventListener("change", (e) => {
  const i = Number(e.target.value);
  afficher(i);
  history.replaceState(null, "", `?exemple=${encodeURIComponent(exemples[i].id)}`);
});
