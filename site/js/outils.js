// Garde-Fil : outils partagés par les pages de la démo.
// Tout le texte est inséré avec textContent (jamais innerHTML) : aucune injection de code possible.

const NBSP = " ";

/** Typographie française pour les textes issus des données : espaces insécables, décimales à virgule. */
export function typo(texte) {
  return String(texte)
    .replace(/ ([:;!?»])/g, `${NBSP}$1`)
    .replace(/« /g, `«${NBSP}`)
    .replace(/(\d)\.(\d)/g, "$1,$2");
}

/** 0.972 -> « 0,97 » */
export const formatScore = (x) => x.toFixed(2).replace(".", ",");

/** 12.4 -> « 0:12 » */
export const formatTemps = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

/** Crée un élément ; les chaînes passées en enfants deviennent du texte (typographie appliquée). */
export function el(tag, attributs = {}, ...enfants) {
  const e = document.createElement(tag);
  for (const [nom, valeur] of Object.entries(attributs)) {
    if (nom === "class") e.className = valeur;
    else e.setAttribute(nom, valeur);
  }
  for (const enfant of enfants) {
    if (enfant === null || enfant === undefined || enfant === false) continue;
    e.append(typeof enfant === "string" ? typo(enfant) : enfant);
  }
  return e;
}

export async function chargerJSON(chemin) {
  const reponse = await fetch(chemin);
  if (!reponse.ok) throw new Error(`${chemin} : ${reponse.status}`);
  return reponse.json();
}

/**
 * Bloc « verdict » : tampon encré + explication + jauge du score.
 * niveau : "élevé" | "modéré" | "faible" ; libelles : { tampon, sens } par niveau.
 */
const CLASSES = { "élevé": "eleve", "modéré": "modere", "faible": "faible" };

export function verdict(niveau, score, libelles) {
  const { tampon, sens } = libelles[niveau];
  const jauge = el("meter", {
    min: "0", max: "1", low: "0.5", high: "0.75", optimum: "0",
    value: String(score), "aria-label": "Score de risque",
  });
  return el("div", { class: `verdict verdict--${CLASSES[niveau]}` },
    el("p", { class: "tampon" }, tampon),
    el("div", {},
      el("p", {}, sens),
      el("p", { class: "score" }, `Score : ${formatScore(score)} sur 1`, jauge),
    ),
  );
}
