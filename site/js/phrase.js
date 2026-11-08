// Garde-Fil : page « Tester une phrase ». Le calcul se fait entièrement dans le navigateur.
// Transformers.js et son moteur WebAssembly sont hébergés sur le site (vendor/) : la CSP n'autorise
// aucun script tiers. Seul le modèle est téléchargé depuis Hugging Face ; la phrase ne quitte jamais la page.
import { chargerJSON, el, formatScore, typo, verdict } from "./outils.js";

const VENDOR = new URL("../vendor/transformers/", import.meta.url);
const MODELE_NAVIGATEUR = "Xenova/paraphrase-multilingual-MiniLM-L12-v2"; // version ONNX du modèle Python

// Libellés prudents : un score bas n'est jamais présenté comme une absence de danger
const LIBELLES = {
  "élevé": { tampon: "Harcèlement probable",
             sens: "Une vidéo contenant cette phrase serait montrée en priorité à un modérateur humain." },
  "modéré": { tampon: "Harcèlement possible",
              sens: "Une vidéo contenant cette phrase serait montrée à un modérateur humain." },
  "faible": { tampon: "Score bas",
              sens: "Le modèle n'a pas reconnu de harcèlement dans cette phrase. Cela ne veut pas dire qu'elle "
                  + "est sans danger : ce prototype ne reconnaît pas de façon fiable les menaces, les incitations "
                  + "à se faire du mal ou les formulations indirectes." },
};

const $ = (id) => document.getElementById(id);
const params = await chargerJSON("data/text_model.json");
let extracteur = null;

function chargerExtracteur() {
  // Chargé au premier usage seulement : la page reste légère pour qui ne teste pas de phrase
  extracteur ??= (async () => {
    const { pipeline, env } = await import(new URL("transformers.min.js", VENDOR).href);
    env.allowLocalModels = false;
    env.backends.onnx.wasm.wasmPaths = VENDOR.href;  // moteur WebAssembly servi par le site, pas par un CDN
    return pipeline("feature-extraction", MODELE_NAVIGATEUR, {
      dtype: "q8",  // modèle compressé en 8 bits, vérifié contre Python (voir #verif)
      progress_callback: (p) => {
        if (p.status === "progress" && p.total) {
          $("etat").textContent = `Téléchargement du modèle : ${Math.round((100 * p.loaded) / p.total)} %`;
        }
      },
    });
  })();
  return extracteur;
}

async function scorer(texte) {
  const extraire = await chargerExtracteur();
  // Même calcul qu'en Python : moyenne des vecteurs de mots, normalisée, puis régression logistique
  const sortie = await extraire(texte, { pooling: "mean", normalize: true });
  let z = params.intercept;
  for (let i = 0; i < params.coef.length; i++) z += sortie.data[i] * params.coef[i];
  return 1 / (1 + Math.exp(-z));
}

function niveau(texte, score) {
  const mots = texte.match(/[\p{L}\p{M}\p{N}_]+/gu) ?? [];  // équivalent de \w+ en Python
  const tropCourte = mots.length < params.min_words_to_flag;
  if (score < params.alert_threshold || tropCourte) return ["faible", tropCourte];
  return [score >= params.high_threshold ? "élevé" : "modéré", false];
}

$("formulaire").addEventListener("submit", async (evenement) => {
  evenement.preventDefault();
  const texte = $("phrase").value.trim();
  if (!texte) {
    $("etat").textContent = "Écrivez une phrase d'abord.";
    $("phrase").focus();
    return;
  }
  $("analyser").disabled = true;
  $("etat").textContent = "Analyse en cours. La première fois, le modèle se télécharge.";
  try {
    const score = await scorer(texte);
    const [n, tropCourte] = niveau(texte, score);
    $("resultat").replaceChildren(
      el("h2", {}, "Résultat"),
      verdict(n, score, LIBELLES),
      tropCourte && el("p", { class: "aide" },
        "Phrase de moins de 3 mots : trop courte pour déclencher une alerte à elle seule, "
        + "car sans contexte le système se trompe facilement."),
      el("p", { class: "aide" },
        "Si cette phrase vous concerne, ou concerne un proche : 3114 (gratuit, 24 h/24, 7 j/7). "
        + "En cas de danger immédiat : 15 ou 112."),
    );
    $("etat").textContent = "";
  } catch (erreur) {
    extracteur = null;
    $("etat").textContent = "Le modèle n'a pas pu être chargé (connexion ou navigateur trop ancien). Réessayez.";
    console.error(erreur);
  } finally {
    $("analyser").disabled = false;
  }
});

$("formulaire").addEventListener("reset", () => {
  $("resultat").replaceChildren();
  $("etat").textContent = "";
});

// ---------- Vérification (#verif) : navigateur contre scores Python de référence ----------
function afficherVerification() {
  if (location.hash === "#verif") $("verif").hidden = false;
}
afficherVerification();
window.addEventListener("hashchange", afficherVerification);

$("verif-lancer").addEventListener("click", async () => {
  $("verif-lancer").disabled = true;
  $("verif-resume").textContent = "Vérification en cours…";
  const reference = await chargerJSON("data/reference.json");
  let ecartMax = 0;
  let memeDecision = 0;
  const lignes = [];
  for (const r of reference) {
    const s = await scorer(r.text);
    const ecart = Math.abs(s - r.score);
    ecartMax = Math.max(ecartMax, ecart);
    if ((s >= params.alert_threshold) === (r.score >= params.alert_threshold)) memeDecision++;
    lignes.push(el("tr", {},
      el("td", {}, r.text),
      el("td", { class: "nombre" }, formatScore(r.score)),
      el("td", { class: "nombre" }, formatScore(s)),
      el("td", { class: "nombre" }, formatScore(ecart))));
  }
  $("verif-lignes").replaceChildren(...lignes);
  $("verif-resume").textContent = typo(
    `Écart maximal : ${formatScore(ecartMax)} · même décision au seuil de 0,50 : ${memeDecision} sur ${reference.length}`);
  $("verif-lancer").disabled = false;
});
