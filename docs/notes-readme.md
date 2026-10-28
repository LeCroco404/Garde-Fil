# Notes pour le README (à rédiger en phase 6)

Constats chiffrés relevés pendant le développement. Chaque phase ajoute les siennes.

## Phase 1 : pipeline texte

1. Classifieur (embeddings MiniLM multilingue + régression logistique), jeu textdetox FR, 955 textes de test : F1 0,894 (précision 0,888, rappel 0,901 sur la classe « toxique »). VP=427, FP=54, FN=47, VN=427.
2. Robustesse aux erreurs de transcription : Whisper a transcrit « T'es » en « C'était », la vidéo reste détectée (0,82). L'embedding capte le sens global, une liste de mots aurait échoué. Argument central contre les listes de mots.
3. Vidéo de prévention (« Le harcèlement en ligne, ça suffit… appelle le 3018 ») : 0,47, juste sous le seuil. Le modèle réagit au thème plus qu'à la posture : c'est le problème central à traiter pour le vrai sujet (témoignages de guérison).

## Phase 2 : OCR

1. Vidéo où le danger n'est qu'écrit : parole 0,06, texte à l'écran 0,86, alerte « affiché à l'écran à 0:02 ». Sans OCR, la vidéo passait inaperçue.
2. Le titre « Test toxique » obtient 0,07 : le modèle juge le sens, pas les mots. Un filtre par mots-clés l'aurait signalé.

## Phase 3 : fusion et explication

1. Choix de règles explicables plutôt qu'un modèle de fusion appris : le modérateur doit comprendre pourquoi une vidéo lui est présentée (supervision humaine, AI Act, DSA).
2. « Personne ne veut de toi ici » (exclusion calme) : 0,56, contre 0,86 pour une formulation agressive. La couverture du jeu de données compte plus que le modèle : c'est l'argument pour utiliser un jeu de recherche annoté par des cliniciens (CLPsych) lors de la transposition.
3. Limite identifiée : saturation à 1,00 (0,94 + bonus de convergence), qui empêche de classer les cas graves entre eux. À corriger au calibrage.

## Phase 4 : revue humaine et réapprentissage

1. Une correction de modérateur a fait passer le score du passage corrigé de 0,24 à 0,36, sans dégrader les performances générales (F1 0,894 → 0,893). Garde-fou de non-régression : modèle refusé si le F1 baisse de plus de 0,01, ancien modèle sauvegardé.
2. Précaution d'interprétation : la phrase corrigée fait partie de l'entraînement, donc ce +0,12 prouve l'assimilation, pas la généralisation (à mesurer en phase 6 sur des formulations nouvelles).
3. Choix de conception : décision sans valeur par défaut (limite le biais d'automatisation), mode audit des vidéos non signalées (seul moyen de trouver les faux négatifs), journal JSONL en ajout seul (traçabilité), pseudonyme et pas de commentaire libre (minimisation RGPD), une correction isolée ne renverse pas le modèle (protection contre l'erreur ou la malveillance d'un modérateur).

## Phase 5a : performance

1. Le profilage a montré que 66 % du temps de la première analyse venait du chargement à froid des modèles, pas du calcul. Après préchargement : 7,90 s → 3,05 s pour la première vidéo (÷ 2,6), résultats identiques au centième.
2. Transcription = poste principal une fois les modèles chargés (~70 % du temps).
3. OCR sélectif : une seule image lue sur quatre quand le texte est fixe. Critère de changement robuste aux textes courts (le premier critère, par écart moyen, les ratait).
4. Benchmark Whisper sur 30 s d'audio : base RTF 0,11, small RTF 0,27. Choix de small : base a transcrit « ferme-la et dégage » en « ferme l'aédégage », perdant le mot le plus agressif. Le rappel prime sur quelques secondes de calcul en file de modération.
5. Biais de mesure identifié : temps de chargement non comparables (premier lancement), d'où la nécessité d'un échauffement dans un benchmark rigoureux.
