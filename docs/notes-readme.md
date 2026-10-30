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

## Phase 5b : analyse d'image (CLIP zero-shot)

1. Sans aucun entraînement : photo de repas 0,00 ; dispute 0,91 (« geste menaçant ») ; vidéo toxique sur fond uni 0,00 à l'image (la détection reste portée par la parole).
2. Faux positif attendu : match de boxe 0,96 (« bagarre »). CLIP détecte le contenu, pas l'intention ni le contexte : même limite que la vidéo de prévention côté texte.
3. Règle ajoutée : une source image seule (non calibrée) plafonne au niveau MODÉRÉ ; le niveau ÉLEVÉ exige la convergence d'une source texte. Le score est conservé et le plafond est expliqué au modérateur.
4. Bruit d'OCR sur photo réelle (« À à | BE À », cordes du ring) : filtre durci, au moins un mot de 3 lettres.
5. Points éthiques : descriptions neutres indispensables (score relatif) ; biais hérités du web (risque d'associer « agressivité » à certaines apparences) à mesurer par groupe avant tout usage réel ; descriptions du vrai sujet à définir avec des cliniciens, jamais rédigées dans le projet.

## Phase 6a : évaluation de bout en bout et calibrage

1. Jeu de test vidéo synthétique versionné (`eval/cases.yaml`) : 23 cas en 8 catégories, dont 5 de prévention (y compris un témoignage de guérison) qui ne doivent pas être signalés. Indicateur clé : prévention signalée à tort.
2. Analyse coûteuse mise en cache, décision rejouée : 11 seuils testés en quelques secondes.
3. Première mesure (seuil 0,50) : précision 0,70, rappel 0,78, prévention signalée à tort 1/5. Le « meilleur compromis » calculé (seuil 0,65) supprimait les fausses alertes mais faisait perdre un vrai cas de harcèlement : refusé, car rater un contenu à risque est plus grave qu'une fausse alerte.
4. Analyse de cause plutôt que réglage de seuil : la fausse alerte de prévention venait du titre « Chiffres clés » (2 mots, score 0,61), un fragment trop court pour être jugé sans contexte. Règle ajoutée : un texte de moins de 3 mots ne peut pas alerter seul. Contrepartie assumée : un titre d'un seul mot n'est plus détecté seul.
5. Bug trouvé par l'évaluation : sans passage signalé, le score informatif pouvait quand même faire monter le niveau (contournement de la règle). Corrigé : seul un passage signalé peut élever le niveau.
6. Résultat final (seuil 0,50 maintenu) : précision 0,78, rappel 0,78, F1 0,78, prévention signalée à tort 0/5, risque explicite 5/5, neutre 5/5, sans perte de rappel. Après correction, monter le seuil ne fait plus que perdre du rappel : la cause était bien structurelle.
7. Généralisation des corrections du modérateur, sur des formulations d'exclusion jamais vues : +0,03 à +0,10 ; un cas (cantine) franchit le seuil uniquement grâce à la correction (0,43 → 0,53). Aucun effet collatéral sur la prévention (±0,04).
8. Limites restantes, documentées : exclusion calme mal couverte (2/3 manqués, problème de données, pas de seuil) ; négativité confondue avec la toxicité (« cette journée est horrible », 0,50) ; boxe signalée par l'image (plafonnée à MODÉRÉ) ; jeu petit et synthétique, calibré et mesuré sur les mêmes cas, donc chiffres indicatifs.
9. Bonus de fusion rendus proportionnels (part de la marge restante) : plus de saturation à 1,00 (0,94 + convergence = 0,97).

## Démo publique

1. Hébergement : Hugging Face a rendu payants les Spaces Docker sur le matériel gratuit (refus « 402 Payment Required »). Choix d'une page statique sur GitHub Pages : gratuite, sans serveur, sans mise en veille.
2. Aucune vidéo envoyée par les visiteurs : pas de données personnelles reçues, pas de point de collecte de contenus dangereux. Huit vidéos synthétiques analysées à l'avance (photos réelles exclues : licences des banques d'images).
3. Test de phrase exécuté dans le navigateur (Transformers.js, modèle compressé en 8 bits) : la phrase ne quitte pas l'ordinateur du visiteur. Vérification contre Python : écart maximal 0,04, même décision sur 22/22 phrases de référence.
4. Modèle exporté en JSON (coefficients) plutôt qu'en pickle : charger un pickle revient à exécuter du code.
5. Limite révélée par un test sur la démo en ligne : une incitation à se faire du mal formulée sans insulte obtient 0,07. Le jeu d'entraînement (toxicité, surtout des insultes) ne couvre ni les menaces ni les incitations. Correction de la présentation, pas du modèle : avertissement « prototype limité », libellé « Score bas » sans couleur rassurante (au lieu de « Risque faible » en vert), rappel du 3114 sous chaque résultat. Pas de liste de mots ni de phrases dangereuses écrites pour l'entraînement : la vraie correction passe par des données annotées et encadrées.
