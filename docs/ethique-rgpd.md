# Éthique, RGPD et transposition au vrai sujet

Ce document décrit le cadre dans lequel Garde-Fil a été conçu, ce que le prototype met déjà en œuvre, ce qu'il faudrait ajouter avant tout usage réel, et comment passer du sujet voisin (harcèlement) au sujet visé (automutilation et suicide) sans collecter soi-même de contenus dangereux.

## 1. Finalité

Garde-Fil aide un modérateur humain à trier des vidéos courtes. Il attribue un score de risque, explique ses raisons et recommande une action. Il ne supprime, ne bloque et ne signale rien de lui-même à l'extérieur. Pour une vidéo jugée à risque, l'action recommandée est d'orienter la personne vers une aide (un message d'orientation vers le 3114 sur le vrai sujet), plutôt que de seulement retirer le contenu.

## 2. Pourquoi un sujet voisin

Collecter soi-même des vidéos d'automutilation sur les réseaux sociaux poserait trois problèmes. Le premier est la sécurité : on accumule et on manipule des contenus qui peuvent nuire, y compris à la personne qui les traite. Le deuxième est légal : ces contenus révèlent des informations sur la santé de personnes identifiables, une catégorie particulière de données au sens de l'article 9 du RGPD, dont le traitement est en principe interdit sauf exceptions encadrées. Le troisième est éthique : les auteurs n'ont pas consenti à servir de matière d'entraînement.

Le projet a donc été développé sur le harcèlement en ligne, avec un jeu public sous licence (`textdetox/multilingual_toxicity_dataset`) et des vidéos synthétiques écrites pour l'occasion. Ce sujet partage les mêmes difficultés techniques : un vocabulaire ambigu, des contenus de prévention qui parlent du sujet sans être à risque, des formulations indirectes. La méthode se transpose ; les données, non.

## 3. Choix de conception liés à l'éthique

1. Supervision humaine. Le niveau de risque ne déclenche qu'une revue. La décision sur la vidéo n'a pas de valeur par défaut dans le tableau de bord, pour limiter le biais d'automatisation (la tendance à valider ce que propose la machine).
2. Explicabilité. Chaque signalement indique le passage en cause, sa source (parole, écran, image), son horodatage et son score. Des règles de fusion simples ont été préférées à un modèle de fusion appris, pour que le modérateur puisse comprendre et contester.
3. Pas de liste de mots-clés. Une liste est facile à contourner et signale à tort les messages de prévention, qui emploient le même vocabulaire. Elle obligerait aussi à écrire et maintenir un répertoire de formulations dangereuses, ce que le projet refuse.
4. Les contenus de prévention comme cas de test prioritaire. Le jeu d'évaluation contient une catégorie dédiée, dont un témoignage de guérison. L'indicateur « prévention signalée à tort » est suivi à chaque réglage.
5. Asymétrie des erreurs assumée. Rater un contenu à risque est plus grave qu'une fausse alerte. Un réglage qui supprime des fausses alertes en perdant un vrai cas a été refusé (voir [evaluation.md](evaluation.md)).
6. Prudence sur les sources non calibrées. Si seule l'image signale un risque, le niveau est plafonné à « modéré ».
7. Robustesse de la boucle de réapprentissage. Les corrections du modérateur sont pondérées, mais une correction isolée ne renverse pas le modèle. Un nouveau modèle n'est adopté que s'il ne dégrade pas les performances sur un jeu de test fixe, et l'ancien est conservé.

## 4. Données personnelles et RGPD

Dans un déploiement réel, les vidéos, leurs transcriptions, le texte lu à l'écran et les images extraites sont des données personnelles. Sur le vrai sujet, elles peuvent révéler des informations de santé et concerner des mineurs. Une analyse d'impact relative à la protection des données (AIPD, article 35 du RGPD) devrait être menée avant toute mise en production, avec le délégué à la protection des données de l'organisme. La base légale et les durées de conservation dépendent du responsable de traitement (la plateforme) et ne peuvent pas être fixées par le prototype.

Mesures déjà en place dans le prototype :

- traitement entièrement local, sans API externe pour l'analyse ;
- `data/` et `models/` exclus de Git, de sorte qu'aucune vidéo, transcription, revue ou modèle entraîné n'est publié ;
- le cache d'embeddings ne contient que le jeu public, jamais de transcriptions ;
- modérateur identifié par un pseudonyme ;
- aucun champ de commentaire libre dans le formulaire de revue (minimisation) ;
- journal des revues en ajout seul, qui garde la trace de qui a décidé quoi et quand.

À ajouter avant un usage réel : durées de conservation et purge automatique des transcriptions et des images, chiffrement des données au repos, authentification et contrôle d'accès au tableau de bord, journalisation des accès, procédure pour l'exercice des droits des personnes.

## 5. Cadre réglementaire

Le règlement européen sur les services numériques (DSA, article 17) impose aux hébergeurs qui restreignent un contenu de fournir à l'utilisateur concerné un exposé des motifs. Celui-ci doit notamment préciser si des moyens automatisés ont été utilisés pour détecter ou identifier le contenu, et quelles voies de recours existent. Les raisons horodatées produites par Garde-Fil et le journal des revues fournissent la matière de cet exposé.

Le règlement européen sur l'IA (AI Act) impose des obligations qui dépendent de l'usage précis du système. La qualification juridique d'un déploiement réel reste à établir avec des juristes. Le prototype applique dans tous les cas les principes de supervision humaine, de transparence et de traçabilité.

## 6. Biais et équité

Le classifieur de texte hérite des choix d'annotation de son jeu d'entraînement. L'évaluation a déjà montré une confusion entre négativité et toxicité (« cette journée est horrible », 0,50). CLIP a été entraîné sur des images et légendes du web et peut associer plus facilement certaines notions (« agressivité », par exemple) à certaines apparences physiques. Avant tout usage réel, il faudrait mesurer les taux de faux positifs par groupe (langue, registre, accent pour la transcription, apparence pour l'image), et faire relire les cas litigieux par des personnes concernées.

## 7. Sécurité

Le projet est aussi un exercice de cybersécurité. Le modèle de menace couvre quatre points.

1. Contournement. Un auteur peut chercher à échapper à la détection en déformant le texte, en le plaçant dans l'image plutôt que dans la parole, ou en dégradant l'audio. La combinaison de trois sources et la classification par le sens réduisent l'effet de ces techniques sans le supprimer. Le projet ne documente pas de techniques de contournement spécifiques au vrai sujet.
2. Empoisonnement de la boucle de réapprentissage. Un modérateur malveillant ou négligent pourrait injecter de fausses étiquettes. Les mesures : poids limité des corrections, garde-fou de non-régression, conservation de l'ancien modèle, traçabilité de chaque revue par pseudonyme.
3. Confidentialité. Traitement local, données hors du dépôt Git, cache limité aux données publiques.
4. Accès. Le tableau de bord n'a pas d'authentification : il ne doit pas être exposé sur un réseau. Un déploiement réel exigerait authentification, gestion des rôles et chiffrement des échanges.

## 8. Transposition au vrai sujet

### 8.1 Données

Pas de collecte personnelle sur les réseaux sociaux. Deux voies encadrées existent.

La première passe par des jeux de recherche sous convention. Le plus connu est l'UMD Reddit Suicidality Dataset, dont la version 2 contient les données de la tâche partagée CLPsych 2019 et des annotations d'experts. L'accès se fait sur dossier, examiné par un comité issu d'un conseil créé par l'American Association of Suicidology. Le dossier doit inclure un protocole de recherche et l'approbation (ou l'exemption) d'un comité d'éthique de la recherche, ainsi qu'un plan de gestion des données. Une simple validation par un encadrant ne suffit pas. Pour un étudiant, cela suppose de s'inscrire dans un projet de recherche encadré. Ce jeu est en anglais : le modèle d'embeddings multilingue permettrait de l'utiliser, mais les performances sur des contenus francophones devraient être mesurées à part.

La seconde est un partenariat avec une plateforme ou une association, avec un cadre contractuel, une AIPD et l'implication de cliniciens.

### 8.2 Ce qui change dans le code

Le pipeline, la fusion, le tableau de bord et la boucle de réapprentissage restent identiques. Il faut changer :

1. le jeu d'entraînement du classifieur de texte (`DATASET_NAME` dans `src/config.py`) ;
2. la ressource d'aide (`HELP_RESOURCE`), qui passe au 3114. Ce numéro est gratuit, ouvert 24 h/24 et 7 j/7 dans toute la France ; les appels sont pris par des infirmiers et des psychologues formés, et il s'adresse aux personnes en détresse, à leurs proches et aux professionnels ;
3. les descriptions visuelles de CLIP (`VISION_PROMPTS`), à définir avec des cliniciens à partir d'une taxonomie validée. Elles ne sont pas rédigées dans ce projet ;
4. le jeu d'évaluation (`eval/cases.yaml`), à reconstruire à partir de données réelles encadrées, avec une catégorie « prévention et témoignages de guérison » au moins aussi fournie que dans la version actuelle, et des jeux de validation et de test distincts.

### 8.3 Ce qui doit être défini avec des cliniciens

La définition des niveaux de risque et des actions associées. Le contenu des messages d'orientation : les recommandations de l'OMS sur la manière de parler du suicide, relayées en France par le programme Papageno, sont un point de départ. Une procédure d'escalade pour les situations d'urgence, que la décision « Escalade » du tableau de bord ne fait qu'amorcer. Le soutien psychologique des modérateurs, exposés de façon répétée à des contenus éprouvants.

## 9. Ce que Garde-Fil ne fait pas

Il ne pose pas de diagnostic et ne juge pas l'état d'une personne. Il ne contacte personne. Il ne supprime aucun contenu. Il ne remplace ni un modérateur, ni un professionnel de santé.

## Sources

- [Santé.fr : le numéro national de prévention du suicide (3114)](https://www.sante.fr/le-numero-national-de-prevention-du-suicide-0)
- [UMD Reddit Suicidality Dataset : conditions d'accès](https://psresnik.github.io/umd_reddit_suicidality_dataset.html)
- [Santé publique France : Suicide, comment en parler ? Le programme Papageno](https://www.santepubliquefrance.fr/docs/suicide-comment-en-parler-le-programme-papageno)
- [DSA, article 17 : exposé des motifs](https://www.springlex.eu/en/packages/dsa/dsa-regulation/article-17/)
- [textdetox/multilingual_toxicity_dataset](https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset)
