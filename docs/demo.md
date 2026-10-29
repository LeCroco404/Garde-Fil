# Démo de Garde-Fil

Scénario de 2 minutes environ, à enregistrer ou à dérouler en direct. Toutes les vidéos utilisées sont synthétiques.

## Préparation

```bash
cd ~/Documents/Ethan/Informatique/PROJETS/Garde-Fil && source .venv/bin/activate
streamlit run app/dashboard.py
```

Dans la barre latérale, clique sur « Réanalyser toutes les vidéos » et attends la fin de l'analyse. Vérifie que la file contient au moins `convergence.mp4`, `ecran.mp4`, `prevention.mp4` et `img_boxe.mp4`.

## Scénario

1. File de revue (15 s). Montrer la barre latérale : vidéos triées par urgence, badges de niveau, compteurs de revues et de corrections. Phrase clé : « le système trie, il ne décide pas ».
2. Convergence (30 s). Ouvrir `convergence.mp4`. Montrer la raison « dit ET affiché à l'écran » et le bonus de convergence, puis le bouton « Aller à 0:00 » qui place la vidéo sur le passage.
3. Écran seul (20 s). Ouvrir `ecran.mp4`. La parole est neutre, le message n'est qu'écrit : sans OCR, la vidéo passait inaperçue.
4. Prévention (20 s). Activer « Inclure les vidéos non signalées », ouvrir `prevention.mp4`. Elle parle de harcèlement mais n'est pas transmise au modérateur.
5. Image et contexte (15 s). Ouvrir `img_boxe.mp4`. CLIP voit une bagarre ; le niveau est plafonné à modéré et la raison l'explique.
6. Revue (20 s). Sur une vidéo, tenter d'enregistrer sans décision (message d'erreur), corriger une étiquette, choisir une décision, enregistrer. La vidéo passe en « revue ».
7. Conclusion (10 s). Montrer `docs/evaluation.md` : 0/5 contenus de prévention signalés à tort.

## Captures d'écran pour le README

À ranger dans `docs/img/` (versionné), en vérifiant qu'aucune donnée personnelle n'y figure :

1. `file-de-revue.png` : barre latérale avec la file triée.
2. `convergence.png` : vue d'une vidéo signalée avec ses raisons et l'action recommandée.
3. `formulaire-revue.png` : formulaire de revue des passages.
4. `prevention.png` : une vidéo de prévention en mode audit, non transmise.
5. `pipeline-terminal.png` : sortie de `python -m src.pipeline` avec la ligne `Temps :`.

Sur macOS, Cmd + Maj + 4 capture une zone de l'écran ; Cmd + Maj + 5 enregistre une vidéo de l'écran, à convertir en GIF si besoin :

```bash
ffmpeg -i demo.mov -vf "fps=10,scale=960:-1" -loop 0 docs/img/demo.gif
```
