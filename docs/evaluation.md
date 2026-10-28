# Évaluation de Garde-Fil (sujet voisin : harcèlement)

Jeu de test synthétique : 23 vidéos (`eval/cases.yaml`). Généré par `scripts/evaluate.py`.

## A. Réglages actuels

Seuil d'alerte 0.5, part de convergence 0.5.

| Précision | Rappel | F1 | VP | FP | FN | VN | Prévention signalée à tort |
|---|---|---|---|---|---|---|---|
| 0.778 | 0.778 | 0.778 | 7 | 2 | 2 | 12 | 0/5 |

| Catégorie | Attendu | Corrects |
|---|---|---|
| neutre | ignoré | 5/5 |
| negatif_sans_cible | ignoré | 1/2 |
| risque | signalé | 5/5 |
| exclusion | signalé | 1/3 |
| prevention | ignoré | 5/5 |
| image_neutre | ignoré | 1/1 |
| image_risque | signalé | 1/1 |
| image_contexte | ignoré | 0/1 |

| Cas | Catégorie | Attendu | Niveau | Score | OK |
|---|---|---|---|---|---|
| n1 | neutre | ignoré | faible | 0.08 | ✅ |
| n2 | neutre | ignoré | faible | 0.01 | ✅ |
| n3 | neutre | ignoré | faible | 0.05 | ✅ |
| n4 | neutre | ignoré | faible | 0.06 | ✅ |
| n5 | neutre | ignoré | faible | 0.07 | ✅ |
| h1 | negatif_sans_cible | ignoré | faible | 0.41 | ✅ |
| h2 | negatif_sans_cible | ignoré | modéré | 0.50 | ❌ |
| r1 | risque | signalé | élevé | 0.98 | ✅ |
| r2 | risque | signalé | élevé | 0.86 | ✅ |
| r3 | risque | signalé | élevé | 0.87 | ✅ |
| r4 | risque | signalé | modéré | 0.69 | ✅ |
| r5 | risque | signalé | élevé | 0.92 | ✅ |
| e1 | exclusion | signalé | faible | 0.23 | ❌ |
| e2 | exclusion | signalé | modéré | 0.53 | ✅ |
| e3 | exclusion | signalé | faible | 0.18 | ❌ |
| p1 | prevention | ignoré | faible | 0.42 | ✅ |
| p2 | prevention | ignoré | faible | 0.61 | ✅ |
| p3 | prevention | ignoré | faible | 0.31 | ✅ |
| p4 | prevention | ignoré | faible | 0.37 | ✅ |
| p5 | prevention | ignoré | faible | 0.29 | ✅ |
| i1 | image_neutre | ignoré | faible | 0.00 | ✅ |
| i2 | image_risque | signalé | modéré | 0.91 | ✅ |
| i3 | image_contexte | ignoré | modéré | 0.96 | ❌ |

## B. Balayage du seuil d'alerte

Meilleur compromis (F1, puis le moins de prévention signalée, puis la précision) : seuil 0.45.

| Seuil | Précision | Rappel | F1 | Prévention signalée à tort |
|---|---|---|---|---|
| 0.3 | 0.538 | 0.778 | 0.636 | 3/5 |
| 0.35 | 0.583 | 0.778 | 0.667 | 2/5 |
| 0.4 | 0.636 | 0.778 | 0.7 | 1/5 |
| 0.45 | 0.778 | 0.778 | 0.778 | 0/5 |
| 0.5 | 0.778 | 0.778 | 0.778 | 0/5 |
| 0.55 | 0.857 | 0.667 | 0.75 | 0/5 |
| 0.6 | 0.857 | 0.667 | 0.75 | 0/5 |
| 0.65 | 0.857 | 0.667 | 0.75 | 0/5 |
| 0.7 | 0.833 | 0.556 | 0.667 | 0/5 |
| 0.75 | 0.833 | 0.556 | 0.667 | 0/5 |
| 0.8 | 0.833 | 0.556 | 0.667 | 0/5 |

## C. Modèle précédent → actuel (score texte max)

| Cas | Catégorie | Avant | Après |
|---|---|---|---|
| e1 | exclusion | 0.19 | 0.23 |
| e2 | exclusion | 0.43 | 0.53 |
| e3 | exclusion | 0.15 | 0.18 |
| p1 | prevention | 0.46 | 0.42 |
| p2 | prevention | 0.63 | 0.62 |
| p3 | prevention | 0.3 | 0.31 |
| p4 | prevention | 0.36 | 0.37 |
| p5 | prevention | 0.3 | 0.29 |

## Limites

Jeu de test petit et synthétique (voix de synthèse, fonds unis) : les chiffres sont indicatifs. Calibrer et mesurer sur le même jeu surestime les performances ; un usage réel exigerait un jeu de validation et un jeu de test distincts, issus de données réelles encadrées.
