#!/usr/bin/env bash
# Garde-Fil : génère une vidéo verticale de test (720x1280) avec voix française + texte incrusté.
# Aucun contenu réel : tout est synthétique.
#
# Usage : ./scripts/make_test_video.sh "texte parlé" "texte à l'écran" sortie.mp4 [début_incrustation_s]
#   - le texte à l'écran accepte apostrophes, deux-points, etc.
#   - "|" dans le texte à l'écran = retour à la ligne
#   - début_incrustation_s (optionnel, défaut 0) : seconde à laquelle le texte apparaît
set -euo pipefail

SPEECH="${1:-Bonjour, ceci est une vidéo de test du pipeline de modération.}"
OVERLAY="${2:-Texte incrusté de test}"
OUT="${3:-data/videos/test_01.mp4}"
OVERLAY_START="${4:-0}"
FONT="/System/Library/Fonts/Supplement/Arial.ttf"

# Dossier temporaire supprimé automatiquement à la fin du script, même en cas d'erreur
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# 1. Synthèse vocale française (voix "Thomas", installée par défaut sur macOS)
say -v Thomas -o "$TMP_DIR/voice.aiff" "$SPEECH"

# 2. Le texte incrusté passe par un fichier (textfile=) au lieu d'être écrit dans la commande :
#    ffmpeg n'a alors plus besoin d'échapper les caractères spéciaux (' : % etc.)
printf '%s' "$OVERLAY" | tr '|' '\n' > "$TMP_DIR/overlay.txt"

# 3. Fond uni + texte incrusté (blanc cerclé de noir, style TikTok) + piste audio
#    enable='gte(t,N)' : le texte n'apparaît qu'à partir de la seconde N
#    -shortest : la vidéo s'arrête quand l'audio se termine
ffmpeg -y -loglevel error \
  -f lavfi -i "color=c=navy:s=720x1280:d=30" \
  -i "$TMP_DIR/voice.aiff" \
  -vf "drawtext=fontfile=${FONT}:textfile=${TMP_DIR}/overlay.txt:fontcolor=white:fontsize=48:borderw=3:bordercolor=black:line_spacing=12:x=(w-text_w)/2:y=h*0.7:enable='gte(t,${OVERLAY_START})'" \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "$OUT"

echo "Vidéo créée : $OUT"
