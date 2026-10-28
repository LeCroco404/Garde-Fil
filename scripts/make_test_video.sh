#!/usr/bin/env bash
# Garde-Fil : génère une vidéo verticale de test (720x1280) avec voix française + texte incrusté.
# Aucun contenu réel : tout est synthétique.
# Usage : ./scripts/make_test_video.sh "texte parlé" "texte à l'écran" sortie.mp4
set -euo pipefail

SPEECH="${1:-Bonjour, ceci est une vidéo de test du pipeline de modération.}"
OVERLAY="${2:-Texte incrusté de test}"
OUT="${3:-data/videos/test_01.mp4}"
FONT="/System/Library/Fonts/Supplement/Arial.ttf"

# Dossier temporaire supprimé automatiquement à la fin du script, même en cas d'erreur
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# 1. Synthèse vocale française (voix "Thomas", installée par défaut sur macOS)
say -v Thomas -o "$TMP_DIR/voice.aiff" "$SPEECH"

# 2. Fond uni + texte incrusté + piste audio
#    -shortest : la vidéo s'arrête quand l'audio se termine
ffmpeg -y -loglevel error \
  -f lavfi -i "color=c=navy:s=720x1280:d=30" \
  -i "$TMP_DIR/voice.aiff" \
  -vf "drawtext=fontfile=${FONT}:text='${OVERLAY}':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=h*0.7" \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "$OUT"

echo "Vidéo créée : $OUT"
