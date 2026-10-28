#!/usr/bin/env bash
# Garde-Fil : fabrique une vidéo verticale de test (720x1280) à partir d'une IMAGE fixe.
# Sert à tester l'analyse d'image (CLIP). Utilise des images dont tu as les droits
# (tes propres photos ou une banque d'images libre de droits) : jamais de contenu réel collecté.
#
# Usage : ./scripts/make_image_video.sh image.jpg sortie.mp4 ["texte parlé"]
#   - sans texte parlé : vidéo de 5 s avec une piste audio silencieuse
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage : $0 image.jpg sortie.mp4 [\"texte parlé\"]" >&2; exit 1
fi
IMAGE="$1"
OUT="$2"
SPEECH="${3:-}"
if [ ! -f "$IMAGE" ]; then
  echo "Image introuvable : $IMAGE (dépose-la d'abord dans data/images/)" >&2; exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# 1. Audio : voix de synthèse, ou 5 s de silence
if [ -n "$SPEECH" ]; then
  say -v Thomas -o "$TMP_DIR/voice.aiff" "$SPEECH"
  AUDIO=(-i "$TMP_DIR/voice.aiff")
else
  AUDIO=(-f lavfi -t 5 -i "anullsrc=r=44100:cl=mono")
fi

# 2. Image en boucle, redimensionnée et centrée sur fond noir au format 9:16
ffmpeg -y -loglevel error \
  -loop 1 -i "$IMAGE" "${AUDIO[@]}" \
  -vf "scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p" \
  -r 25 -c:v libx264 -c:a aac -shortest "$OUT"

echo "Vidéo créée : $OUT"
