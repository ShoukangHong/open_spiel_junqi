#!/bin/bash
# Pack project for cloud deployment — no git needed on the server.
# Run locally (Git Bash):  bash scripts/pack.sh
# Upload open_spiel_junqi.tar.gz to cloud, then:
#   tar -xzf open_spiel_junqi.tar.gz && cd open_spiel_junqi && bash scripts/cloud_setup.sh
set -euo pipefail

OUTPUT="open_spiel_junqi.tar.gz"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJ_NAME="$(basename "$PROJ_DIR")"

echo "Packing $PROJ_DIR → $OUTPUT ..."

cd "$PROJ_DIR/.."

tar -czf "$OUTPUT" \
    --exclude="$PROJ_NAME/build" \
    --exclude="$PROJ_NAME/venv" \
    --exclude="$PROJ_NAME/othello_train_v2" \
    "$PROJ_NAME"

echo "Done: $(du -sh $OUTPUT | cut -f1)"
echo ""
echo "Upload:  scp $OUTPUT user@host:~/"
echo "Extract: tar -xzf $OUTPUT"
echo "Setup:   cd $PROJ && bash scripts/cloud_setup.sh"
