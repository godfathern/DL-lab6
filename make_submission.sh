#!/usr/bin/env bash
# Build the Lab 6 submission zip in the required layout.
#   ./make_submission.sh <studentID> <YOUR_NAME>
# Expects report/report.pdf (with your name, ID, and screenshots filled in)
# and the generated images in outputs/final/images/{test,new_test}/.
set -euo pipefail
cd "$(dirname "$0")"
[ $# -eq 2 ] || { echo "usage: $0 <studentID> <YOUR_NAME>" >&2; exit 1; }
NAME="DL_LAB6_$1_$2"
STAGE="$(mktemp -d)/$NAME"
mkdir -p "$STAGE/images"

cp report/report.pdf "$STAGE/$NAME.pdf"
cp -R src "$STAGE/src"
cp -R kaggle "$STAGE/kaggle"
cp README.md "$STAGE/"
find "$STAGE" -name '__pycache__' -prune -exec rm -rf {} +
for split in test new_test; do
  mkdir -p "$STAGE/images/$split"
  for i in $(seq 0 31); do cp "outputs/final/images/$split/$i.png" "$STAGE/images/$split/"; done
done

# The lab forbids including the training data or the evaluator checkpoint.
if find "$STAGE" \( -name '*.pth' -o -name '*.pt' -o -name '*.npz' -o -name 'CLEVR_*' \) | grep -q .; then
  echo "refusing: dataset or checkpoint files found in the archive" >&2; exit 1
fi

rm -f "$NAME.zip"
OUT="$PWD/$NAME.zip"
# Files sit at the archive root, as in the lab's required structure.
(cd "$STAGE" && zip -qr "$OUT" .)
echo "wrote $NAME.zip"
unzip -l "$NAME.zip" | tail -1
