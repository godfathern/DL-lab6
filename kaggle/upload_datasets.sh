#!/usr/bin/env bash
# Stage and upload the two private Kaggle datasets used by kaggle/kernel/run.py.
#   ./kaggle/upload_datasets.sh data    upload packed images + evaluator files (first time)
#   ./kaggle/upload_datasets.sh code    upload or update the src/*.py files
set -euo pipefail
cd "$(dirname "$0")/.."
USER_SLUG=phantungbui
STAGE=data/kaggle_stage

stage() {  # stage <slug> <title>
  rm -rf "$STAGE/$1" && mkdir -p "$STAGE/$1"
  cat > "$STAGE/$1/dataset-metadata.json" <<JSON
{"title": "$2", "id": "$USER_SLUG/$1", "licenses": [{"name": "unknown"}]}
JSON
}

push() {  # push <slug> <message>; creates the dataset (private) or adds a version
  if kaggle datasets status "$USER_SLUG/$1" >/dev/null 2>&1; then
    kaggle datasets version -p "$STAGE/$1" -m "$2" -r zip
  else
    kaggle datasets create -p "$STAGE/$1" -r zip
  fi
}

case "${1:-}" in
  data)
    stage lab6-ddpm-data lab6-ddpm-data
    cp data/iclevr_64.npz file/evaluator.py file/checkpoint.pth file/objects.json \
       file/test.json file/new_test.json "$STAGE/lab6-ddpm-data/"
    push lab6-ddpm-data "packed images and evaluator files"
    ;;
  code)
    stage lab6-ddpm-code lab6-ddpm-code
    cp src/*.py "$STAGE/lab6-ddpm-code/"
    push lab6-ddpm-code "update code"
    ;;
  *) echo "usage: $0 data|code" >&2; exit 1 ;;
esac
