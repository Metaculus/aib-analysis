#!/usr/bin/env bash
# Render the three survey-analysis markdown docs to self-contained HTML with Quarto.
# Run this after `python -m aib_analysis.survey_analysis_v2.run`.
#
# Cache dirs are redirected to a temp location so this works inside a sandbox
# where the default Quarto/Deno cache path is not writable.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$REPO_ROOT/local/spring_survey_analysis_v2"

CACHE_BASE="${TMPDIR:-/tmp}/quarto_survey_v2"
export XDG_CACHE_HOME="$CACHE_BASE/cache"
export DENO_DIR="$CACHE_BASE/deno"
export XDG_DATA_HOME="$CACHE_BASE/data"
mkdir -p "$XDG_CACHE_HOME" "$DENO_DIR" "$XDG_DATA_HOME"

cd "$OUT_DIR"
for doc in spring_survey_analysis parsing_decisions parsing_review; do
  echo "Rendering $doc.md -> $doc.html"
  quarto render "$doc.md" --to html --embed-resources -M toc=true -M toc-location=left
done
echo "Done. Open $OUT_DIR/index.html"
