#!/usr/bin/env bash
# Render the three survey-analysis markdown docs to self-contained HTML (Quarto)
# and to PDF (headless Chrome printing the rendered HTML, so the PDF matches the
# HTML exactly, charts included). Run this after
# `python -m aib_analysis.survey_analysis.run`.
#
# Cache dirs are redirected to a temp location so this works inside a sandbox
# where the default Quarto/Deno cache path is not writable.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$REPO_ROOT/local/spring_survey_analysis"

CACHE_BASE="${TMPDIR:-/tmp}/quarto_survey_v2"
export XDG_CACHE_HOME="$CACHE_BASE/cache"
export DENO_DIR="$CACHE_BASE/deno"
export XDG_DATA_HOME="$CACHE_BASE/data"
mkdir -p "$XDG_CACHE_HOME" "$DENO_DIR" "$XDG_DATA_HOME"

# Locate a Chrome/Chromium binary for PDF export. PDF is skipped with a warning
# if none is found (the HTML is still produced).
CHROME_BIN=""
for candidate in google-chrome google-chrome-stable chromium chromium-browser; do
  if command -v "$candidate" >/dev/null 2>&1; then CHROME_BIN="$candidate"; break; fi
done
CHROME_PROFILE="$CACHE_BASE/chrome-profile"

cd "$OUT_DIR"
for doc in spring_survey_analysis parsing_decisions parsing_review; do
  echo "Rendering $doc.md -> $doc.html"
  quarto render "$doc.md" --to html --embed-resources -M toc=true -M toc-location=left
  if [ -n "$CHROME_BIN" ]; then
    echo "Printing $doc.html -> $doc.pdf"
    # A dedicated user-data-dir keeps this isolated from the user's own Chrome
    # profile; --no-pdf-header-footer drops the default date/URL print margins.
    if "$CHROME_BIN" --headless=new --no-sandbox --disable-gpu \
        --user-data-dir="$CHROME_PROFILE" --disable-crash-reporter --no-first-run \
        --no-pdf-header-footer \
        --print-to-pdf="$OUT_DIR/$doc.pdf" \
        "file://$OUT_DIR/$doc.html" >/dev/null 2>&1; then
      echo "  wrote $doc.pdf"
    else
      echo "  WARNING: PDF export failed for $doc (HTML still produced)"
    fi
  fi
done

if [ -z "$CHROME_BIN" ]; then
  echo "WARNING: no Chrome/Chromium found on PATH; skipped PDF export."
fi
echo "Done. Open $OUT_DIR/index.html"
