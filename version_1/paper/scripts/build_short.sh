#!/bin/sh
# Build the short paper (paper/short/main.pdf) with tectonic. Usage: sh paper/scripts/build_short.sh [path-to-tectonic]
# tectonic runs bibtex itself; first run downloads TeX support files. Any XeLaTeX/pdfLaTeX + bibtex works as well.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SHORT="$HERE/../short"
TECT="${1:-tectonic}"
cd "$SHORT"
"$TECT" -X compile main.tex
echo "built $SHORT/main.pdf"
