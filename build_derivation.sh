#!/bin/sh
# Build derivation.pdf and report errors / undefined references / undefined citations.
#
# Two bibtex passes are required: derivationNotes.bib stores footnote text that
# itself contains \cite commands, so the citations inside the notes only reach
# the .aux after the .bbl has been typeset once.
set -e
cd "$(dirname "$0")"
LOG=.buildlog
pdflatex -interaction=nonstopmode derivation.tex > $LOG.1 2>&1 || true
bibtex derivation                                > $LOG.b1 2>&1 || true
pdflatex -interaction=nonstopmode derivation.tex > $LOG.2 2>&1 || true
bibtex derivation                                > $LOG.b2 2>&1 || true
pdflatex -interaction=nonstopmode derivation.tex > $LOG.3 2>&1 || true
pdflatex -interaction=nonstopmode derivation.tex > $LOG.4 2>&1 || true

echo "--- errors ---"
grep -n '^! ' $LOG.4 || echo "none"
echo "--- undefined references ---"
grep -n 'Reference .* undefined' $LOG.4 || echo "none"
echo "--- undefined citations ---"
grep -n 'Citation .* undefined' $LOG.4 || echo "none"
echo "--- multiply-defined ---"
grep -n 'multiply defined' $LOG.4 || echo "none"
echo "--- pages ---"
grep -o 'Output written on derivation.pdf ([0-9]* pages' $LOG.4 || true
