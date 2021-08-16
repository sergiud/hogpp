#!/bin/sh
biber --tool \
    --isbn-normalise \
    --output-align \
    --output-encoding=ascii \
    --output-fieldcase=lower \
    --output-format=bibtex \
    --output-indent=2 \
    $@
