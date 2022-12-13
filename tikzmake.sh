#!/bin/bash

set -e

python $1.py 
#pdflatex $1.tex
xelatex $1.tex

set +e

rm *.aux *.log *.vscodeLog
rm *.tex

#pdftoppm -png $1.pdf > $1.png
if [[ "$OSTYPE" == "darwin"* ]]; then
    open $1.pdf
else
    xdg-open $1.pdf
fi
