# Build the mini-project synopsis.
#   make            -> synopsis.pdf
#   make clean      -> remove LaTeX intermediates and the PDF

PDF = synopsis.pdf
TEX = synopsis.tex

.PHONY: all clean

all: $(PDF)

$(PDF): $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX)

clean:
	rm -f synopsis.aux synopsis.log synopsis.out synopsis.toc synopsis.pdf
