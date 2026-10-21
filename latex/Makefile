# Build the mini-project synopsis.
#   make            -> synopsis.pdf
#   make clean      -> remove LaTeX intermediates and the PDF

PDF = report.pdf
TEX = report.tex

.PHONY: all clean

all: $(PDF)

$(PDF): $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error $(TEX)

clean:
	rm -f report.aux report.log report.out report.toc report.pdf
