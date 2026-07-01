all: main.pdf

main.pdf: main.tex references.bib colm2026_conference.sty colm2026_conference.bst
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

clean:
	del /Q main.aux main.bbl main.blg main.fdb_latexmk main.fls main.log main.out main.pdf 2>NUL

.PHONY: all clean
