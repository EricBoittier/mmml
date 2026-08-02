# Build ACS-style hybrid Methods PDF with tectonic (conda-forge).
TECTONIC ?= tectonic
MAIN := main.tex

.PHONY: all pdf clean

all: pdf

pdf:
	$(TECTONIC) -X compile $(MAIN)

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out main.synctex.gz
