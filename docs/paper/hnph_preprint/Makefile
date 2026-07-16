REPO_ROOT := ../../..
PROTOCOL := $(REPO_ROOT)/configs/protocol/hnph_phase0_v0.4.yaml
FIGURE_SCRIPT := $(REPO_ROOT)/scripts/analysis/plot_hnph_preprint_figures.py
FIGURE_DIR := $(REPO_ROOT)/docs/figures/hnph_protocol
PAPER := kahlus_hnph_preprint

.PHONY: all figures pdf clean

all: figures pdf

figures:
	python $(FIGURE_SCRIPT) --protocol $(PROTOCOL) --out-dir $(FIGURE_DIR)

pdf: figures
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(PAPER).tex

clean:
	latexmk -c $(PAPER).tex
