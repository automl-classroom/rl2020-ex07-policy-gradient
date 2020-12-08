PYTHONPATH := $(PYTHONPATH):./src:./tests
export PYTHONPATH

all: init test

init:
	pip install -r requirements.txt

test:
	pytest tests

clean:
	rm *.log *.aux *.gz *.out *.synctex.gz

tex:
	pdflatex *.tex

.PHONY: init test all clean tex
.DEFAULT_GOAL := test
