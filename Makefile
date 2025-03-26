SRCDIR = src
DATADIR = data

train	:
	python3 $(SRCDIR)/train.py $(DATADIR)/data.csv

predict	:
	python3 $(SRCDIR)/predict.py

analyze	:
	python3 $(SRCDIR)/analyze.py $(DATADIR)/data.csv

test	:
	python3 $(SRCDIR)/test.py $(DATADIR)/data.csv

setup	:
	pip install -r requirements.txt