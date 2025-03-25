train	:
	python3 train.py data/data.csv

predict	:
	python3 predict.py

setup	:
	pip install -r requirements.txt