train	:
	python3 src/train.py data/data.csv

predict	:
	python3 src/predict.py

setup	:
	pip install -r requirements.txt