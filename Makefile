init:
	pip install -r requirements.txt

test:
	./dsolver.py

.PHONY: init test
