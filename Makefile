init:
	pip install -r requirements.txt

test:
	$(py) unittest/test_util.py -v
	$(py) unittest/test_dispersion_tensor.py -v
	$(py) unittest/test_isotropic.py -v
	$(py) unittest/test_parallel.py -v

.PHONY: init test
