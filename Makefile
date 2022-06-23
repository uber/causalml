build:
	python setup.py bdist_wheel

install:
	pip install .

test:
	python setup.py clean --all build_ext --force --inplace
	pytest -vs --cov causalml/
	python setup.py clean --all

clean:
	python setup.py clean --all






