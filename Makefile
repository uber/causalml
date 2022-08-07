.PHONY: build
build: clean
	python setup.py build_ext --force --inplace
	python setup.py bdist_wheel

.PHONY: install
install:
	pip install .

.PHONY: test
test:
	python setup.py clean --all build_ext --force --inplace
	pytest -vs --cov causalml/
	python setup.py clean --all

.PHONY: clean
clean:
	python setup.py clean --all
	rm -rf ./build ./dist ./causalml.egg-info





