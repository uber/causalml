.PHONY: build_ext
build_ext: clean
	python setup.py build_ext --force --inplace

.PHONY: build
build: build_ext
	python setup.py bdist_wheel

.PHONY: install
install: build_ext
	pip install .

.PHONY: test
test: build_ext
	pytest -vs --cov causalml/
	python setup.py clean --all

.PHONY: clean
clean:
	python setup.py clean --all
	rm -rf ./build ./dist ./eggs ./causalml.egg-info
	find ./causalml -type f \( -name "*.so" -o -name "*.c" -o -name "*.html" \) -delete

.PHONY: setup_local
setup_local:
	pip install pre-commit
	pre-commit install
