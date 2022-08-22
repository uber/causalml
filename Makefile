.PHONY: build_ext
build_ext: install_req clean
	python setup.py build_ext --force --inplace

.PHONY: install_req
install_req:
	pip install -r requirements.txt

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
	rm -rf ./build ./dist ./causalml.egg-info