# disable builtin and suffix rules
MAKEFLAGS += --no-builtin-rules
.SUFFIXES:

.PHONY: help clean deps lint test coverage docs release install jenkins

help:
	@echo "clean - remove all build, test, coverage and Python artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-test - remove test and coverage artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "deps - force update of requirements specs"
	@echo "release - package and upload a release"
	@echo "install - install the package to the active Python's site-packages"

env:
	virtualenv --setuptools $@
	$@/bin/pip install -U "setuptools>=19,<20"
	$@/bin/pip install -U "pip>=7,<8"
	$@/bin/pip install -U "pip-tools==1.6.4"

# sentinel file to ensure installed requirements match current specs
env/.requirements: requirements.txt requirements-test.txt docs/requirements-docs.txt | env
	$|/bin/pip-sync $^
	touch $@

clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -fr htmlcov/

lint: env/.requirements
	env/bin/flake8 hello-world tests

test: env/.requirements
	env/bin/python setup.py test $(TEST_ARGS)

# tests that emit artifacts jenkins/phabricator look for
jenkins: env/.requirements
	env/bin/py.test -s --tb short --cov-config .coveragerc --cov hello-world --cov-report term-missing --cov-report xml \
	    --junitxml junit.xml \
	    tests

coverage: env/.requirements test
	env/bin/coverage run --source hello-world setup.py test
	env/bin/coverage report -m
	env/bin/coverage html
	open htmlcov/index.html

docs: env/.requirements
	rm -f docs/hello-world.rst
	rm -f docs/modules.rst
	env/bin/sphinx-apidoc -o docs/ hello-world
	PATH=$(PATH):$(CURDIR)/env/bin $(MAKE) -C docs clean
	PATH=$(PATH):$(CURDIR)/env/bin $(MAKE) -C docs html
	# open docs/_build/html/index.html

release: env/.requirements clean
	env/bin/fullrelease

install: clean
	python setup.py install

deps:
	@touch requirements.in requirements-test.in docs/requirements-docs.in
	$(MAKE) requirements.txt requirements-test.txt docs/requirements-docs.txt

requirements.txt requirements-test.txt docs/requirements-docs.txt: %.txt: %.in | env
	$|/bin/pip-compile --no-index $^
