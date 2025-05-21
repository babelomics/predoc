#!make
include .env
.PHONY: install test format
.ONESHELL:

SHELL = /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate

all: install format test
install:
	conda env update --prune -p ${ENV_FOLDER} -f environment.yml
	$(CONDA_ACTIVATE) ${ENV_FOLDER}
	pip install -e .
format:
	$(CONDA_ACTIVATE) ${ENV_FOLDER}
	autoflake -r --in-place --remove-unused-variables --remove-all-unused-imports predoc
	isort --profile black predoc tests
	black predoc tests
#	(cd docs && poetry run make html)
test:
	$(CONDA_ACTIVATE) ${ENV_FOLDER}
	pytest
