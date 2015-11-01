all: doc test lint

doc:
	make -C docs html

test:
	python -m doctest -v stamilarity/__init__.py

lint:
	flake8 stamilarity/__init__.py

upload: doc test lint
	git push
	python setup.py sdist upload -r pypi
