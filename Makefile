all: doc test lint

doc:
	make -C docs html

test:
	python -m doctest -v stamilarity/stamilarity.py

lint:
	flake8 stamilarity/stamilarity.py

upload: doc test lint
	git push
	python setup.py sdist upload -r pypi
