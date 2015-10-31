all: doc test

doc:
	make -C docs html

test:
	python -m doctest -v stamilarity/stamilarity.py

