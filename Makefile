.PHONY: test

test:
	nosetests -v -s sourcetracker --with-coverage --cover-package=sourcetracker

