.PHONY: test

test:
	nosetests -v sourcetracker sourcetracker/* --with-coverage --cover-package=sourcetracker
