.PHONY: test

test:
	nosetests -v -s sourcetracker --cover-package=sourcetracker
	nosetests -v -s sourcetracker/_cli --cover-package=sourcetracker 
	nosetests -v -s sourcetracker/_q2 --cover-package=sourcetracker 
