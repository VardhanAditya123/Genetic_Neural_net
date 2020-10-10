
.PHONY: all
all: git-commit run

.PHONY: git-commit
git-commit:
		git checkout master 
		git add lab2.py Makefile 
		git commit -a -m 'Commit' 
		git push origin -f master

.PHONY: run
run:
		python -W ignore lab2.py