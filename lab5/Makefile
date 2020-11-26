
.PHONY: all
all: git-commit run

.PHONY: git-commit
git-commit:
		git checkout master >> .local.git.out || echo
		git add lab5 Makefile >> .local.git.out  || echo
		git commit -a -m 'Commit' >> .local.git.out || echo
		git push origin -f master

.PHONY: run
run:
		python -W ignore lab5.py