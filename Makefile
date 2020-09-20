.PHONY: git-commit
git-commit:
		git checkout master >> .local.git.out || echo
		git add lab1.py Makefile >> .local.git.out  || echo
		git commit -a -m 'Commit' >> .local.git.out || echo
		git push origin master