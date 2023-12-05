install:
	poetry install

export_poetry:
	poetry export -o requirements.txt --without-hashes

auto_commit:
	git add . && git commit -m 'Update' && git push
