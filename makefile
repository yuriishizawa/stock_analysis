install:
	pip install --upgrade pip && pip install -r requirements.txt

export_poetry:
	poetry export -o requirements.txt --without-hashes

auto_commit:
	git add . && git commit -m 'Update' && git push
