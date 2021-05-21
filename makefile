install:
	pip install --upgrade pip && pip install -r requirements.txt

auto_commit:
	git add . && git commit -m 'Update' && git push
