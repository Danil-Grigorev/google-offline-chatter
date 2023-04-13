
virtualenv:
	virtualenv -p python3.9 .venv

install: virtualenv
	sh -c "source .venv/bin/activate && pip install -r requirements.txt"

