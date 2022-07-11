#!/bin/sh
#chmod +x bootstrap.sh
#arrumar:
export FLASK_APP=/home/jaco/Projetos/realDeployML/realdeployml/tools/bareBones/app.py
#source $(pipenv --venv)/bin/activate
flask run -h 0.0.0.0