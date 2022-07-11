##criar a imagem:
##https://www.pluralsight.com/guides/create-docker-images-docker-hub
##docker build --network=host -t baseimagedeploy .
##docker run -it --name baseimagedeploy_app baseimagedeploy
##docker commit baseimagedeploy_app baseimagedeploy:latest
##docker login
##docker tag baseimagedeploy:latest jacobiu/baseimagedeploy:latest
##docker push jacobiu/baseimagedeploy:latest
FROM python:3.9.7-slim-bullseye
RUN pip install --upgrade tensorflow
RUN pip install gunicorn
RUN pip install Flask