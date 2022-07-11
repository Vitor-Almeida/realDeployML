import tensorflow as tf
import os
import numpy as np
from flask import Flask, request, jsonify
#from config.definitions import ROOT_DIR ?

#http://127.0.0.1:5000/?id=10
#FLASK_APP=app.py flask run

app = Flask(__name__)

def getPredict(id):

  ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__),'..','..','..')) #como arrumar isso aqui?
  #ROOT_DIR="/home/jaco/Projetos/realDeployML" ?
  model_path = os.path.join(ROOT_DIR, 'realdeployml','models','rankingv1','1')

  model = tf.saved_model.load(model_path)

  #save at tmp, if tmp, dont load.
  test_movie_titles = np.genfromtxt(os.path.join(model_path,'allmovietitles.csv'), delimiter='||',dtype=str)
  test_movie_titles = test_movie_titles.tolist()
  test_movie_titles = test_movie_titles[:3]

  test_ratings = {}
  for movie_title in test_movie_titles:
    test_ratings[movie_title] = model({
        "user_id": np.array([id]),
        "movie_title": np.array([movie_title])
    })

  title_list = []
  score_list = []

  for title, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
    title_list.append(title)
    score_list.append(score)

  scoredMovies = {
      'response':{
        'id':id,
        'titles':np.array(title_list).tolist(),
        'scores':np.array(score_list).flatten().tolist()
      }
  }

  return jsonify(scoredMovies)

@app.route('/', methods=['GET'])
def predict():
    id = request.args.get('id')
    scoredMovies = getPredict(id)
    return scoredMovies

