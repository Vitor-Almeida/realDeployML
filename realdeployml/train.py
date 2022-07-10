import os
from config.definitions import ROOT_DIR
from typing import Dict, Text
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds

class RankingModel(tf.keras.Model):

  def __init__(self,unique_movie_titles,unique_user_ids):
    super().__init__()
    embedding_dimension = 32

    # Compute embeddings for users.
    self.user_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    # Compute embeddings for movies.
    self.movie_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_movie_titles, mask_token=None),
      tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
    ])

    # Compute predictions.
    self.ratings = tf.keras.Sequential([
      # Learn multiple dense layers.
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      # Make rating predictions in the final layer.
      tf.keras.layers.Dense(1)
  ])

  def call(self, inputs):

    user_id, movie_title = inputs

    user_embedding = self.user_embeddings(user_id)
    movie_embedding = self.movie_embeddings(movie_title)

    return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))

class MovielensModel(tfrs.models.Model):

  def __init__(self,unique_movie_titles,unique_user_ids):
    super().__init__()
    self.ranking_model: tf.keras.Model = RankingModel(unique_movie_titles,unique_user_ids)
    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
      loss = tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    return self.ranking_model(
        (features["user_id"], features["movie_title"]))

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    labels = features.pop("user_rating")

    rating_predictions = self(features)

    # The task computes the loss and the metrics.
    return self.task(labels=labels, predictions=rating_predictions)

class fitEvaluate():

  def __init__(self,train,test,unique_movie_titles,unique_user_ids):
    super().__init__()
    self.model = MovielensModel(unique_movie_titles,unique_user_ids)

    self.model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))    
    cached_train = train.shuffle(100000).batch(8192).cache()
    cached_test = test.batch(4096).cache()

    #ml flow:

    self.model.fit(cached_train, epochs=3)
    self.model.evaluate(cached_test, return_dict=True)
    
def getData():

    ratings = tfds.load("movielens/100k-ratings", split="train",data_dir=os.path.join(ROOT_DIR, 'data','movielens','ratings'))
    #movies = tfds.load("movielens/100k-movies", split="train",data_dir=os.path.join(ROOT_DIR, 'data','movielens','movies'))

    ratings = ratings.map(lambda x: {
        "movie_title": x["movie_title"],
        "user_id": x["user_id"],
        "user_rating": x["user_rating"]
    })
    tf.random.set_seed(42)
    shuffled = ratings.shuffle(100000, seed=42, reshuffle_each_iteration=False)
    train = shuffled.take(80000)
    test = shuffled.skip(80000).take(20000)
    movie_titles = ratings.batch(1000000).map(lambda x: x["movie_title"])
    user_ids = ratings.batch(1000000).map(lambda x: x["user_id"])
    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))
    
    return train,test,unique_movie_titles,unique_user_ids

def getPredict(model,id,test_movie_titles):
  test_ratings = {}
  for movie_title in test_movie_titles:
    test_ratings[movie_title] = model({
        "user_id": np.array(id),
        "movie_title": np.array([movie_title])
    })

  print("Ratings:")
  for title, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
    print(f"{title}: {score}")

  return None

def main():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    train,test,unique_movie_titles,unique_user_ids = getData()

    titlesExport = [title.decode('UTF-8') for title in np.array(unique_movie_titles)]

    np.savetxt(os.path.join(ROOT_DIR, 'realdeployml','models','rankingv1','1','allmovietitles.csv'), np.array([titlesExport]), delimiter="||", fmt='%s')

    model = fitEvaluate(train,test,unique_movie_titles,unique_user_ids)

    tf.saved_model.save(model.model, os.path.join(ROOT_DIR, 'realdeployml','models','rankingv1','1'))

    #debugger:
    #getPredict(model.model,['42'],["M*A*S*H (1970)", "Dances with Wolves (1990)", "Speed (1994)"])

    return None

if __name__ == '__main__':
    main()