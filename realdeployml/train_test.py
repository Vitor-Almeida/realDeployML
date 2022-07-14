from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
import tools._prefect.train_prefect as pre

@task
def save(model,unique_movie_titles,version):
  pre.saveFiles(model,unique_movie_titles,version)

@task
def fit(train,test,unique_movie_titles,unique_user_ids):
  model = pre.fitEvaluate(train,test,unique_movie_titles,unique_user_ids)
  return model
  
#@task
def data():
  return pre.getData()

@flow
def main():

    train,test,unique_movie_titles,unique_user_ids = data()
    model = fit(train,test,unique_movie_titles,unique_user_ids)
    save(model,unique_movie_titles,'1')

    return None

if __name__ == '__main__':
    main()