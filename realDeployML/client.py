import requests
import json

endpoint = "http://localhost:8501/v1/models/rankingv1:predict"

###read from csv: get all unique ids and movies####
###################################################

user_id = '"42"'
movie_title = '"Speed (1994)"'
stringfinal = f'"user_id": {user_id},"movie_title":{movie_title}'
data = json.loads('{"instances": [{ '+stringfinal+' }]}')
#data2 = json.loads('{"instances": [{"user_id": "42","movie_title": "Speed (1994)"}]}')
response = requests.post(endpoint, data=json.dumps(data))
prediction = response.json()['predictions'][0][0]
print(prediction)