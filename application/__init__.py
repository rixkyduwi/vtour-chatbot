from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy
import tflearn
import tensorflow
import nltk,pickle,json,random;#nltk.download('popular')
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('application/content.json').read())
UPLOAD_FOLDER = 'application/static/upload/'
app = Flask(__name__)
from dotenv import load_dotenv
load_dotenv()
import os
SECRET_KEY = os.getenv("MY_SECRET")
app.config.update(dict(
SECRET_KEY="powerful secretkey",
WTF_CSRF_SECRET_KEY="dudu rohosio"
    ))
CORS(app)
@app.route('/')
def index():
  return "<h1>hello world</h1>"
@app.route('/json')
def defjson():
  return jsonify({"status":200,"description":"hello world"})
@app.route('/get')
def method_name():
  userText = request.args.get('msg')
  return chat(userText)
try:
  with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)
except:
  words = []
  labels = []
  docs_x = []
  docs_y = []
with open('application/content.json') as user_file:
  data = json.load(user_file)
with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load('model.tflearn')
def bag_of_words(s, words):
  bag = [0 for _ in range(len(words))]
  s_words = nltk.word_tokenize(s)
  s_words = [stemmer.stem(word.lower()) for word in s_words]

  for se in s_words:
    for i, w in enumerate(words):
      if w == se:
        bag[i] = 1

  return numpy.array(bag)

def chat(msg):
    results = model.predict([bag_of_words(msg, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    print(tag)
    for tg in data["intents"]:
      if tg['tag'] == tag:
        responses = tg['responses']
    print(responses)
    return responses

