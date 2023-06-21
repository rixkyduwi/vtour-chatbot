import os
import requests
from flask import Flask, render_template, url_for, request, redirect, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime
from dotenv import load_dotenv
from flask_mail import Mail, Message

load_dotenv()

app = Flask(__name__)
mail = Mail(app)

if app.config["DEBUG"] == True:
    app.config.from_object('config.DevConfig')
else:
    app.config.from_object('config.ProdConfig')

db = SQLAlchemy(app)
migrate = Migrate(app, db)

class Faq(db.Model):
    __tablename__ = 'faq'
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(150))
    answer = db.Column(db.String(100))
    timestamp = db.Column(db.String(100))

    def __init__(self, question, answer, timestamp):
        self.question = question
        self.answer = answer
        self.timestamp = timestamp

with app.app_context():
    db.create_all()

#index route
@app.route('/')
def index():
    data = Faq.query.all()
    return render_template('index.html', data=data)

#add faq
@app.route('/add-faq', methods=['GET', 'POST'])
def addfaq():
    if request.method == 'POST':
        question = request.form.get('question')
        answer = request.form.get('answer')
        timestamp = datetime.utcnow()
        
        data = Faq(question, answer, timestamp)
        db.session.add(data)
        db.session.commit()

        return redirect(url_for('index'))
    return render_template('add-faq.html')

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy
import tflearn
import tensorflow
import nltk,pickle,json,random;#nltk.download('popular')
#nltk.data.path.append('./nltk_data/')
#nltk.download("punkt", "application/nltk_data/")
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('application/content.json').read())
UPLOAD_FOLDER = 'application/static/upload/'
from dotenv import load_dotenv
load_dotenv()
import os
SECRET_KEY = os.getenv("MY_SECRET")
app.config.update(dict(
SECRET_KEY="powerful secretkey",
WTF_CSRF_SECRET_KEY="dudu rohosio"
    ))
CORS(app)
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

if __name__ == '__main__':
    app.run()