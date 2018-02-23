from flask import Flask
from flask import request
import predict
from flask import jsonify
from flask import Response
app = Flask(__name__)

#http://flask.pocoo.org/docs/0.12/quickstart/#accessing-request-data
@app.route('/predict', methods=['POST'])
def respond():
    paragraph = request.form['paragraph']
    question = request.form['question']
    prediction = predict.get_answer(paragraph, question)
    return Response(prediction, mimetype="text/plain")

#https://stackoverflow.com/questions/20646822/how-to-serve-static-files-in-flask
@app.route('/qa')
def root():
    return app.send_static_file('home.html')

@app.route('/about')
def about():
    return app.send_static_file('info.html')