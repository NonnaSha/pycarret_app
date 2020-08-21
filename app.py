from flask_ngrok import run_with_ngrok
from flask import Flask, request, url_for, redirect, render_template, jsonify
from pycaret.regression import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

run_with_ngrok(app)   

model = load_model('deployment')
cols = ['Height']

@app.route("/")
def home():
   return render_template("home.html")

@app.route("/predict", methods = ['POST'])
def predict():
  features = float(request.form['Height'])
  final = np.array(features)
  data_unseen = pd.DataFrame([final], columns = cols)
  prediction = predict_model(model, data = data_unseen, round = 0)
  prediction = prediction.Label[0]
  
  return render_template('home.html', pred = 'Prediction weight will be {} kg'.format(prediction))


if __name__ == '__main__':
    app.run()