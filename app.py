
from flask import Flask, jsonify, request, render_template, redirect
import joblib
import socket
import json
import pandas as pd
import os
import sys
import requests
from model import model_predict, model_train

app = Flask(__name__)

@app.route("/")
def hello():
    html = "<h3>Hello!</h3>" \
           
    return html    

@app.route('/predict', methods=['GET'])
def predict_form():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form["Date"]
    year = text.split('-')[0]
    month = text.split('-')[1]
    date = text.split('-')[2]
    country = request.form["Country"]
    prediction = model_predict(country, year, month, date)
    prediction_jsonify = prediction['y_pred'].tolist()[0]
    print(prediction_jsonify)
    output_text = country+": Predicted Forecast for 30 day period on "+text+" is: "+str(round(prediction_jsonify, 2))
    print(output_text)
    # return jsonify(prediction['y_pred'].tolist())
    # return jsonify(prediction['y_pred'].tolist())
    return jsonify(output_text)


if __name__ == '__main__':
    saved_model = 'models/model_all.joblib'
    model = joblib.load(saved_model)
    app.run(host='0.0.0.0', port=8080,debug=True)
