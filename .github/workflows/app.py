# webapp/app.py

import os
import requests
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

AZUREML_ENDPOINT_URL = os.getenv("AZUREML_ENDPOINT_URL")
AZUREML_API_KEY = os.getenv("AZUREML_API_KEY")  # Must be set as a secret/env variable

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form.get("text")
    if not user_input:
        return render_template("index.html", prediction="Please enter some text.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZUREML_API_KEY}"
    }

    payload = {"input_data": [user_input]}

    try:
        response = requests.post(AZUREML_ENDPOINT_URL, headers=headers, json=payload)
        response.raise_for_status()
        prediction = response.json().get("predictions", ["No prediction"])[0]
        return render_template("index.html", prediction=prediction, input_text=user_input)
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}", input_text=user_input)

if __name__ == "__main__":
    app.run(debug=True)
