from flask import Flask, request, jsonify
import numpy as np
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Load the model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def welcome():
    return "You Are Welcome to COVID-19 Detector."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the data from the POST request
        int_features = [int(x) for x in request.json.values()]
        features = [np.array(int_features)]

        # Make a prediction
        prediction = model.predict(features)[0]

        # Prepare the response
        if prediction == 1:
            response = {
                "prediction_text": "You are a Covid Patient, Kindly go to your nearby Hospital"}
        else:
            response = {
                "prediction_text": "You are not a Covid Patient, but kindly go out for necessary reasons."}
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
