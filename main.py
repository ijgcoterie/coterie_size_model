import numpy as np
from flask import Flask, request
from size_up_model.modeling.predict import predict_for_customer

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    """Basic HTML response."""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to my Flask API</h1>"
        "</body>"
        "</html>"
    )
    return body

@app.route("/predict", methods=["POST"])
def predict():
    data_json = request.get_json()

    customer_id = data_json["customer_id"]

    prediction = predict_for_customer(customer_id)

    return prediction

if __name__ == "__main__":
    app.run(port=8080)