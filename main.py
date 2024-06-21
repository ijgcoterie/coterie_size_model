import numpy as np
from flask import Flask, request, jsonify
from size_up_model.modeling.predict import predict_for_customer, load_resources

app = Flask(__name__)

# Load resources at startup
customer_timeline, model, label_encoder = load_resources()

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

    prediction = predict_for_customer(customer_id, customer_timeline, model, label_encoder)

    return jsonify(prediction)

if __name__ == "__main__":
    app.run(port=8080)
