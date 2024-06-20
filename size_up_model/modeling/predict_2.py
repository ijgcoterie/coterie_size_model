import pandas as pd
from utils.data_utils import load_data_from_gcs, load_model_from_gcs

def predict_for_customer(customer_id):
    customer_timeline = load_data_from_gcs('cot-dev-sizeup-model-storage', 'processed/customer_timeline.csv')

    # Get the customer's most recent entry
    customer_data = customer_timeline[customer_timeline['SHOPIFY_CUSTOMER_ID'] == customer_id].iloc[-1]

    # Create features representing the customer's current state
    current_features = {
        'order_count_in_size': customer_data['order_count_in_size'],
        'cumulative_time_in_size': customer_data['cumulative_time_in_size'],
        'PRODUCT_SIZE': customer_data['PRODUCT_SIZE'],
        'PRODUCT_CATEGORY': customer_data['PRODUCT_CATEGORY']
    }

    # Convert the current features to a DataFrame
    current_features_df = pd.DataFrame([current_features])

    model = load_model_from_gcs('cot-dev-sizeup-model-storage', 'models/size_up_model.pkl')
    predictions = model.predict(current_features_df)

    size_change_prediction = predictions[0][0]
    next_size_product_encoded = predictions[0][1]

    # Load the saved LabelEncoder for next_size_product
    label_encoder = load_model_from_gcs('cot-dev-sizeup-model-storage', 'models/next_size_product_encoder.pkl')

    # Decode the predicted next size and product
    next_size_product_prediction = label_encoder.inverse_transform([next_size_product_encoded])[0]

    prediction_result = {
        "customer_id": customer_id,
        "size_change_prediction": bool(size_change_prediction),
        "next_size_product": next_size_product_prediction
    }

    return prediction_result

if __name__ == "__main__":
    customer_id = 6739597033666 #6739597033666  # Replace with the desired customer ID
    prediction = predict_for_customer(customer_id)
    if prediction:
        print(prediction)
    else:
        print("Prediction failed. Please check the logs for more information.")