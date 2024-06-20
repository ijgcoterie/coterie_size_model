from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from utils.data_utils import load_data_from_gcs, save_model_to_gcs

def train_model():
    data = load_data_from_gcs('cot-dev-sizeup-model-storage', 'processed/features_and_targets.csv')

    # old_features = ['time_since_last_order', 'order_count_in_size', 'cumulative_time_in_size', 'total_orders', 'unique_sizes', 'total_size_changes', 'PRODUCT_SIZE', 'PRODUCT_CATEGORY']
    features = ['order_count_in_size', 'cumulative_time_in_size','PRODUCT_SIZE', 'PRODUCT_CATEGORY']
    targets = ['size_change', 'next_size_product']

    X_train, X_test, y_train, y_test = train_test_split(data[features], data[targets], test_size=0.2, random_state=42)

    model = MultiOutputClassifier(RandomForestClassifier(random_state=42))
    model.fit(X_train, y_train)

    # Save model to GCS
    save_model_to_gcs(model, 'cot-dev-sizeup-model-storage', 'models/size_up_model.pkl')

if __name__ == "__main__":
    train_model()