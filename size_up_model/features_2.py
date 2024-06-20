from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from utils.data_utils import load_data_from_gcs, save_model_to_gcs, save_data_to_gcs

def build_features():
    customer_timeline = load_data_from_gcs('cot-dev-sizeup-model-storage', 'processed/customer_timeline.csv')

    # Create target variables
    # customer_timeline['size_change_in_28_days'] = (
    #         customer_timeline.groupby('SHOPIFY_CUSTOMER_ID')['size_change'].rolling(28,
    #                                                                                 min_periods=1).sum().reset_index(0,
    #                                                                                                                  drop=True) > 0
    # ).astype(int)

    customer_timeline['next_size_product'] = (
            customer_timeline['next_size'].fillna('').replace('nan', '').astype(str) + '_' +
            customer_timeline['next_product'].fillna('').replace('nan', '').astype(str)
    )
    customer_timeline.loc[customer_timeline['size_change'] == 0, 'next_size_product'] = 'No_Change'

    # Select and encode categorical features
    features = ['order_count_in_size', 'cumulative_time_in_size', 'PRODUCT_SIZE', 'PRODUCT_CATEGORY']

    label_encoder = LabelEncoder()

    customer_timeline['PRODUCT_SIZE'] = label_encoder.fit_transform(customer_timeline['PRODUCT_SIZE'])
    customer_timeline['PRODUCT_CATEGORY'] = label_encoder.fit_transform(customer_timeline['PRODUCT_CATEGORY'])
    customer_timeline['next_size_product'] = label_encoder.fit_transform(customer_timeline['next_size_product'])

    # Save the fitted LabelEncoder for next_size_product
    # joblib.dump(label_encoder, "../models/next_size_product_encoder.pkl")
    save_model_to_gcs(label_encoder, 'cot-dev-sizeup-model-storage', 'models/next_size_product_encoder.pkl')

    # Impute null or missing values
    imputer = SimpleImputer(strategy='mean')
    customer_timeline[features] = imputer.fit_transform(customer_timeline[features])

    # Save updated customer timeline to csv
    # customer_timeline.to_csv(customer_timeline_path)
    save_data_to_gcs(customer_timeline, 'cot-dev-sizeup-model-storage', 'processed/customer_timeline.csv')

    # Save feature matrix and target variables
    # customer_timeline[features + ['size_change', 'next_size_product']].to_csv(
    #     features_and_targets_path, index=False)
    save_data_to_gcs(customer_timeline[features + ['size_change', 'next_size_product']], 'cot-dev-sizeup-model-storage', 'processed/features_and_targets.csv')

if __name__ == "__main__":
    build_features()