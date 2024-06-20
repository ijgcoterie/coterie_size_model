# dataset.py
import datetime
from pathlib import Path
import pandas as pd
from loguru import logger
from datetime import datetime
from utils.data_utils import load_data_from_gcs, save_data_to_gcs

def preprocess_data():
    reference_date = datetime.now()

    # Load orders raw data from GCS
    orders_df = load_data_from_gcs('cot-dev-sizeup-model-storage', 'raw/ML_ORDERS.csv')

    # Convert date columns to datetime
    date_columns = {
        'orders_df': ['CREATED_AT'],
        # 'quiz_df': ['EVENT_DATE', 'UPDATED_AT', 'DUE_DATE', 'BIRTHDATE'],
        # 'subscriptions_df': ['CREATED_AT', 'CANCELLED_AT', 'FIRST_ORDER_AT'],
        # 'update_events_df': ['EVENT_DATE']
    }

    for df_name, columns in date_columns.items():
        for col in columns:
            if col in locals()[df_name].columns:
                locals()[df_name][col] = pd.to_datetime(locals()[df_name][col])

    # Filter for auto-renew orders and anchor products
    orders_df = orders_df[
        (orders_df['ORDER_TYPE'] == 'Auto Renew') &
        (orders_df['PRODUCT_CATEGORY'].isin(['Diapers', 'Pants']))
    ]

    # Sort orders by customer and date
    customer_timeline = orders_df.sort_values(['SHOPIFY_CUSTOMER_ID', 'CREATED_AT'])

    # Keep only the first order line for each customer on each date.
    # I'm fine with the fact that this will exclude cases where multiple anchor products are purchased at once.
    customer_timeline = customer_timeline.groupby(['SHOPIFY_CUSTOMER_ID', 'CREATED_AT']).first().reset_index()

    # Calculate time since last order
    customer_timeline['prev_order_date'] = customer_timeline.groupby('SHOPIFY_CUSTOMER_ID')['CREATED_AT'].shift(1)
    customer_timeline['time_since_last_order'] = (
                customer_timeline['CREATED_AT'] - customer_timeline['prev_order_date']).dt.days

    # Identify size changes and product switches
    customer_timeline['next_size'] = customer_timeline.groupby('SHOPIFY_CUSTOMER_ID')['PRODUCT_SIZE'].shift(-1)
    customer_timeline['next_product'] = customer_timeline.groupby('SHOPIFY_CUSTOMER_ID')['PRODUCT_CATEGORY'].shift(-1)
    customer_timeline['next_order_date'] = customer_timeline.groupby('SHOPIFY_CUSTOMER_ID')['CREATED_AT'].shift(-1)

    customer_timeline['size_change'] = (
            (customer_timeline['next_size'] != customer_timeline['PRODUCT_SIZE']) &
            (customer_timeline['next_size'].notnull())
    ).astype(int)

    customer_timeline['product_switch'] = (
            (customer_timeline['next_product'] != customer_timeline['PRODUCT_CATEGORY']) &
            (customer_timeline['next_product'].notnull())
    ).astype(int)

    # Calculate time until next order
    customer_timeline['time_to_next_order'] = (
                customer_timeline['next_order_date'] - customer_timeline['CREATED_AT']).dt.days

    # For the last order of each size or product, assume the customer stayed for 28 days
    last_size_order_mask = customer_timeline.groupby(['SHOPIFY_CUSTOMER_ID', 'PRODUCT_SIZE'])['CREATED_AT'].transform(
        'max') == customer_timeline['CREATED_AT']
    last_product_order_mask = customer_timeline.groupby(['SHOPIFY_CUSTOMER_ID', 'PRODUCT_CATEGORY'])[
                                  'CREATED_AT'].transform('max') == customer_timeline['CREATED_AT']
    customer_timeline.loc[last_size_order_mask, 'time_to_next_order'] = 28
    customer_timeline.loc[last_product_order_mask, 'time_to_next_order'] = 28

    # Calculate cumulative time spent in each size and product
    customer_timeline['cumulative_time_in_size'] = customer_timeline.groupby(['SHOPIFY_CUSTOMER_ID', 'PRODUCT_SIZE'])[
        'time_to_next_order'].cumsum()
    customer_timeline['cumulative_time_in_product'] = \
    customer_timeline.groupby(['SHOPIFY_CUSTOMER_ID', 'PRODUCT_CATEGORY'])['time_to_next_order'].cumsum()

    # Calculate the number of orders for each size and product
    customer_timeline['order_count_in_size'] = customer_timeline.groupby(
        ['SHOPIFY_CUSTOMER_ID', 'PRODUCT_SIZE']).cumcount() + 1
    customer_timeline['order_count_in_product'] = customer_timeline.groupby(
        ['SHOPIFY_CUSTOMER_ID', 'PRODUCT_CATEGORY']).cumcount() + 1

    # Calculate customer-level statistics
    customer_stats = customer_timeline.groupby('SHOPIFY_CUSTOMER_ID').agg({
        'SHOPIFY_ORDER_ID': 'count',
        'PRODUCT_SIZE': 'nunique',
        'PRODUCT_CATEGORY': 'nunique',
        'size_change': 'sum',
        'product_switch': 'sum'
    }).rename(columns={
        'SHOPIFY_ORDER_ID': 'total_orders',
        'PRODUCT_SIZE': 'unique_sizes',
        'PRODUCT_CATEGORY': 'unique_products',
        'size_change': 'total_size_changes',
        'product_switch': 'total_product_switches'
    })

    # Merge customer stats back to the timeline
    customer_timeline = customer_timeline.merge(customer_stats, on='SHOPIFY_CUSTOMER_ID', suffixes=('', '_total'))

    # Save processed data to GCS
    save_data_to_gcs(customer_timeline, 'cot-dev-sizeup-model-storage', 'processed/customer_timeline.csv')
    logger.info(f"Processed orders data saved to GCS: cot-dev-sizeup-model-storage/processed/customer_timeline.csv")

if __name__ == "__main__":
    preprocess_data()