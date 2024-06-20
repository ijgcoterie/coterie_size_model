import io
import pandas as pd
import joblib
from google.cloud import storage

def load_data_from_gcs(bucket_name, file_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    data = blob.download_as_string()
    return pd.read_csv(io.BytesIO(data))

def save_data_to_gcs(data, bucket_name, file_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    blob.upload_from_string(data.to_csv(index=False), content_type='text/csv')

def save_model_to_gcs(model, bucket_name, model_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.upload_from_filename(model_path)

def load_model_from_gcs(bucket_name, model_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.download_to_filename('/tmp/local_model_path.pkl')
    loaded_model = joblib.load('/tmp/local_model_path.pkl')
    return loaded_model