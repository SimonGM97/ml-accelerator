import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import os
from io import BytesIO, StringIO
import s3fs
import boto3
from botocore.exceptions import ClientError
import pickle
import json
import yaml


def get_secrets(secret_name: str = 'access_keys') -> str:
    # Create a Secrets Manager client
    session = boto3.session.Session()
    secrets_client = session.client(
        service_name='secretsmanager',
        region_name=REGION
    )

    try:
        get_secret_value_response = secrets_client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    secret = json.loads(get_secret_value_response['SecretString'])

    return secret

# Load config file
with open(os.path.join("config", "config.yaml")) as file:
    config: dict = yaml.load(file, Loader=yaml.FullLoader)

# Load region
REGION = config["ENV_PARAMS"]["REGION"]

# Extract secrets
ACCESS_KEYS = get_secrets(secret_name='access_keys')

# Create an S3 client instance
# print('Instanciating S3_CLIENT.\n')
S3_CLIENT = boto3.client(
    's3',
    region_name=REGION,
    aws_access_key_id=ACCESS_KEYS["AWS_ACCESS_KEY_ID"], # os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=ACCESS_KEYS["AWS_SECRET_ACCESS_KEY"], # os.environ.get("AWS_SECRET_ACCESS_KEY")
)


def load_from_s3(path: str):
    # Extract bucket, key & format
    bucket, key = path.split('/')[0], '/'.join(path.split('/')[1:])
    read_format = key.split('.')[-1]

    if read_format == 'csv':
        # Retrieve stored object
        obj: dict = S3_CLIENT.get_object(
            Bucket=bucket,
            Key=key
        )

        # Read csv
        asset: pd.DataFrame = pd.read_csv(
            StringIO(obj['Body'].read().decode('utf-8'))
        )

    elif read_format == 'pickle':
        # Retrieve stored object
        obj: dict = S3_CLIENT.get_object(
            Bucket=bucket,
            Key=key
        )

        # Read pickle
        asset: dict = pickle.loads(
            BytesIO(obj['Body'].read()).read()
        )
    elif read_format == 'json':
        # Retrieve stored object
        obj: dict = S3_CLIENT.get_object(
            Bucket=bucket,
            Key=key
        )

        # Read json
        asset: dict = json.loads(
            BytesIO(obj['Body'].read()).read()
        )
    else:
        raise Exception(f'Invalid "read_format" parameter: {read_format}, extracted from path: {path}.\n\n')
    
    # assert len(asset) > 0, f"Loaded asset from s3://{path} contains zero keys. {asset}"

    return asset


def save_to_s3(
    asset, 
    path: str,
    partition_column: str = None
):
    # Extract bucket, key & format
    bucket, key = path.split('/')[0], '/'.join(path.split('/')[1:])
    write_format = key.split('.')[-1]

    # Delete object from S3
    delete_from_s3(path=path)

    if write_format == 'csv':
        asset: pd.DataFrame = asset.copy(deep=True)

        # Convert DataFrame to CSV in memory (StringIO)
        csv_buffer = StringIO()
        asset.to_csv(csv_buffer, index=False)

        # Save new object
        S3_CLIENT.put_object(
            Bucket=bucket,
            Key=key,
            Body=csv_buffer.getvalue()
        )
    elif write_format == 'pickle':
        # Save new object
        S3_CLIENT.put_object(
            Bucket=bucket,
            Key=key,
            Body=pickle.dumps(asset)
        )
    elif write_format == 'json':
        # Save new object
        S3_CLIENT.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(asset)
        )
    else:
        raise Exception(f'Invalid "write_format" parameter: {write_format}, extracted from path: {path}.\n\n')
    

def delete_from_s3(
    path: str
):
    bucket, key = path.split('/')[0], '/'.join(path.split('/')[1:])

    S3_CLIENT.delete_object(
        Bucket=bucket, 
        Key=key
    )