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
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Set
from pprint import pprint


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

# Create an s3fs.S3FileSystem instance
FS = s3fs.S3FileSystem(
    key=ACCESS_KEYS["AWS_ACCESS_KEY_ID"], # os.environ.get("AWS_ACCESS_KEY_ID"),
    secret=ACCESS_KEYS["AWS_SECRET_ACCESS_KEY"], # os.environ.get("AWS_SECRET_ACCESS_KEY"),
    anon=False  # Set to True if your bucket is public
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


def find_keys(
    bucket: str,
    subdir: str = None,
    include_additional_info: bool = False
) -> Set[str]:
    # Validate subdir
    if subdir is None:
        subdir = ''

    # Define keys to populate
    s3_keys = set()

    # Find dirs
    prefixes = S3_CLIENT.list_objects_v2(
        Bucket=bucket,
        Prefix=subdir, 
        Delimiter='/'
    ).get('CommonPrefixes')

    if prefixes is not None:
        prefixes = [p['Prefix'] for p in prefixes]

        for prefix in prefixes:
            # Find prefix contents
            contents = S3_CLIENT.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix
            ).get('Contents', [])
        
            if len(contents) > 0:
                if not include_additional_info:
                    s3_keys = s3_keys | {
                        content['Key'] for content in contents
                        if not(content['Key'].endswith('/'))
                    }
                else:
                    s3_keys = s3_keys | {
                        (content['Key'], content['Size'], content['LastModified']) for content in contents
                        if not(content['Key'].endswith('/'))
                    }

        # print('s3_keys:')
        # pprint(s3_keys)
        # print('\n\n')
        
        return s3_keys
    print(f'[WARNING] No keys were found for bucket: {bucket}, subdir: {subdir}.\n')
    return {}


def find_prefixes(
    bucket: str, 
    prefix: str = None, 
    results: set = set(),
    debug: bool = False
):
    if prefix is None:
        prefix = ''

    if debug:
        print(f'bucket: {bucket}\n'
              f'prefix: {prefix}\n\n')

    result: dict = S3_CLIENT.list_objects_v2(
        Bucket=bucket, 
        Prefix=prefix, 
        Delimiter='/'
    )
    if debug:
        print(f'result')
        pprint(result)
        print('\n\n')
    
    for common_prefix in result.get('CommonPrefixes', []):
        subdir = common_prefix.get('Prefix')
        results.add(subdir)
        
        # Recursively list subdirectories
        find_prefixes(bucket, subdir, results)

    return results


def delete_s3_directory(
    bucket, 
    directory
):
    # List objects with the common prefix
    objects = S3_CLIENT.list_objects_v2(Bucket=bucket, Prefix=directory)
    
    # Check if there are objects to delete
    if 'Contents' in objects:
        for obj in objects['Contents']:
            # print(f'deleating: {obj["Key"]}')
            S3_CLIENT.delete_object(Bucket=bucket, Key=obj['Key'])

    # Check if there are subdirectories (common prefixes) to delete
    if 'CommonPrefixes' in objects:
        for subdir in objects['CommonPrefixes']:
            delete_s3_directory(bucket, subdir['Prefix'])

    # Finally, delete the common prefix (the "directory" itself)
    S3_CLIENT.delete_object(Bucket=bucket, Key=directory)

    # print('\n')


def sincronize_buckets(
    source_bucket: str, 
    destination_bucket: str, 
    sub_dir: str = None,
    debug: bool = False
):
    """
    Objects
    """
    # Find destination objects
    dest_objects = find_keys(
        bucket=destination_bucket,
        include_additional_info=False
    )
    while len(dest_objects) > 0:
        if debug:
            print(f'dest_objects:')
            pprint(dest_objects)
            print('\n\n')

        # Remove destination objects
        print(f'Removing objects from {destination_bucket}:')
        for obj in tqdm(dest_objects):
            # print(f"Removing: {destination_bucket}/{obj}")
            delete_from_s3(path=f"{destination_bucket}/{obj}")

        # Re-setting dest_objects
        dest_objects = find_keys(
            bucket=destination_bucket,
            include_additional_info=False
        )
    
    print(f'\nAll objects in {destination_bucket} have been removed.\n\n')
    
    # Find source objects
    source_objects = find_keys(
        bucket=source_bucket,
        include_additional_info=False
    )
    
    while len(source_objects - dest_objects) > 0:
        # Copy source objects to destination bucket
        print(f"Copying: objects from {source_bucket} to {destination_bucket}")
        for obj in tqdm(source_objects - dest_objects):
            # print(f"Copying: {obj} from {source_bucket} to {destination_bucket}")
            S3_CLIENT.copy_object(
                Bucket=destination_bucket, 
                CopySource={
                    'Bucket': source_bucket, 
                    'Key': obj
                },
                Key=obj
            )
        
        # Re-setting dest_objects
        dest_objects = find_keys(
            bucket=destination_bucket,
            include_additional_info=False
        )
    
    print(f'\n{destination_bucket} has been filled.\n\n')
