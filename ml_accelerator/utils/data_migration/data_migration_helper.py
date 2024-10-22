from ml_accelerator.utils.logging.logger_helper import get_logger
from ml_accelerator.utils.aws import s3_helper
from ml_accelerator.utils.filesystem import filesystem_helper
from tqdm import tqdm
import os
from pprint import pformat
from typing import Set


# Get logger
LOGGER = get_logger(name=__name__)


def filesystem_to_s3(
    source_bucket: str,
    destination_bucket: str,
    subdir: str = ''
) -> None:
    # Find source paths
    source_paths: Set[str] = filesystem_helper.find_paths(
        bucket=source_bucket,
        directory=subdir
    )

    LOGGER.info('Copying: objects from %s (filesystem) to %s (S3).', source_bucket, destination_bucket)

    # Copy local files into S3 destination bucket
    for source_path in tqdm(source_paths):
        # Define destination object
        destination_object = source_path.replace(f'{source_bucket}/', '')

        # Upload files into S3 bucket
        s3_helper.S3_CLIENT.upload_file(source_path, destination_bucket, destination_object)

    LOGGER.info('Bucket %s (filesystem) was successfully copied into %s (S3).', source_bucket, destination_bucket)


def s3_to_filesystem(
    source_bucket: str,
    destination_bucket: str,
    subdir: str = ''
) -> None:
    # Find source objects
    source_objects: Set[str] = s3_helper.find_keys(
        bucket=source_bucket,
        subdir=subdir,
        include_additional_info=False
    )

    LOGGER.info('Copying: objects from %s (S3) to %s (filesystem).', source_bucket, destination_bucket)

    # Copy S3 objects into filesystem destination bucket
    for source_object in tqdm(source_objects):
        # Define destination object
        destination_path = os.path.join(destination_bucket, source_object)
        
        # Create the destination directory if it doesn't exist
        destination_dir = os.path.dirname(destination_path)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        # Download files into filesystem bucket
        s3_helper.S3_CLIENT.download_file(destination_bucket, source_object, destination_path)

    LOGGER.info('Bucket %s (S3) was successfully copied into %s (filesystem).', source_bucket, destination_bucket)
    

def migrate_data(
    source_bucket: str, 
    destination_bucket: str,
    source_type: str = 'filesystem',
    destination_type: str = 'S3',
    subdir: str = '',
    delete_destination: bool = True,
    debug: bool = False
) -> None:
    # Delete destination bucket
    if delete_destination:
        if destination_type == 'filesystem':
            filesystem_helper.delete_bucket(bucket=destination_bucket)
        elif destination_type == 'S3':
            s3_helper.delete_bucket(bucket=destination_bucket)
        else:
            raise ValueError(f'destination_type: {destination_type} is not a valid destination.')
        
    # Copy source objects to destination bucket
    if source_type == 'filesystem':
        if destination_type == 'filesystem':
            # Copy local files into local destination bucket
            filesystem_helper.copy_bucket(
                source_bucket=source_bucket,
                destination_bucket=destination_bucket,
                subdir=subdir,
                delete_destination=False
            )
        elif destination_type == 'S3':
            filesystem_to_s3(
                source_bucket=source_bucket,
                destination_bucket=destination_bucket,
                subdir=subdir
            )
        else:
            raise ValueError(f'destination_type: {destination_type} is not a valid destination.')

    elif source_type == 'S3':
        if destination_type == 'filesystem':
            s3_to_filesystem(
                source_bucket=source_bucket,
                destination_bucket=destination_bucket,
                subdir=subdir
            )
        elif destination_type == 'S3':
            # Copy S3 files into S3 destination bucket
            s3_helper.copy_bucket(
                source_bucket=source_bucket,
                destination_bucket=destination_bucket,
                subdir=subdir,
                delete_destination=False
            )
        else:
            raise ValueError(f'destination_type: {destination_type} is not a valid destination.')
    else:
        raise ValueError(f'source_type: {source_type} is not a valid source.')


# conda deactivate
# source .ml_accel_venv/bin/activate
# .ml_accel_venv/bin/python ml_accelerator/utils/data_migration/data_migration_helper.py
if __name__ == "__main__":
    migrate_data(
        source_bucket='breast-cancer-bucket-dev',
        destination_bucket='breast-cancer-bucket-dev',
        source_type='S3', # filesystem | S3
        destination_type='filesystem', # filesystem | S3
        # subdir='datasets',
        delete_destination=True,
        debug=True
    )
