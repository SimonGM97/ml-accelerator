from ml_accelerator.config.params import Params
from ml_accelerator.utils.filesystem.filesystem_helper import save_to_filesystem
from ml_accelerator.utils.aws.s3_helper import save_to_s3
import os

def save_inference(inference: dict) -> None:
    # Extract inference attrs
    pred_id = inference['pred_id']
    year = inference['year']
    month = inference['month']
    day = inference['day']

    date: str = inference['date']
    hour: str = date.split(' ')[1].split(':')[0]
    mins: str = date.split(' ')[1].split(':')[1]
    sec: str = date.split(' ')[1].split(':')[2].split('.')[0]

    # Save inference
    save_name = f'inference_pred_id={pred_id}_hour={hour}_mins={mins}_sec={sec}.json'
    if Params.DATA_STORAGE_ENV == 'filesystem':
        save_to_filesystem(
            asset=inference,
            path=os.path.join(Params.BUCKET, *os.environ.get('INFERENCE_PATH').split('/'), year, month, day, save_name)
        )
    elif Params.DATA_STORAGE_ENV == 'S3':
        save_to_s3(
            asset=inference,
            path=f"{Params.BUCKET}/{os.environ.get('INFERENCE_PATH') + '/'.join([year, month, day])}/{save_name}"
        )