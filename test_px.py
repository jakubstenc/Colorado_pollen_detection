import boto3
from botocore.config import Config
import urllib3
urllib3.disable_warnings()
from aicsimageio import AICSImage
import cv2

s3 = boto3.client(
    "s3",
    endpoint_url="https://s3.cl4.du.cesnet.cz",
    aws_access_key_id="1Y920BKC0SAWPNDE8RD6",
    aws_secret_access_key="SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD",
    verify=False,
    config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False})
)

paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket='bucket', Prefix='PEG/Colorado/Source/Pollen_deposition/')
for page in pages:
    for obj in page.get('Contents', []):
        if obj['Key'].endswith('.czi'):
            print("Found:", obj['Key'])
            s3.download_file('bucket', obj['Key'], '/tmp/test2.czi')
            img = AICSImage('/tmp/test2.czi')
            print("Physical pixel sizes:", img.physical_pixel_sizes)
            px = getattr(img.physical_pixel_sizes, 'X', None)
            print("px_size:", px)
            break
    break
