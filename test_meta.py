import boto3
from botocore.client import Config
import urllib3
urllib3.disable_warnings()
from aicsimageio import AICSImage
s3 = boto3.client("s3", endpoint_url="https://s3.cl4.du.cesnet.cz", verify=False, config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False}))
paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket='bucket', Prefix='PEG/Colorado/Source/')
for page in pages:
    for obj in page.get('Contents', []):
        if obj['Key'].endswith('.czi'):
            print("Found:", obj['Key'])
            s3.download_file('bucket', obj['Key'], '/tmp/test.czi')
            img = AICSImage('/tmp/test.czi')
            print("px_size:", getattr(img.physical_pixel_sizes, 'X', 'Missing'))
            break
    break
