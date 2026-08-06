import boto3
from botocore.config import Config
s3_endpoint = "https://s3.cl4.du.cesnet.cz"
s3_bucket = "bucket"
s3 = boto3.client("s3", endpoint_url=s3_endpoint, aws_access_key_id="1Y920BKC0SAWPNDE8RD6", aws_secret_access_key="SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD", config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False}), verify=False)
paginator = s3.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket=s3_bucket, Prefix='PEG/Colorado/Pollen_deposition'):
    for obj in page.get('Contents', []):
        print(obj['Key'])
        break
    break
