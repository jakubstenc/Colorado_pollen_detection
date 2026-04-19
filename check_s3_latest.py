import boto3
from botocore.client import Config
import urllib3
urllib3.disable_warnings()

s3 = boto3.client(
    's3',
    endpoint_url='https://s3.cl4.du.cesnet.cz',
    aws_access_key_id='1Y920BKC0SAWPNDE8RD6',
    aws_secret_access_key='SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD',
    verify=False,
    config=Config(signature_version='s3v4')
)

paginator = s3.get_paginator('list_objects_v2')
latest_time = None
latest_key = None

for page in paginator.paginate(Bucket='bucket', Prefix='PEG/Colorado/'):
    for obj in page.get("Contents", []):
        if obj["Key"].endswith("_labeled.jpg") or "_viz.jpg" in obj["Key"]:
            if latest_time is None or obj["LastModified"] > latest_time:
                latest_time = obj["LastModified"]
                latest_key = obj["Key"]

print(f"Latest S3 output: {latest_key} at {latest_time}")
