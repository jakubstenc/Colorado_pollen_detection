import os
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

def count_prefix(prefix):
    paginator = s3.get_paginator('list_objects_v2')
    count = 0
    for page in paginator.paginate(Bucket='bucket', Prefix='PEG/Colorado/'):
        count += sum(1 for obj in page.get("Contents", []) if obj["Key"].endswith("_labeled.jpg"))
    return count

print(f"Total processed overview image files in S3: {count_prefix('')}")
