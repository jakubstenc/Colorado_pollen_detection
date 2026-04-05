import boto3
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from botocore.client import Config

s3 = boto3.client("s3", endpoint_url="https://s3.cl4.du.cesnet.cz", aws_access_key_id="1Y920BKC0SAWPNDE8RD6", aws_secret_access_key="SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD", verify=False, config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False}))
paginator = s3.get_paginator('list_objects_v2')

total_czi = 0
for page in paginator.paginate(Bucket='bucket', Prefix='PEG/Colorado/Source/'):
    total_czi += sum(1 for obj in page.get('Contents', []) if obj['Key'].endswith('.czi'))

print(f'Total CZI files in S3: {total_czi}')

processed = 0
try:
    with open('dataset_logs.txt', 'r') as f:
        for line in f:
            if 'Extracting' in line and '.czi' in line:
                processed += 1
    print(f'Processed files so far: {processed}')
except Exception as e:
    print('Could not read dataset_logs.txt', e)
