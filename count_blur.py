import boto3
import pandas as pd
from botocore.config import Config
import urllib3
import io
import os

urllib3.disable_warnings()

endpoint = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
access_key = os.environ.get("AWS_ACCESS_KEY_ID", "1Y920BKC0SAWPNDE8RD6")
secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD")

config = Config(connect_timeout=60, retries={'max_attempts': 5})
s3 = boto3.client('s3', endpoint_url=endpoint, aws_access_key_id=access_key, aws_secret_access_key=secret_key, verify=False, config=config)

bucket = "bucket"
prefix = "PEG/Colorado/Detected/"

paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

blur_count = 0
ok_count = 0
total_count = 0

print("Fetching and reading summary CSVs...")
for page in pages:
    for obj in page.get('Contents', []):
        key = obj['Key']
        if key.endswith('.csv') and 'summary_' in key.split('/')[-1]:
            try:
                response = s3.get_object(Bucket=bucket, Key=key)
                df = pd.read_csv(io.BytesIO(response['Body'].read()))
                status = str(df['Status'].iloc[0]).strip().upper()
                if status == 'BLURRY':
                    blur_count += 1
                elif status == 'OK':
                    ok_count += 1
                total_count += 1
            except Exception as e:
                pass

print(f"Total: {total_count}")
print(f"OK: {ok_count}")
print(f"BLURRY: {blur_count}")
