import boto3
from botocore.config import Config
import urllib3
urllib3.disable_warnings()

s3 = boto3.client(
    "s3",
    endpoint_url="https://s3.cl4.du.cesnet.cz",
    aws_access_key_id="1Y920BKC0SAWPNDE8RD6",
    aws_secret_access_key="SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD",
    config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False}),
    verify=False
)

prefix = "PEG/Colorado/Species_model/Trainig_data/Deposition_Stigmas/"
paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket="bucket", Prefix=prefix)

delete_us = dict(Objects=[])
count = 0
for page in pages:
    for obj in page.get("Contents", []):
        delete_us['Objects'].append(dict(Key=obj['Key']))
        count += 1
        
        if len(delete_us['Objects']) >= 1000:
            s3.delete_objects(Bucket="bucket", Delete=delete_us)
            delete_us = dict(Objects=[])

if len(delete_us['Objects']) > 0:
    s3.delete_objects(Bucket="bucket", Delete=delete_us)

print(f"Deleted {count} objects from {prefix}")
