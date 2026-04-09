import os
import boto3
import urllib3
from botocore.client import Config

urllib3.disable_warnings()

s3_endpoint  = "https://s3.cl4.du.cesnet.cz"
s3_bucket    = "bucket"
access_key   = "1Y920BKC0SAWPNDE8RD6"
secret_key   = "SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD"

s3 = boto3.client("s3", endpoint_url=s3_endpoint, aws_access_key_id=access_key, aws_secret_access_key=secret_key, verify=False, config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False}))

prefix = "PEG/Colorado/Species_model/Trainig_data/"
out_dir = os.path.expanduser("~/Desktop/Species_model")

paginator = s3.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        rel_path = key[len(prefix):]
        if not rel_path or key.endswith('/'): continue
        local_path = os.path.join(out_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            s3.download_file(s3_bucket, key, local_path)
        except Exception as e:
            print(f"⚠️ Skipped {key} due to S3 Exception: {e}")

print("✅ Desktop Sync Complete! The dataset is ready at ~/Desktop/Species_model")
