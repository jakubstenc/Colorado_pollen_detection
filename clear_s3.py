import boto3
from botocore.client import Config
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

s3 = boto3.client("s3", endpoint_url="https://s3.cl4.du.cesnet.cz", aws_access_key_id="1Y920BKC0SAWPNDE8RD6", aws_secret_access_key="SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD", verify=False, config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False}))

bucket = "bucket"
prefix = "PEG/Colorado/Species_model/Trainig_data/"

paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

delete_us = []
for page in pages:
    for obj in page.get('Contents', []):
        delete_us.append({'Key': obj['Key']})
        if len(delete_us) >= 1000:
            s3.delete_objects(Bucket=bucket, Delete={'Objects': delete_us})
            delete_us = []
            print("Deleted 1000 objects from S3...")

if delete_us:
    s3.delete_objects(Bucket=bucket, Delete={'Objects': delete_us})
    print(f"Deleted remaining {len(delete_us)} objects.")
print("✅ Fully wiped old Trainig_data from S3!")
