import boto3
from botocore.config import Config
import urllib3
urllib3.disable_warnings()

s3_endpoint = "https://s3.cl4.du.cesnet.cz"
s3_bucket = "bucket"
aws_access_key_id = "1Y920BKC0SAWPNDE8RD6"
aws_secret_access_key = "SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD"

s3 = boto3.client(
    "s3",
    endpoint_url=s3_endpoint,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False}),
    verify=False
)

paginator = s3.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket=s3_bucket, Prefix="PEG/Colorado/Detected/"):
    for obj in page.get("Contents", []):
        if "csv" in obj["Key"]:
            print(obj["Key"])
