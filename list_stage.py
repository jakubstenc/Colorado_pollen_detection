import urllib3
import boto3
from botocore.client import Config
urllib3.disable_warnings()

s3 = boto3.client("s3", endpoint_url="https://s3.cl4.du.cesnet.cz", 
                  aws_access_key_id="1Y920BKC0SAWPNDE8RD6", 
                  aws_secret_access_key="SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD", 
                  verify=False, config=Config(signature_version="s3v4"))

res = s3.get_paginator('list_objects_v2').paginate(Bucket="bucket", Prefix="PEG/Colorado/")
zips = []
for page in res:
    for obj in page.get("Contents", []):
        if obj["Key"].endswith(".zip"):
            zips.append(obj["Key"])
print(f"Total zip files found in PEG/Colorado/: {len(zips)}")
for z in zips: print(" -", z)
