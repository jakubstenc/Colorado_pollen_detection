import os
import boto3
from botocore.client import Config

s3_endpoint  = "https://s3.cl4.du.cesnet.cz"
s3_bucket    = "bucket"
access_key   = "1Y920BKC0SAWPNDE8RD6"
secret_key   = "SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD"

resource = boto3.resource(
    "s3",
    endpoint_url=s3_endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False}),
)

bucket = resource.Bucket(s3_bucket)

print("--- CZI Files ---")
czi_count = 0
for obj in bucket.objects.filter(Prefix="PEG/Colorado/Source/"):
    if obj.key.endswith('.czi'):
        print(obj.key)
        czi_count += 1
        if czi_count >= 10:
            break

print("\n--- Model Files ---")
model_count = 0
for obj in bucket.objects.filter(Prefix="PEG/Colorado/trained_models/"):
    if obj.key.endswith('.pt'):
        print(obj.key)
        model_count += 1
        if model_count >= 10:
            break
