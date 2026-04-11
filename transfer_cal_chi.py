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

client = resource.meta.client

source_prefix = "PEG/Colorado/Detected/Cal_chi_Inference_Results/"
target_prefix = "PEG/Colorado/Species_model/Trainig_data/"

print(f"Checking for results in {source_prefix}...")

paginator = client.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=s3_bucket, Prefix=source_prefix)

count = 0
for page in pages:
    if "Contents" not in page:
        continue
    for obj in page['Contents']:
        src_key = obj['Key']
        if src_key.endswith('/'): 
            continue
            
        # Determine the relative path
        rel_path = src_key[len(source_prefix):]
        if not rel_path:
            continue
            
        # Target key
        dst_key = target_prefix + rel_path
        
        copy_source = {
            'Bucket': s3_bucket,
            'Key': src_key
        }
        
        client.copy(copy_source, s3_bucket, dst_key)
        # client.delete_object(Bucket=s3_bucket, Key=src_key) # Optional: clean up old
        print(f"Copied: {rel_path} -> {dst_key}")
        count += 1

print(f"\nDone! Successfully transferred {count} files natively across S3.")
