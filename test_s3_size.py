import boto3
import os

S3_ENDPOINT = "https://s3.cl4.du.cesnet.cz"
S3_BUCKET = "bucket"
PREFIX = "Ostatni/Pollen_latitude/Source/260709_Pollen-Production/"

s3 = boto3.client(
    's3', 
    endpoint_url=S3_ENDPOINT, 
    aws_access_key_id='1Y920BKC0SAWPNDE8RD6',
    aws_secret_access_key='SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD',
    verify=False
)

def analyze():
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=PREFIX)
    
    small_files = 0
    small_in_unknown = 0
    large_files = 0
    
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                size_mb = obj['Size'] / (1024 * 1024)
                key = obj['Key']
                
                is_unknown = 'unknown' in key.lower()
                
                if size_mb < 20: # arbitrary threshold for secondary files (1-10MB)
                    small_files += 1
                    if is_unknown:
                        small_in_unknown += 1
                else:
                    large_files += 1
                    
    print(f"Total large files to keep: {large_files}")
    print(f"Total small files to delete (excluding Unknown): {small_files - small_in_unknown}")
    print(f"Total small files in Unknown (to KEEP): {small_in_unknown}")

analyze()
