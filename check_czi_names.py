import os
import boto3
import urllib3
urllib3.disable_warnings()
from botocore.client import Config
import csv

s3_endpoint  = "https://s3.cl4.du.cesnet.cz"
s3_bucket    = "bucket"
access_key   = "1Y920BKC0SAWPNDE8RD6"
secret_key   = "SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD"

s3 = boto3.client("s3", endpoint_url=s3_endpoint, aws_access_key_id=access_key, aws_secret_access_key=secret_key, verify=False, config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False}))

paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=s3_bucket, Prefix="PEG/Colorado/Source/")

czi_keys = []
for page in pages:
    for obj in page.get("Contents", []):
        if obj["Key"].endswith(".czi"):
            czi_keys.append(obj["Key"])
            
print(f"Total CZI files found: {len(czi_keys)}")

manifest = "/home/meow/Documents/Antigravity/Colorado_pollen_detection/src/species_manifest.csv"
codes = []
with open(manifest, 'r') as f:
    lines = f.readlines()[1:]
    for line in lines:
        code = line.split(",")[0].strip()
        if code:
            codes.append(code)

matched = set()
for key in czi_keys:
    fname = key.split("/")[-1].lower()
    for code in codes:
        if code.lower() in fname:
            matched.add(code)

print("Matched species count:", len(matched))
print("\nUnmatched species count:", len(set(codes) - matched))
print("Unmatched species:")
unmatched = sorted(list(set(codes) - matched))
print(unmatched)

import re
print("\nCandidate files for unmatched species:")
for un in unmatched:
    prefix = un[:3].lower()
    candidates = [k.split("/")[-1] for k in czi_keys if prefix in k.lower()]
    print(f"{un} (prefix {prefix}): {list(set(candidates))[:5]}")
