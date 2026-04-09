import os
import boto3
import urllib3
import glob
from botocore.client import Config

urllib3.disable_warnings()

s3 = boto3.client("s3", endpoint_url="https://s3.cl4.du.cesnet.cz", 
                  aws_access_key_id="1Y920BKC0SAWPNDE8RD6", 
                  aws_secret_access_key="SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD", 
                  verify=False, config=Config(signature_version="s3v4"))

s3_bucket = "bucket"
s3_prefix = "PEG/Colorado/Curated_Retrain_Data/"
local_dir = os.path.expanduser("~/Desktop/Retrain_Dataset")

print(f"Scanning {local_dir} for curated files...")
files = glob.glob(os.path.join(local_dir, "**", "*.*"), recursive=True)
files = [f for f in files if os.path.isfile(f)]

if not files:
    print("❌ Retrain_Dataset is completely empty! Nothing to upload.")
    exit(0)

import requests

print("Found", len(files), "files to sync securely to S3...")

success_count = 0
for file_path in files:
    rel_path = os.path.relpath(file_path, local_dir)
    s3_key = os.path.join(s3_prefix, rel_path).replace("\\\\", "/")
    
    try:
        # Generate presigned URL to bypass boto3 chunking issues
        url = s3.generate_presigned_url('put_object', Params={'Bucket': s3_bucket, 'Key': s3_key}, ExpiresIn=3600)
        
        file_size = os.path.getsize(file_path)
        with open(file_path, "rb") as f:
            resp = requests.put(url, data=f, headers={"Content-Length": str(file_size)}, verify=False)
            if resp.status_code == 200:
                success_count += 1
                if success_count % 50 == 0:
                    print(f"Uploaded {success_count}/{len(files)} files...")
            else:
                print(f"⚠️ Failed to upload {rel_path}: HTTP {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"⚠️ Failed to upload {rel_path}: {e}")

print(f"✅ Upload Complete! Successfully synced {success_count} files to S3 (PEG/Colorado/Curated_Retrain_Data/)!")
print("You can manage them directly inside your Cloud storage before kicking off the next cluster job.")
