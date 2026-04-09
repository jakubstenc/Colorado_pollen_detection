import zipfile
import urllib3
import boto3
import os
import random
from botocore.client import Config
urllib3.disable_warnings()

s3 = boto3.client("s3", endpoint_url="https://s3.cl4.du.cesnet.cz", 
                  aws_access_key_id="1Y920BKC0SAWPNDE8RD6", 
                  aws_secret_access_key="SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD", 
                  verify=False, config=Config(signature_version="s3v4"))

zip_path = "Pollen_lazy_labeling.zip"
print("Downloading staging zip from S3...")
s3.download_file("bucket", "PEG/Colorado/Staged_area/Pollen_lazy_labeling.v6i.yolov8.zip", zip_path)

print("Extracting and sampling labels...")
with zipfile.ZipFile(zip_path, 'r') as z:
    txt_files = [f for f in z.namelist() if f.endswith('.txt') and 'labels/' in f]
    sample = random.choice(txt_files)
    print(f"\n--- Sample Label: {sample} ---")
    content = z.read(sample).decode('utf-8')
    print(content[:500] + "..." if len(content) > 500 else content)
    
    # Let's count how many points the first annotation has.
    lines = content.strip().split('\n')
    if lines and lines[0]:
        parts = lines[0].split()
        num_points = (len(parts) - 1) // 2
        print(f"\nFirst annotation has {num_points} vertices (polygon).")
        if num_points == 4:
            print("This is a simple bounding box (4 points).")
        elif num_points > 4:
            print("This is a complex polygon! (Likely the cross shape).")
            
os.remove(zip_path)
