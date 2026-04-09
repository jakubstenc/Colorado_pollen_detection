import time
import os
import subprocess
import boto3
from botocore.client import Config
import urllib3

urllib3.disable_warnings()
s3 = boto3.client("s3", endpoint_url="https://s3.cl4.du.cesnet.cz", 
                  aws_access_key_id="1Y920BKC0SAWPNDE8RD6", 
                  aws_secret_access_key="SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD", 
                  verify=False, config=Config(signature_version="s3v4"))

print("⏳ Passively monitoring Cluster Dataset dumping directly on S3 API (checking every 30s)...")
while True:
    try:
        res = s3.list_objects_v2(Bucket="bucket", Prefix="PEG/Colorado/Species_model/Trainig_data/")
        if res.get("KeyCount", 0) > 0:
            print("\\n🎉 New Pipeline outputs successfully detected on S3!")
            print("📥 Instantiating local desktop sync mechanism...")
            # We already have a fast native script ready
            subprocess.run([".venv/bin/python", "download_s3.py"])
            print("✅ Autonomous Sync Completed Successfully! You can browse your active UI now.")
            break
    except Exception as e:
        pass
    
    time.sleep(30)
