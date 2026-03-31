import os
import boto3
from botocore.client import Config

def load_env():
    try:
        with open('/home/meow/Documents/Antigravity/Colorado_pollen_detection/.env', 'r') as f:
            for line in f:
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    os.environ[k] = v
    except:
        pass

load_env()
s3 = boto3.resource(
    "s3",
    endpoint_url=os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz"),
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False}),
)

bucket = s3.Bucket(os.environ.get("S3_BUCKET", "bucket"))
for obj in bucket.objects.filter(Prefix="Ostatni/"):
    if obj.key.endswith('.pt'):
        print(obj.key)
