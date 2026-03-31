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
    verify=False
)
bucket = s3.Bucket(os.environ.get("S3_BUCKET", "bucket"))

# Create a small test file
with open("test_dummy.txt", "w") as f:
    f.write("test content")

print("Uploading via put_object...")
try:
    with open("test_dummy.txt", "rb") as data:
        file_bytes = data.read()
        bucket.put_object(Key="Ostatni/Colorado_pollen_detection/test_dummy.txt", Body=file_bytes)
    print("put_object succeeded!")
except Exception as e:
    print("put_object failed:", e)

print("Uploading via upload_file...")
try:
    bucket.upload_file("test_dummy.txt", "Ostatni/Colorado_pollen_detection/test_dummy2.txt")
    print("upload_file succeeded!")
except Exception as e:
    print("upload_file failed:", e)
