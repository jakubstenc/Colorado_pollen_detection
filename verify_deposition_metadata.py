import boto3
from botocore.config import Config
import urllib3
from pathlib import Path
from aicsimageio import AICSImage
import os

urllib3.disable_warnings()

def get_s3_client():
    endpoint = os.environ.get("S3_ENDPOINT", "https://s3.cl4.du.cesnet.cz")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "1Y920BKC0SAWPNDE8RD6")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD")
    
    config = Config(connect_timeout=60, retries={'max_attempts': 5})
    return boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        verify=False,
        config=config
    )

def inspect_czi(s3_client, bucket, key, label):
    filename = key.split('/')[-1]
    tmp_path = Path(f"/tmp/{filename}")
    print(f"\n--- Inspecting [{label}]: {filename} ---")
    try:
        s3_client.download_file(bucket, key, str(tmp_path))
        img = AICSImage(str(tmp_path))
        print(f"  Dims: {img.dims}")
        print(f"  Shape: {img.shape}")
        print(f"  Physical Pixel Sizes: {img.physical_pixel_sizes}")
        print(f"  Channel Names: {img.channel_names}")
    except Exception as e:
        print(f"  Error: {e}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

def main():
    s3 = get_s3_client()
    bucket = "bucket"
    
    # 1. Get sample standard source CZI
    standard_key = None
    res = s3.list_objects_v2(Bucket=bucket, Prefix="PEG/Colorado/Source/Ran_ado/", MaxKeys=5)
    for obj in res.get('Contents', []):
        if obj['Key'].endswith('.czi'):
            standard_key = obj['Key']
            break
            
    # 2. Get sample Pollen_deposition CZI
    deposition_key = None
    res_dep = s3.list_objects_v2(Bucket=bucket, Prefix="PEG/Colorado/Source/Pollen_deposition/Ran_ado/", MaxKeys=5)
    for obj in res_dep.get('Contents', []):
        if obj['Key'].endswith('.czi'):
            deposition_key = obj['Key']
            break

    if standard_key:
        inspect_czi(s3, bucket, standard_key, "Standard Source Scan")
    if deposition_key:
        inspect_czi(s3, bucket, deposition_key, "Pollen Deposition Scan")

if __name__ == '__main__':
    main()
