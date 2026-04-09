import os
import boto3
from botocore.client import Config
import urllib3
import traceback
from src.build_species_dataset import AICSImage, get_mip_rgb, tile_image
urllib3.disable_warnings()

s3 = boto3.client(
    's3',
    endpoint_url='https://s3.cl4.du.cesnet.cz',
    aws_access_key_id='1Y920BKC0SAWPNDE8RD6',
    aws_secret_access_key='SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD',
    verify=False,
    config=Config(signature_version='s3v4')
)

print("Listing keys...")
paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket='bucket', Prefix="PEG/Colorado/Source/")

target_key = None
for page in pages:
    for obj in page.get("Contents", []):
        if "Ran_ado" in obj["Key"] and obj["Key"].endswith(".czi"):
            target_key = obj["Key"]
            break
    if target_key: break

print(f"Target: {target_key}")
local_path = "test.czi"
s3.download_file('bucket', target_key, local_path)

print("Downloaded! Testing AICSImage...")
try:
    img = AICSImage(local_path)
    print("AICSImage success")
    rgb = get_mip_rgb(img)
    print("MIP RGB success")
    for tile, tx, ty in tile_image(rgb):
        print("Tiled!", tile.shape)
        break
except Exception as e:
    print(f"CRASH: {e}")
    traceback.print_exc()

os.remove(local_path)
