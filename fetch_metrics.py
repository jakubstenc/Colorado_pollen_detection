import os
import boto3
from botocore.client import Config
import urllib3
urllib3.disable_warnings()

s3 = boto3.client('s3', endpoint_url='https://s3.cl4.du.cesnet.cz', aws_access_key_id='1Y920BKC0SAWPNDE8RD6', aws_secret_access_key='SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD', verify=False, config=Config(signature_version='s3v4'))

prefix = "PEG/Colorado/trained_models/general_pollen/"
res = s3.list_objects_v2(Bucket='bucket', Prefix=prefix, Delimiter='/')

folders = [p.get('Prefix') for p in res.get('CommonPrefixes', [])]
if not folders:
    print("No model runs found!")
    exit(1)

latest_folder = sorted(folders)[-1]
print(f"Latest metrics directory: {latest_folder}")

os.makedirs('diagnostic_plots', exist_ok=True)

# Fetch all PNGs representing F1 confidence curves, P-R curves, labels overlays, etc.
files_res = s3.list_objects_v2(Bucket='bucket', Prefix=latest_folder)
for obj in files_res.get('Contents', []):
    key = obj['Key']
    if key.endswith('.png') or key.endswith('.jpg'):
        fname = key.split('/')[-1]
        print(f"Downloading {fname}...")
        s3.download_file('bucket', key, f'diagnostic_plots/{fname}')

print("✅ Successfully pulled YOLOv8 visual analytics to diagnostic_plots/")
