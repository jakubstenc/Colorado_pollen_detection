import boto3
import urllib3
import os
urllib3.disable_warnings()

s3 = boto3.client('s3', 
    endpoint_url='https://s3.cl4.du.cesnet.cz', 
    aws_access_key_id='1Y920BKC0SAWPNDE8RD6', 
    aws_secret_access_key='SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD', 
    verify=False
)
bucket = 'bucket'

# List existing Ran_ado in Pollen_production
paginator = s3.get_paginator('list_objects_v2')
existing_keys = []
for page in paginator.paginate(Bucket=bucket, Prefix='PEG/Colorado/Source/Pollen_production/Ran_ado/'):
    if 'Contents' in page:
        for obj in page['Contents']:
            if obj['Key'].endswith('.czi'):
                existing_keys.append(os.path.basename(obj['Key']))

print(f"Found {len(existing_keys)} files already in Pollen_production/Ran_ado")
if existing_keys:
    print("Example existing files:")
    for k in existing_keys[:3]:
        print(f"  {k}")

# Find Ran_ado in other Source folders
other_folders = [
    'PEG/Colorado/Source/260224_Colorado2025/',
    'PEG/Colorado/Source/260326_Colorado2025_oprava/',
    'PEG/Colorado/Source/260701_Colorado2025/',
    'PEG/Colorado/Source/260702_Colorado2025/',
    'PEG/Colorado/Source/260703_Colorado2025-bad/'
]

candidates = []
for folder in other_folders:
    for page in paginator.paginate(Bucket=bucket, Prefix=folder):
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].endswith('.czi'):
                    lname = obj['Key'].lower()
                    if 'ran_ado' in lname and 'dep_ran_ado' not in lname:
                        candidates.append(obj['Key'])

print(f"Found {len(candidates)} Ran_ado production candidates in other folders")

missing = []
for c in candidates:
    basename = os.path.basename(c)
    if basename not in existing_keys:
        missing.append(c)

print(f"Missing files to be moved: {len(missing)}")
for m in missing:
    basename = os.path.basename(m)
    dest_key = f"PEG/Colorado/Source/Pollen_production/Ran_ado/{basename}"
    print(f"Copying {basename}...")
    s3.copy_object(
        Bucket=bucket,
        CopySource={'Bucket': bucket, 'Key': m},
        Key=dest_key
    )
print("All missing files copied successfully!")
