import boto3
from botocore.client import Config
import urllib3
urllib3.disable_warnings()

s3 = boto3.client('s3', endpoint_url='https://s3.cl4.du.cesnet.cz', aws_access_key_id='1Y920BKC0SAWPNDE8RD6', aws_secret_access_key='SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD', verify=False, config=Config(signature_version='s3v4'))

prefix = "PEG/Colorado/Curated_Retrain_Data/"
print(f"🔍 Scanning S3: {prefix}")
res = s3.list_objects_v2(Bucket='bucket', Prefix=prefix)
contents = res.get('Contents', [])

images = [c['Key'] for c in contents if '/images/' in c['Key'] and c['Key'].endswith('.jpg')]
labels = [c['Key'] for c in contents if '/labels/' in c['Key'] and c['Key'].endswith('.txt')]

img_stems = set([i.split('/')[-1].replace('.jpg','') for i in images])
lbl_stems = set([l.split('/')[-1].replace('.txt','') for l in labels])

print(f"Total curated images: {len(images)}")
print(f"Total curated labels: {len(labels)}")

missing_labels = img_stems - lbl_stems
missing_images = lbl_stems - img_stems

if missing_labels: 
    print(f"❌ Images missing labels: {len(missing_labels)}")
    # We can delete them from S3 or just prune them during training...
if missing_images: 
    print(f"❌ Labels missing images: {len(missing_images)}")

if not missing_labels and not missing_images: 
    print("✅ PERFECT 1:1 MATCH! The curated dataset is structurally pristine.")
    
# Check what kind of files are hard negatives vs positives
# If a label file is 0 bytes, it's a hard negative background
empty_labels = 0
for l in contents:
    if '/labels/' in l['Key'] and l['Key'].endswith('.txt'):
        if l['Size'] == 0:
            empty_labels += 1

print(f"Background (Hard Negatives): {empty_labels}")
print(f"Positive Pollen Labels: {len(labels) - empty_labels}")
