import boto3
from botocore.client import Config
import urllib3
urllib3.disable_warnings()

s3 = boto3.client('s3', endpoint_url='https://s3.cl4.du.cesnet.cz', aws_access_key_id='1Y920BKC0SAWPNDE8RD6', aws_secret_access_key='SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD', verify=False, config=Config(signature_version='s3v4'))

prefix = "PEG/Colorado/Curated_Retrain_Data/"
res = s3.list_objects_v2(Bucket='bucket', Prefix=prefix)
contents = res.get('Contents', [])

images = {c['Key'].split('/')[-1].replace('.jpg',''): c['Key'] for c in contents if '/images/' in c['Key'] and c['Key'].endswith('.jpg')}
labels = set([c['Key'].split('/')[-1].replace('.txt','') for c in contents if '/labels/' in c['Key'] and c['Key'].endswith('.txt')])

missing_labels = set(images.keys()) - labels

if missing_labels:
    print(f"Deleting {len(missing_labels)} orphaned images...")
    # Delete in batches of 1000
    objects_to_delete = [{'Key': images[stem]} for stem in missing_labels]
    s3.delete_objects(Bucket='bucket', Delete={'Objects': objects_to_delete})
    print("✅ S3 Cleanup Complete!")
else:
    print("✅ Already pristine 1:1 match.")
