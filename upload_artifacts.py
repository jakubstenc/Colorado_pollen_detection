import boto3
from botocore.client import Config
import urllib3
import os
urllib3.disable_warnings()

s3 = boto3.client(
    's3',
    endpoint_url='https://s3.cl4.du.cesnet.cz',
    aws_access_key_id='1Y920BKC0SAWPNDE8RD6',
    aws_secret_access_key='SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD',
    verify=False,
    config=Config(signature_version='s3v4')
)

files_to_upload = [
    'BoxF1_curve.png', 'BoxP_curve.png', 'BoxPR_curve.png', 'BoxR_curve.png',
    'confusion_matrix_normalized.png', 'confusion_matrix.png',
    'MaskF1_curve.png', 'MaskP_curve.png', 'MaskPR_curve.png', 'MaskR_curve.png',
    'val_batch0_labels.jpg', 'val_batch0_pred.jpg',
    'val_batch1_labels.jpg', 'val_batch1_pred.jpg',
    'val_batch2_labels.jpg', 'val_batch2_pred.jpg'
]

artifacts_path = "/home/meow/.gemini/antigravity/brain/628c0976-1c89-475b-9df8-070ae27f397f"
prefix = "PEG/Colorado/trained_models/general_pollen/general_pollen_20260407_1430/"
bucket = "bucket"

print("Uploading legacy evaluation curves back to S3...")
for file in files_to_upload:
    local_path = os.path.join(artifacts_path, file)
    s3_key = prefix + file
    if os.path.exists(local_path):
        with open(local_path, 'rb') as f:
            s3.put_object(Bucket=bucket, Key=s3_key, Body=f, ContentLength=os.path.getsize(local_path))
        print(f"Uploaded {file}")
    else:
        print(f"Missing {file}")

print("Upload Complete.")
