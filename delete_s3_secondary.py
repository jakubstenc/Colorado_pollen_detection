import boto3

S3_ENDPOINT = "https://s3.cl4.du.cesnet.cz"
S3_BUCKET = "bucket"
PREFIX_SOURCE = "Ostatni/Pollen_latitude/Source/260709_Pollen-Production/"
PREFIX_DETECTED = "Ostatni/Pollen_latitude/Detected/260709_Pollen-Production/"
PREFIX_EVALUATED = "Ostatni/Pollen_latitude/Evaluated/260709_Pollen-Production/"

s3 = boto3.client(
    's3', 
    endpoint_url=S3_ENDPOINT, 
    aws_access_key_id='1Y920BKC0SAWPNDE8RD6',
    aws_secret_access_key='SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD',
    verify=False
)

def delete_batch(keys):
    if not keys: return
    for i in range(0, len(keys), 1000):
        batch = keys[i:i+1000]
        s3.delete_objects(
            Bucket=S3_BUCKET,
            Delete={
                'Objects': [{'Key': k} for k in batch],
                'Quiet': True
            }
        )

def process_deletions():
    paginator = s3.get_paginator('list_objects_v2')
    
    # Analyze Source
    print("Gathering Source files to delete...")
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=PREFIX_SOURCE)
    source_to_delete = []
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Size'] / (1024*1024) < 20 and 'unknown' not in obj['Key'].lower():
                    source_to_delete.append(obj['Key'])
    
    # Analyze Detected
    print("Gathering Detected files to delete...")
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=PREFIX_DETECTED)
    detected_to_delete = []
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if 'unknown' not in key.lower():
                    if '_2-1' in key or '_2-2' in key or '_2-3' in key or '_2-4' in key:
                        detected_to_delete.append(key)

    # Analyze Evaluated
    print("Gathering Evaluated files to delete...")
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=PREFIX_EVALUATED)
    eval_to_delete = []
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if 'unknown' not in key.lower():
                    if '_2-1' in key or '_2-2' in key or '_2-3' in key or '_2-4' in key:
                        eval_to_delete.append(key)

    all_keys = source_to_delete + detected_to_delete + eval_to_delete
    print(f"Total Source files: {len(source_to_delete)}")
    print(f"Total Detected files: {len(detected_to_delete)}")
    print(f"Total Evaluated files: {len(eval_to_delete)}")
    print(f"Total files to delete: {len(all_keys)}")
    
    if all_keys:
        print("Executing batch deletion...")
        delete_batch(all_keys)
        print("Deletion complete!")
    else:
        print("No files to delete.")

process_deletions()
