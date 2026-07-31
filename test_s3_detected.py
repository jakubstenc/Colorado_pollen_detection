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

def analyze():
    paginator = s3.get_paginator('list_objects_v2')
    
    # Analyze Source
    print("--- Source Files ---")
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=PREFIX_SOURCE)
    source_to_delete = 0
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Size'] / (1024*1024) < 20 and 'unknown' not in obj['Key'].lower():
                    source_to_delete += 1
    print(f"Source files to delete: {source_to_delete}")
    
    # Analyze Detected
    print("--- Detected Files (Zips) ---")
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=PREFIX_DETECTED)
    detected_to_delete = 0
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                # The secondary zips end with _2-1.zip, _2-2.zip etc. or contain _2-1_, _2-2_
                # Let's match based on size < 20MB? Or name?
                # A primary image could also produce few tiles if no pollen is found. 
                # Better to match the names. The source files end in _2-1.czi, _2-2.czi.
                # The zip files are named out_<original_name_without_extension>.zip
                # So they would contain _2-1 or _2-2.
                key = obj['Key']
                if 'unknown' not in key.lower():
                    if '_2-1' in key or '_2-2' in key or '_2-3' in key or '_2-4' in key:
                        detected_to_delete += 1
    print(f"Detected files to delete: {detected_to_delete}")

    # Analyze Evaluated
    print("--- Evaluated Files (.jpg/.csv) ---")
    pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=PREFIX_EVALUATED)
    eval_to_delete = 0
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if 'unknown' not in key.lower():
                    if '_2-1' in key or '_2-2' in key or '_2-3' in key or '_2-4' in key:
                        eval_to_delete += 1
    print(f"Evaluated files to delete: {eval_to_delete}")

analyze()
