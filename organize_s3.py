import boto3
from botocore.config import Config
import urllib3
from pathlib import Path

urllib3.disable_warnings()

def main():
    s3_endpoint = "https://s3.cl4.du.cesnet.cz"
    s3_bucket = "bucket"
    aws_access_key_id = "1Y920BKC0SAWPNDE8RD6"
    aws_secret_access_key = "SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD"

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False}),
        verify=False
    )

    prefix = "PEG/Colorado/Detected/"
    print(f"Reorganizing S3 bucket {s3_bucket} prefix {prefix}...")
    
    paginator = s3.get_paginator('list_objects_v2')
    objects_to_delete = []
    
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = Path(key).name
            
            if not filename.endswith(".csv") and not filename.endswith(".jpg"):
                continue
                
            # Avoid re-processing if already in correct folder
            if "Pollen_production/" in key or "Pollen_deposition/" in key:
                continue

            is_dep = "Dep_" in filename
            is_pro = "pol_pro" in filename
            
            if not is_dep and not is_pro:
                continue
                
            # Extract species
            species = "Unknown"
            if "Ran_ado" in filename or "Ran_Ado" in filename: species = "Ran_ado"
            elif "Cal_Chi" in filename or "Cal_chi" in filename: species = "Cal_chi"
            elif "Gen_alg" in filename or "Gen_Alg" in filename: species = "Gen_alg"
            elif "Sed_lan" in filename or "Sed_Lan" in filename: species = "Sed_lan"
            elif "Vio_adu" in filename or "Vio_Adu" in filename: species = "Vio_adu"
            
            subfolder = "Pollen_deposition" if is_dep else "Pollen_production"
            new_key = f"PEG/Colorado/Detected/{subfolder}/{species}/{filename}"
            
            print(f"Copying {key} -> {new_key}")
            try:
                s3.copy_object(
                    Bucket=s3_bucket,
                    CopySource={'Bucket': s3_bucket, 'Key': key},
                    Key=new_key
                )
                objects_to_delete.append({'Key': key})
            except Exception as e:
                print(f"Failed to copy {key}: {e}")
                
    if objects_to_delete:
        print(f"Deleting {len(objects_to_delete)} old objects...")
        # Delete in batches of 1000
        for i in range(0, len(objects_to_delete), 1000):
            batch = objects_to_delete[i:i+1000]
            s3.delete_objects(
                Bucket=s3_bucket,
                Delete={'Objects': batch}
            )
    print("S3 reorganization complete.")

if __name__ == "__main__":
    main()
