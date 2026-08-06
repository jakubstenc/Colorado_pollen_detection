import os
import shutil
from pathlib import Path
import boto3
from botocore.config import Config
import urllib3
urllib3.disable_warnings()
import pandas as pd

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

    base_dir = Path("results")
    prod_dir = base_dir / "Pollen_production"
    dep_dir = base_dir / "Pollen_deposition"
    
    prod_dir.mkdir(parents=True, exist_ok=True)
    dep_dir.mkdir(parents=True, exist_ok=True)

    # 1. Move existing local species folders if they contain pol_pro files
    species_folders = ["Cal_chi", "Gen_alg", "Ran_ado", "Sed_lan", "Vio_adu"]
    for sp in species_folders:
        old_path = base_dir / sp
        if old_path.exists() and old_path.is_dir():
            new_path = prod_dir / sp
            new_path.mkdir(exist_ok=True)
            for f in old_path.iterdir():
                if f.is_file():
                    shutil.move(str(f), str(new_path / f.name))
            try:
                old_path.rmdir()
            except OSError:
                pass # Not empty

    # 2. Download from S3
    print("Fetching from S3...")
    prefix = "PEG/Colorado/Detected/"
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = Path(key).name
            if not filename.endswith(".csv"):
                continue
                
            # Determine destination based on filename
            is_dep = "Dep_" in filename
            is_pro = "pol_pro" in filename
            
            if not is_dep and not is_pro:
                continue
                
            # Extract species name (heuristics)
            species = "Unknown"
            if "Ran_ado" in filename or "Ran_Ado" in filename: species = "Ran_ado"
            elif "Cal_Chi" in filename or "Cal_chi" in filename: species = "Cal_chi"
            elif "Gen_alg" in filename or "Gen_Alg" in filename: species = "Gen_alg"
            elif "Sed_lan" in filename or "Sed_Lan" in filename: species = "Sed_lan"
            elif "Vio_adu" in filename or "Vio_Adu" in filename: species = "Vio_adu"
            
            dest_dir = dep_dir / species if is_dep else prod_dir / species
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            local_path = dest_dir / filename
            if not local_path.exists():
                print(f"Downloading {filename} to {dest_dir}...")
                s3.download_file(s3_bucket, key, str(local_path))

if __name__ == "__main__":
    main()
