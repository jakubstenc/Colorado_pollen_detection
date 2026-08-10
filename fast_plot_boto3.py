import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import boto3
from botocore.config import Config
import urllib3
import io
urllib3.disable_warnings()

def extract_dep_date(filename):
    match = re.search(r'_([0-9]{1,2})_([0-9]{1,2})_', filename)
    if match:
        day = int(match.group(1))
        month = int(match.group(2))
        return pd.to_datetime(f"2025-{month:02d}-{day:02d}")
    return pd.NaT

def process_deposition_boto3():
    print("Fetching Deposition via boto3...")
    
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
    
    records = []
    paginator = s3.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=s3_bucket, Prefix="PEG/Colorado/Detected/Pollen_deposition/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = os.path.basename(key)
            
            if not filename.startswith("summary_") or "Dep_" not in filename or not filename.endswith(".csv"):
                continue
                
            sp = "Unknown"
            if "Ran_ado" in filename or "Ran_Ado" in filename: sp = "Ran_ado"
            elif "Cal_Chi" in filename or "Cal_chi" in filename: sp = "Cal_chi"
            elif "Gen_alg" in filename or "Gen_Alg" in filename: sp = "Gen_alg"
            elif "Sed_lan" in filename or "Sed_Lan" in filename: sp = "Sed_lan"
            elif "Vio_adu" in filename or "Vio_Adu" in filename: sp = "Vio_adu"
            
            try:
                response = s3.get_object(Bucket=s3_bucket, Key=key)
                df = pd.read_csv(io.BytesIO(response['Body'].read()))
                if not df.empty:
                    row = df.iloc[0]
                    date_obj = extract_dep_date(filename)
                    records.append({
                        "Species": sp,
                        "Collection_Date": date_obj,
                        "File": filename,
                        "Total_Grains": row.get("Total_Grains", 0),
                        "Conspecific": row.get("Conspecific", 0),
                        "Heterospecific": row.get("Heterospecific", 0),
                    })
            except Exception as e:
                print(f"Error reading {key}: {e}")
                
    if not records:
        print("No deposition records found.")
        return
        
    df = pd.DataFrame(records)
    df = df.dropna(subset=["Collection_Date"])
    
    if df.empty:
        print("No valid dates found for deposition.")
        return
        
    agg = df.groupby(["Collection_Date", "Species"])[["Total_Grains", "Conspecific", "Heterospecific"]].agg(["mean", "sem", "count"]).reset_index()
    
    # Flatten multi-level columns
    agg.columns = ["_".join(a).strip("_") for a in agg.columns.to_flat_index()]
    
    # Rename for clarity
    agg.rename(columns={
        "Total_Grains_count": "N_Samples",
        "Total_Grains_mean": "Total_Grains_Avg",
        "Total_Grains_sem": "Total_Grains_SE",
        "Conspecific_mean": "Conspecific_Avg",
        "Conspecific_sem": "Conspecific_SE",
        "Heterospecific_mean": "Heterospecific_Avg",
        "Heterospecific_sem": "Heterospecific_SE"
    }, inplace=True)
    
    agg.fillna(0, inplace=True)
    
    os.makedirs("results", exist_ok=True)
    agg.to_csv("results/pollen_deposition_aggregated.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.get_cmap("tab10").colors
    for i, sp in enumerate(agg["Species"].unique()):
        sub = agg[agg["Species"] == sp].sort_values("Collection_Date")
        
        # Plot Conspecific
        plt.errorbar(sub["Collection_Date"], sub["Conspecific_Avg"], yerr=sub["Conspecific_SE"], 
                     fmt="-o", capsize=5, label=f"{sp} (Conspecific)", color=colors[i], markersize=6)
                     
        # Plot Heterospecific
        plt.errorbar(sub["Collection_Date"], sub["Heterospecific_Avg"], yerr=sub["Heterospecific_SE"], 
                     fmt="--s", capsize=5, label=f"{sp} (Heterospecific)", color=colors[(i + 1) % len(colors)], markersize=6)
                     
    plt.title("Pollen Deposition Over Time (Conspecific vs Heterospecific)", fontsize=16)
    plt.xlabel("Collection Date", fontsize=12)
    plt.ylabel("Average Grains per Image", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(title="Species & Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/pollen_deposition_timeline.png", dpi=300)
    print("Deposition plot saved.")

def extract_prod_date(filename):
    match = re.search(r'_([0-9]{1,2})_([0-9]{1,2})_', filename)
    if match:
        day = int(match.group(1))
        month = int(match.group(2))
        return pd.to_datetime(f"2025-{month:02d}-{day:02d}")
    return pd.NaT

def process_production_boto3():
    print("Fetching Production via boto3...")
    
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
    
    paginator = s3.get_paginator('list_objects_v2')
    species_list = ["Ran_ado", "Cal_chi", "Gen_alg", "Sed_lan", "Vio_adu"]
    
    records = []
    
    for sp in species_list:
        keys_to_try = [
            f"PEG/Colorado/Detected/Pollen_production/{sp}/",
            f"PEG/Colorado/Detected/{sp}/"
        ]
        
        for prefix in keys_to_try:
            for page in paginator.paginate(Bucket=s3_bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    filename = os.path.basename(key)
                    
                    if key.endswith("_details.csv"):
                        try:
                            response = s3.get_object(Bucket=s3_bucket, Key=key)
                            df = pd.read_csv(io.BytesIO(response['Body'].read()))
                            detections = len(df) if not df.empty else 0
                            
                            date_obj = extract_prod_date(filename)
                            if pd.notna(date_obj):
                                records.append({
                                    "Species": sp,
                                    "Collection_Date": date_obj,
                                    "File": filename,
                                    "Detections": detections
                                })
                        except Exception as e:
                            print(f"Failed to process {key}: {e}")

    if not records:
        print("No production records found on S3.")
        return
        
    df = pd.DataFrame(records)
    df = df.dropna(subset=["Collection_Date"])
    
    if df.empty:
        print("No valid dates found for production.")
        return
        
    agg = df.groupby(["Collection_Date", "Species"])["Detections"].agg(["mean", "sem", "count"]).reset_index()
    agg.rename(columns={"mean": "Average_Detections", "sem": "Standard_Error", "count": "N_Samples"}, inplace=True)
    agg["Standard_Error"] = agg["Standard_Error"].fillna(0)
    
    os.makedirs("results", exist_ok=True)
    agg.to_csv("results/pollen_production_aggregated.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    # 'Set2' colormap has 8 colors, which is perfect for up to 8 species
    # For matplotlib > 3.7, we can use plt.colormaps['Set2'] instead of get_cmap
    colors = plt.cm.get_cmap("Set2").colors
    for i, sp in enumerate(agg["Species"].unique()):
        sub = agg[agg["Species"] == sp].sort_values("Collection_Date")
        plt.errorbar(sub["Collection_Date"], sub["Average_Detections"], yerr=sub["Standard_Error"], 
                     fmt="-o", capsize=5, label=sp, color=colors[i % len(colors)], markersize=6)
                     
    plt.title("Pollen Production Over Time", fontsize=16)
    plt.xlabel("Collection Date", fontsize=12)
    plt.ylabel("Average Grains per Anther", fontsize=12)
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(title="Species")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/pollen_production_timeline.png", dpi=300)
    print("Production plot saved.")

def main():
    process_deposition_boto3()
    process_production_boto3()

if __name__ == "__main__":
    main()
