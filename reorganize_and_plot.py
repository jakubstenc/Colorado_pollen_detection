import os
import shutil
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

def reorganize_mount(detected_path):
    print("Reorganizing mounted S3...")
    prod_dir = detected_path / "Pollen_production"
    dep_dir = detected_path / "Pollen_deposition"
    
    prod_dir.mkdir(exist_ok=True)
    dep_dir.mkdir(exist_ok=True)
    
    # 1. Move species folders (which contain pol_pro files) into Pollen_production
    species = ["Cal_chi", "Gen_alg", "Ran_ado", "Sed_lan", "Vio_adu"]
    for sp in species:
        sp_path = detected_path / sp
        if sp_path.exists() and sp_path.is_dir():
            print(f"Moving {sp} to Pollen_production...")
            try:
                shutil.move(str(sp_path), str(prod_dir / sp))
            except Exception as e:
                print(f"Failed to move {sp}: {e}")

    # 2. Move root files into Pollen_production or Pollen_deposition based on name
    for f in detected_path.iterdir():
        if not f.is_file():
            continue
            
        filename = f.name
        is_dep = "Dep_" in filename
        is_pro = "pol_pro" in filename
        
        if not is_dep and not is_pro:
            continue
            
        sp = "Unknown"
        if "Ran_ado" in filename or "Ran_Ado" in filename: sp = "Ran_ado"
        elif "Cal_Chi" in filename or "Cal_chi" in filename: sp = "Cal_chi"
        elif "Gen_alg" in filename or "Gen_Alg" in filename: sp = "Gen_alg"
        elif "Sed_lan" in filename or "Sed_Lan" in filename: sp = "Sed_lan"
        elif "Vio_adu" in filename or "Vio_Adu" in filename: sp = "Vio_adu"
        
        target_dir = dep_dir / sp if is_dep else prod_dir / sp
        target_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.move(str(f), str(target_dir / filename))
        except Exception as e:
            print(f"Failed to move {filename}: {e}")
        
    print("Reorganization complete.")
    return prod_dir, dep_dir


def extract_dep_date(filename):
    # e.g. measurements_20260701_001_Dep_Ran_Ado_24_7_161b_Colorado2025.czi.csv
    # Looks for Day_Month pattern after Species
    match = re.search(r'_([0-9]{1,2})_([0-9]{1,2})_', filename)
    if match:
        day = int(match.group(1))
        month = int(match.group(2))
        return pd.to_datetime(f"2025-{month:02d}-{day:02d}")
    return pd.NaT

def process_deposition(dep_dir):
    print("Processing Deposition...")
    records = []
    
    # Read summary files
    for sp_dir in dep_dir.iterdir():
        if not sp_dir.is_dir(): continue
        species = sp_dir.name
        
        for f in sp_dir.iterdir():
            if f.name.startswith("summary_") and f.name.endswith(".csv"):
                try:
                    df = pd.read_csv(f)
                    if not df.empty:
                        row = df.iloc[0]
                        # Try to get date from filename
                        date_obj = extract_dep_date(f.name)
                        
                        records.append({
                            "Species": species,
                            "Collection_Date": date_obj,
                            "File": f.name,
                            "Total_Grains": row.get("Total_Grains", 0),
                            "Conspecific": row.get("Conspecific", 0),
                            "Heterospecific": row.get("Heterospecific", 0),
                        })
                except Exception as e:
                    print(f"Error reading {f.name}: {e}")
                    
    if not records:
        print("No deposition records found.")
        return
        
    df = pd.DataFrame(records)
    df = df.dropna(subset=["Collection_Date"])
    
    if df.empty:
        print("No valid dates found for deposition.")
        return
        
    agg = df.groupby(["Collection_Date", "Species"])["Total_Grains"].agg(["mean", "sem", "count"]).reset_index()
    agg.rename(columns={"mean": "Average_Grains", "sem": "Standard_Error", "count": "N_Samples"}, inplace=True)
    agg["Standard_Error"] = agg["Standard_Error"].fillna(0)
    
    agg.to_csv("results/pollen_deposition_aggregated.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.get_cmap("tab10").colors
    for i, sp in enumerate(agg["Species"].unique()):
        sub = agg[agg["Species"] == sp].sort_values("Collection_Date")
        plt.errorbar(sub["Collection_Date"], sub["Average_Grains"], yerr=sub["Standard_Error"], 
                     fmt="-o", capsize=5, label=sp, color=colors[i], markersize=6)
                     
    plt.title("Pollen Deposition Over Time", fontsize=16)
    plt.xlabel("Collection Date", fontsize=12)
    plt.ylabel("Average Grains per Image", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(title="Species")
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

def process_production(prod_dir):
    print("Processing Production...")
    records = []
    
    for sp_dir in prod_dir.iterdir():
        if not sp_dir.is_dir(): continue
        species = sp_dir.name
        
        for f in sp_dir.iterdir():
            if f.name.endswith("_details.csv"):
                try:
                    df = pd.read_csv(f)
                    detections = len(df) if not df.empty else 0
                    date_obj = extract_prod_date(f.name)
                    if pd.notna(date_obj):
                        records.append({
                            "Species": species,
                            "Collection_Date": date_obj,
                            "File": f.name,
                            "Detections": detections
                        })
                except Exception as e:
                    pass
                    
    if not records:
        print("No valid production records found.")
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
    colors = plt.cm.get_cmap("Set2").colors
    for i, sp in enumerate(agg["Species"].unique()):
        sub = agg[agg["Species"] == sp].sort_values("Collection_Date")
        plt.errorbar(sub["Collection_Date"], sub["Average_Detections"], yerr=sub["Standard_Error"], 
                     fmt="-o", capsize=5, label=sp, color=colors[i % len(colors)], markersize=6)
                     
    plt.title("Pollen Production Over Time", fontsize=16)
    plt.xlabel("Collection Date", fontsize=12)
    plt.ylabel("Average Grains per Anther", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(title="Species")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/pollen_production_timeline.png", dpi=300)
    print("Production plot saved.")



def main():
    os.makedirs("results", exist_ok=True)
    detected_path = Path("/home/meow/cesnet_data/PEG/Colorado/Detected")
    if not detected_path.exists():
        print("Mount point not found.")
        return
        
    prod_dir, dep_dir = reorganize_mount(detected_path)
    
    process_deposition(dep_dir)
    process_production(prod_dir)

if __name__ == "__main__":
    main()
