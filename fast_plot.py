import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

def extract_dep_date(filename):
    match = re.search(r'_([0-9]{1,2})_([0-9]{1,2})_', filename)
    if match:
        day = int(match.group(1))
        month = int(match.group(2))
        return pd.to_datetime(f"2025-{month:02d}-{day:02d}")
    return pd.NaT

def process_deposition(detected_path):
    print("Processing Deposition...")
    records = []
    
    # ONLY LIST THE ROOT FOLDER! Do not use os.walk!
    filenames = os.listdir(detected_path)
    
    for filename in filenames:
        if not (filename.startswith("summary_") and "Dep_" in filename and filename.endswith(".csv")):
            continue
            
        f_path = os.path.join(detected_path, filename)
        
        sp = "Unknown"
        if "Ran_ado" in filename or "Ran_Ado" in filename: sp = "Ran_ado"
        elif "Cal_Chi" in filename or "Cal_chi" in filename: sp = "Cal_chi"
        elif "Gen_alg" in filename or "Gen_Alg" in filename: sp = "Gen_alg"
        elif "Sed_lan" in filename or "Sed_Lan" in filename: sp = "Sed_lan"
        elif "Vio_adu" in filename or "Vio_Adu" in filename: sp = "Vio_adu"
        
        try:
            df = pd.read_csv(f_path)
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
            print(f"Error reading {filename}: {e}")
            
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
    
    os.makedirs("results", exist_ok=True)
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

def process_production():
    print("Processing Production...")
    
    # Use existing Gen_alg_summary.csv for mapping dates
    summary_file = Path("results/Gen_alg_summary.csv")
    if summary_file.exists():
        print("Using existing Gen_alg_summary.csv for production dates")
        df_summary = pd.read_csv(summary_file)
        if "Collection_Date" in df_summary.columns:
            df_summary["Collection_Date"] = pd.to_datetime(df_summary["Collection_Date"])
            df = df_summary.dropna(subset=["Collection_Date"])
            
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
                             fmt="-o", capsize=5, label=sp, color=colors[i], markersize=6)
                             
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
            return

def main():
    detected_path = "/home/meow/cesnet_data/PEG/Colorado/Detected"
    if not os.path.exists(detected_path):
        print("Mount point not found.")
        return
        
    process_deposition(detected_path)
    process_production()

if __name__ == "__main__":
    main()
