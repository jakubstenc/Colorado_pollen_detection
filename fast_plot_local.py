import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re

def extract_dep_date(filename):
    match = re.search(r'_([0-9]{1,2})_([0-9]{1,2})_', filename)
    if match:
        day = int(match.group(1))
        month = int(match.group(2))
        return pd.to_datetime(f"2025-{month:02d}-{day:02d}")
    return pd.NaT

def extract_prod_date(filename):
    match = re.search(r'_([0-9]{1,2})_([0-9]{1,2})_', filename)
    if match:
        day = int(match.group(1))
        month = int(match.group(2))
        return pd.to_datetime(f"2025-{month:02d}-{day:02d}")
    return pd.NaT

def process_deposition():
    print("Fetching Deposition via local mount...")
    base_dir = Path("/home/meow/cesnet_data/PEG/Colorado/Detected")
    records = []
    
    # Search directly in the root of Pollen_deposition and Detected
    csv_files = list(base_dir.glob("Pollen_deposition/*/summary_*.csv")) + list(base_dir.glob("summary_*.csv"))
    
    for f in csv_files:
        if "Dep_" not in f.name: continue
        
        sp = "Unknown"
        if "Ran_ado" in f.name or "Ran_Ado" in f.name: sp = "Ran_ado"
        elif "Cal_Chi" in f.name or "Cal_chi" in f.name: sp = "Cal_chi"
        elif "Gen_alg" in f.name or "Gen_Alg" in f.name: sp = "Gen_alg"
        elif "Sed_lan" in f.name or "Sed_Lan" in f.name: sp = "Sed_lan"
        elif "Vio_adu" in f.name or "Vio_Adu" in f.name: sp = "Vio_adu"
        
        try:
            df = pd.read_csv(f)
            if not df.empty:
                row = df.iloc[0]
                date_obj = extract_dep_date(f.name)
                records.append({
                    "Species": sp,
                    "Collection_Date": date_obj,
                    "File": f.name,
                    "Total_Grains": row.get("Total_Grains", 0),
                    "Conspecific": row.get("Conspecific", 0),
                    "Heterospecific": row.get("Heterospecific", 0),
                })
        except:
            pass

    if records:
        df = pd.DataFrame(records).dropna(subset=["Collection_Date"])
        agg = df.groupby(["Collection_Date", "Species"])[["Total_Grains", "Conspecific", "Heterospecific"]].agg(["mean", "sem", "count"]).reset_index()
        agg.columns = ["_".join(a).strip("_") for a in agg.columns.to_flat_index()]
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
            plt.errorbar(sub["Collection_Date"], sub["Conspecific_Avg"], yerr=sub["Conspecific_SE"], 
                         fmt="-o", capsize=5, label=f"{sp} (Conspecific)", color=colors[i], markersize=6)
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

def process_production():
    print("Fetching Production via local mount...")
    base_dir = Path("/home/meow/cesnet_data/PEG/Colorado/Detected")
    records = []
    
    # Also search directly in species folders and Pollen_production folders
    search_dirs = [base_dir]
    prod_dir = base_dir / "Pollen_production"
    if prod_dir.exists():
        search_dirs.append(prod_dir)
        
    for sd in search_dirs:
        for sp_dir in sd.iterdir():
            if not sp_dir.is_dir(): continue
            sp = sp_dir.name
            if sp not in ["Cal_chi", "Ran_ado", "Gen_alg", "Sed_lan", "Vio_adu"]: continue
            
            for f in sp_dir.glob("*_details.csv"):
                if not f.name.endswith("_details.csv"): continue
                try:
                    df = pd.read_csv(f)
                    detections = len(df) if not df.empty else 0
                    date_obj = extract_prod_date(f.name)
                    if pd.notna(date_obj):
                        records.append({
                            "Species": sp,
                            "Collection_Date": date_obj,
                            "Detections": detections
                        })
                except:
                    pass

    if records:
        df = pd.DataFrame(records).dropna(subset=["Collection_Date"])
        agg = df.groupby(["Collection_Date", "Species"])["Detections"].agg(["mean", "sem", "count"]).reset_index()
        agg.rename(columns={"mean": "Average_Detections", "sem": "Standard_Error", "count": "N_Samples"}, inplace=True)
        agg.fillna(0, inplace=True)
        
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
        plt.yscale("log")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(title="Species")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("results/pollen_production_timeline.png", dpi=300)
        print("Production plot saved.")

if __name__ == "__main__":
    process_deposition()
    process_production()
