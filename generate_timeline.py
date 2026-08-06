import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

def main():
    # 1. Load data
    df = pd.read_csv('results/Gen_alg_summary.csv')
    df['Collection_Date'] = pd.to_datetime(df['Collection_Date'])
    
    # 2. Aggregate data
    agg_df = df.groupby(['Collection_Date', 'Species'])['Detections'].agg(['mean', 'sem', 'count']).reset_index()
    agg_df.rename(columns={'mean': 'Average_Detections', 'sem': 'Standard_Error', 'count': 'N_Samples'}, inplace=True)
    
    # Fill NaN sem with 0 for plotting
    agg_df['Standard_Error'] = agg_df['Standard_Error'].fillna(0)
    
    # Save aggregated CSV
    agg_df.to_csv('pollen_deposition_aggregated.csv', index=False)
    
    # 3. Create plot
    plt.figure(figsize=(10, 6))
    
    # We will use matplotlib to add error bars easily
    species_list = agg_df['Species'].unique()
    colors = plt.cm.get_cmap("tab10").colors
    
    for i, species in enumerate(species_list):
        subset = agg_df[agg_df['Species'] == species].sort_values('Collection_Date')
        plt.errorbar(subset['Collection_Date'], subset['Average_Detections'], 
                     yerr=subset['Standard_Error'], fmt='-o', 
                     capsize=5, label=species, color=colors[i], markersize=6)
        
    plt.title('Pollen Deposition Over Time', fontsize=16)
    plt.xlabel('Collection Date', fontsize=12)
    plt.ylabel('Average Detections per Image', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Species')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('pollen_deposition_timeline.png', dpi=300)
    print("Total Detections:", df['Detections'].sum())
    print("Files created: pollen_deposition_timeline.png, pollen_deposition_aggregated.csv")

if __name__ == "__main__":
    main()
