import os
import glob
import math

def polygon_area(x, y):
    return 0.5 * abs(sum(x[i]*y[i-1] - x[i-1]*y[i] for i in range(len(x))))

def analyze_labels(lbl_dir):
    for f in glob.glob(os.path.join(lbl_dir, "*.txt"))[:10]:
        print(f"File: {os.path.basename(f)}")
        with open(f, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) > 5:
                    coords = [float(p) for p in parts[1:]]
                    x = coords[0::2]
                    y = coords[1::2]
                    min_x, max_x = min(x), max(x)
                    min_y, max_y = min(y), max(y)
                    w = max_x - min_x
                    h = max_y - min_y
                    if w == 0 or h == 0: continue
                    aspect_ratio = w / h
                    
                    area = polygon_area(x, y)
                    bbox_area = w * h
                    extent = area / bbox_area
                    
                    print(f"  - Box WxH: {w:.3f}x{h:.3f}, Aspect Ratio: {aspect_ratio:.2f}, Extent: {extent:.3f}")

analyze_labels(os.path.expanduser('~/cesnet_cloud/bucket/PEG/Colorado/Staged_area/Species_curated/Deposition_Stigmas/Discarded/Labels/'))
