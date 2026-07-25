import glob, os

files = glob.glob("/home/meow/cesnet_cloud/bucket/PEG/Colorado/Species_model/Trainig_data/Deposition_Stigmas/Labels/*.txt")
for f in files:
    with open(f, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) > 6:
                coords = [float(p) for p in parts[1:]]
                xs = coords[0::2]
                ys = coords[1::2]
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
                if w < 0.01 or h < 0.01:
                    print(f"File: {os.path.basename(f)}, w: {w:.5f}, h: {h:.5f}")
