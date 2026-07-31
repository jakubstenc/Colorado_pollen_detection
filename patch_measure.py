import sys

with open("src/measure_deposition.py", "r") as f:
    lines = f.readlines()

new_lines = []
in_main = False
skip_models = False

for i, line in enumerate(lines):
    if line.startswith("def main():"):
        in_main = True
        
    # Replace the random.choice logic with sharding and model loading
    if "random.seed()" in line and in_main:
        new_lines.append("    czi_targets.sort(key=lambda x: x[0])\n")
        new_lines.append("    shard_idx = int(os.environ.get('JOB_COMPLETION_INDEX', '0'))\n")
        new_lines.append("    total_shards = 5\n")
        new_lines.append("    my_targets = [t for i, t in enumerate(czi_targets) if i % total_shards == shard_idx]\n")
        new_lines.append("    print(f'Shard {shard_idx}/{total_shards} processing {len(my_targets)} of {len(czi_targets)} images.')\n")
        new_lines.append("    \n")
        new_lines.append("    general_model_path = 'best.pt' if Path('best.pt').exists() else '/app/best.pt' if Path('/app/best.pt').exists() else '/home/meow/Documents/Antigravity/Colorado_pollen_detection/best.pt'\n")
        new_lines.append("    if Path(general_model_path).exists():\n")
        new_lines.append("        general_model = YOLO(general_model_path)\n")
        new_lines.append("    else:\n")
        new_lines.append("        print(f'General model {general_model_path} not found locally!')\n")
        new_lines.append("        return\n")
        new_lines.append("        \n")
        new_lines.append("    print('Downloading latest species model from S3...')\n")
        new_lines.append("    s3.download_file(s3_bucket, 'PEG/Colorado/trained_models/species_classifier/latest.pt', '/tmp/species_latest.pt')\n")
        new_lines.append("    species_model = YOLO('/tmp/species_latest.pt')\n")
        new_lines.append("    species_classes = species_model.names\n")
        new_lines.append("    \n")
        new_lines.append("    for target_key, filename, stigma_species in my_targets:\n")
        new_lines.append("        print(f'\\n--- Processing: {filename} (Stigma Species: {stigma_species}) ---')\n")
        continue

    if "target_key, filename, stigma_species = random.choice(czi_targets)" in line:
        continue
    if 'print(f"Selected: {filename}' in line:
        continue
        
    # Indent everything after random.choice
    if in_main and i > 62 and i < 310:
        # Wait, need to skip the model loading section that is inside the loop originally
        if "    # 3. Load Models" in line:
            skip_models = True
            
        if skip_models:
            if "species_classes = species_model.names" in line:
                skip_models = False
                continue
            continue
            
        # Change returns to continues
        if line.strip() == "return" and not skip_models:
            line = line.replace("return", "continue")
            
        new_lines.append("    " + line)
    else:
        new_lines.append(line)

with open("src/measure_deposition_updated.py", "w") as f:
    f.writelines(new_lines)
