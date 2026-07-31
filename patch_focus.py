import sys

with open("src/measure_deposition.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False

for line in lines:
    if "# 2. Focus Check (run on a center crop" in line or "# 2. Focus Check" in line:
        new_lines.append("    # 2. Focus Check (match focus_check.py logic)\n")
        new_lines.append("    if px_size is not None and px_size < 0.65:\n")
        new_lines.append("        focus_img = cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2), interpolation=cv2.INTER_AREA)\n")
        new_lines.append("    else:\n")
        new_lines.append("        focus_img = rgb\n")
        new_lines.append("    \n")
        new_lines.append("    blur_score = compute_focus_score(focus_img)\n")
        skip = True
        continue
        
    if skip:
        if "is_focused = blur_score >= 150.0" in line:
            skip = False
            new_lines.append(line)
        continue
        
    new_lines.append(line)

with open("src/measure_deposition_focus_fixed.py", "w") as f:
    f.writelines(new_lines)
