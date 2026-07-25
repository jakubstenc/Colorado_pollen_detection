import os
import shutil
import glob
import threading
from flask import Flask, render_template_string, send_file, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.expanduser("~/cesnet_cloud/bucket/PEG/Colorado/Species_model")
DEST_DIR = os.path.expanduser("~/cesnet_cloud/bucket/PEG/Colorado/Curated_Retrain_Data")

DEST_IMG_DIR = os.path.join(DEST_DIR, "images")
DEST_LBL_DIR = os.path.join(DEST_DIR, "labels")

CLS_DEST_DIR = os.path.expanduser("~/cesnet_cloud/bucket/PEG/Colorado/dataset_species_curated/train")
STAGED_AREA_DIR = os.path.expanduser("~/cesnet_cloud/bucket/PEG/Colorado/Staged_area/Species_curated")

os.makedirs(DEST_IMG_DIR, exist_ok=True)
os.makedirs(DEST_LBL_DIR, exist_ok=True)

os.makedirs(DEST_LBL_DIR, exist_ok=True)

pending_cache = None
s3_client = None

def get_pending_images():
    global pending_cache
    if not pending_cache:
        print("⏳ Scraping metadata directory structure from S3 using Boto3...")
        import boto3
        from botocore.client import Config
        s3_endpoint  = "https://s3.cl4.du.cesnet.cz"
        s3_bucket    = "bucket"
        access_key   = "1Y920BKC0SAWPNDE8RD6"
        secret_key   = "SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD"

        global s3_client
        if s3_client is None:
            s3_client = boto3.client(
                "s3",
                endpoint_url="https://s3.cl4.du.cesnet.cz",
                aws_access_key_id="1Y920BKC0SAWPNDE8RD6",
                aws_secret_access_key="SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD",
                config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False}),
            )
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket="bucket", Prefix="PEG/Colorado/Species_model/")
        
        all_viz = []
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if ('Vizualization/' in key and key.endswith('_viz.jpg')) or ('Negatives/' in key and key.endswith('.jpg')):
                        local_path = os.path.expanduser(f"~/cesnet_cloud/bucket/{key}")
                        if "/Staged_area/" not in local_path:
                            all_viz.append(local_path)
        pending_cache = sorted(all_viz)
        print(f"✅ Fast scraped {len(pending_cache)} pending visualization tiles.")
    return pending_cache

def resolve_source_files(viz_path):
    if "/Negatives/" in viz_path:
        species_dir = os.path.dirname(os.path.dirname(viz_path))
        base_stem = os.path.basename(viz_path).replace(".jpg", "")
        return viz_path, "", base_stem, species_dir
    else:
        viz_dir = os.path.dirname(viz_path)
        species_dir = os.path.dirname(viz_dir)
        filename = os.path.basename(viz_path)
        base_stem = filename.replace("_viz.jpg", "")
        img_path = os.path.join(species_dir, "Images", base_stem + ".jpg")
        lbl_path = os.path.join(species_dir, "Labels", base_stem + ".txt")
        return img_path, lbl_path, base_stem, species_dir

def mark_as_reviewed(viz_path, action_type):
    img_path, lbl_path, base_stem, species_dir = resolve_source_files(viz_path)
    
    # Isolate Skipped objects into a Discarded directory per the user's request
    if action_type == "skip":
        target_folder_name = "Discarded"
    elif action_type == "reject":
        target_folder_name = "Negatives"
    else:
        target_folder_name = "Reviewed"
        
    species_name = os.path.basename(species_dir.rstrip('/'))
    rev_dir = os.path.join(STAGED_AREA_DIR, species_name, target_folder_name)
    os.makedirs(os.path.join(rev_dir, "Images"), exist_ok=True)
    os.makedirs(os.path.join(rev_dir, "Labels"), exist_ok=True)
    os.makedirs(os.path.join(rev_dir, "Vizualization"), exist_ok=True)
    
    if os.path.exists(viz_path): shutil.move(viz_path, os.path.join(rev_dir, "Vizualization", os.path.basename(viz_path)))
    if os.path.exists(img_path): shutil.move(img_path, os.path.join(rev_dir, "Images", os.path.basename(img_path)))
    if os.path.exists(lbl_path): 
        target_lbl = os.path.join(rev_dir, "Labels", os.path.basename(lbl_path))
        shutil.move(lbl_path, target_lbl)
        if action_type == "reject":
            # Exterminate hallucinated polygons strictly in the staging area memory
            open(target_lbl, 'w').close()

undo_stack = []

manifest_path = os.path.expanduser("~/Documents/Antigravity/Colorado_pollen_detection/src/species_manifest.csv")
GLOBAL_SPECIES = []
if os.path.exists(manifest_path):
    with open(manifest_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if parts and parts[0] != 'species_code':
                GLOBAL_SPECIES.append(parts[0])

@app.route("/")
def index():
    pending = get_pending_images()
    
    if not pending:
        return "<body style='background:#121212;'><h1 style='color:white; font-family:sans-serif; text-align:center; padding-top:20%'>✅ All items completely verified and mapped! Nothing left in queue.</h1></body>"
    
    req_species = request.args.get("species", "")
    req_pred_type = request.args.get("pred_type", "positive")
    req_target_species = request.args.get("target_species", "")
    req_search_id = request.args.get("search_id", "")
    req_exact_path = request.args.get("exact_path", "")

    if req_exact_path:
        if "/Negatives/" in req_exact_path:
            req_pred_type = "negative"
        elif "/Vizualization/" in req_exact_path:
            req_pred_type = "positive"

    filtered_pending = pending
    if req_pred_type == "positive":
        filtered_pending = [p for p in filtered_pending if "/Vizualization/" in p]
    elif req_pred_type == "negative":
        filtered_pending = [p for p in filtered_pending if "/Negatives/" in p]

    filtered_pending = [p for p in filtered_pending if f"/{req_species}/" in p] if req_species else filtered_pending
    
    if req_target_species:
        filtered_pending = [p for p in filtered_pending if req_target_species.lower() in os.path.basename(p).lower()]
        
    if req_search_id:
        filtered_pending = [p for p in filtered_pending if req_search_id.lower() in os.path.basename(p).lower()]
    
    if not filtered_pending:
        filtered_pending = pending
        req_species = ""
        req_search_id = ""
        
    if req_exact_path and req_exact_path in filtered_pending:
        idx = filtered_pending.index(req_exact_path)
    else:
        try:
            idx = int(request.args.get("idx", 0))
        except ValueError:
            idx = 0
            
        if idx >= len(filtered_pending) or idx < 0:
            idx = 0

    curr_viz = filtered_pending[idx]
    img_path, lbl_path, base_stem, species_dir = resolve_source_files(curr_viz)
    
    species_name = os.path.basename(species_dir)
    
    target_override = species_name
    for sp in GLOBAL_SPECIES:
        if sp.lower() in base_stem.lower():
            target_override = sp
            break
            
    labels_data = []
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
            for line in lines:
                parts = line.split()
                if len(parts) >= 5:
                    if len(parts) in [5, 6]:
                        x_cen, y_cen, w, h = map(float, parts[1:5])
                        x_min = x_cen - w/2
                        y_min = y_cen - h/2
                        conf = parts[5] if len(parts) == 6 else ''
                    else:
                        coords = [float(p) for p in parts[1:]]
                        xs = coords[0::2]
                        ys = coords[1::2]
                        x_min, max_x = min(xs), max(xs)
                        y_min, max_y = min(ys), max(ys)
                        w = max_x - x_min
                        h = max_y - y_min
                        conf = ''
                        
                    labels_data.append({
                        'class_id': parts[0],
                        'x_min': x_min,
                        'y_min': y_min,
                        'w': w,
                        'h': h,
                        'conf': conf,
                        'raw': line
                    })
    num_objs = len(labels_data)
            
    all_species = sorted(list(set([os.path.basename(os.path.dirname(os.path.dirname(p))) for p in pending])))
    
    return render_template_string(HTML_TEMPLATE, 
                                  viz_path=curr_viz, 
                                  img_path=img_path,
                                  labels_data=labels_data,
                                  remaining=len(filtered_pending),
                                  total_remaining=len(pending),
                                  species_name=species_name,
                                  num_objs=num_objs,
                                  base_stem=base_stem,
                                  all_species=all_species,
                                  req_species=req_species,
                                  req_target_species=req_target_species,
                                  req_search_id=req_search_id,
                                  req_pred_type=req_pred_type,
                                  target_override=target_override,
                                  global_species=GLOBAL_SPECIES,
                                  idx=idx,
                                  undo_stack=undo_stack,
                                  undo_available=len(undo_stack) > 0)

@app.route("/image")
def serve_image():
    path = request.args.get("path")
    if not os.path.exists(path):
        # Auto-download from S3 if missing locally
        rel_path = os.path.relpath(path, os.path.expanduser("~/cesnet_cloud/bucket"))
        if rel_path.startswith("PEG/"):
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                global s3_client
                if s3_client is None:
                    import boto3
                    from botocore.client import Config
                    s3_client = boto3.client("s3", endpoint_url="https://s3.cl4.du.cesnet.cz", aws_access_key_id="1Y920BKC0SAWPNDE8RD6", aws_secret_access_key="SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD", config=Config(signature_version="s3v4", s3={"payload_signing_enabled": False}))
                s3_client.download_file("bucket", rel_path, path)
                print(f"Auto-downloaded missing file from S3: {rel_path}")
            except Exception as e:
                print(f"Failed to auto-download {rel_path} from S3: {e}")
                return "File not found", 404
    return send_file(path)

@app.route("/raw_image")
def serve_raw_image():
    path = request.args.get("path")
    return send_file(path)

def process_action_bg(action_type, viz_path, img_path, lbl_path, base_stem, dest_img_dir, dest_lbl_dir, data):
    keep_labels = data.get("keep_labels", None)
    override_species = data.get("override_species", "Unknown")
    
    # Overwrite the original label in place if we are approving partial labels
    if action_type == "approve" and keep_labels is not None and os.path.exists(lbl_path):
        with open(lbl_path, 'w') as f:
            f.write("\n".join(keep_labels) + ("\n" if keep_labels else ""))

    if action_type in ["approve", "reject"]:
        if os.path.exists(img_path):
            shutil.copy(img_path, os.path.join(dest_img_dir, base_stem + ".jpg"))
            
            # --- CLASSIFICATION CROP EXTRACTION ---
            if action_type == "approve" and keep_labels:
                import cv2
                import numpy as np
                img = cv2.imread(img_path)
                if img is not None:
                    H, W = img.shape[:2]
                    out_cls_dir = os.path.join(CLS_DEST_DIR, override_species)
                    os.makedirs(out_cls_dir, exist_ok=True)
                    
                    for idx, line in enumerate(keep_labels):
                        parts = line.split()
                        if len(parts) >= 5:
                            if len(parts) in [5, 6]:
                                x_cen, y_cen, w, h = map(float, parts[1:5])
                                x_min = (x_cen - w/2) * W
                                y_min = (y_cen - h/2) * H
                                box_w = w * W
                                box_h = h * H
                            else:
                                coords = [float(p) for p in parts[1:]]
                                xs = [x * W for x in coords[0::2]]
                                ys = [y * H for y in coords[1::2]]
                                x_min, max_x = min(xs), max(xs)
                                y_min, max_y = min(ys), max(ys)
                                box_w = max_x - x_min
                                box_h = max_y - y_min
                                
                            pad_x, pad_y = box_w * 0.1, box_h * 0.1
                            x1 = max(0, int(x_min - pad_x))
                            y1 = max(0, int(y_min - pad_y))
                            x2 = min(W, int(x_min + box_w + pad_x))
                            y2 = min(H, int(y_min + box_h + pad_y))
                            
                            crop = img[y1:y2, x1:x2]
                            if crop.shape[0] > 10 and crop.shape[1] > 10:
                                crop_path = os.path.join(out_cls_dir, f"{base_stem}_crop_{idx}.jpg")
                                cv2.imwrite(crop_path, crop)
        
        dest_lbl = os.path.join(dest_lbl_dir, base_stem + ".txt")
        if action_type == "approve":
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, dest_lbl)
        elif action_type == "reject":
            open(dest_lbl, 'w').close()
            
    mark_as_reviewed(viz_path, action_type)

@app.route("/action", methods=["POST"])
def action():
    data = request.json
    action_type = data.get("action")
    viz_path = data.get("path")
    
    if not os.path.exists(viz_path):
        return jsonify({"status": "error", "msg": "File not found"}), 404
        
    img_path, lbl_path, base_stem, species_dir = resolve_source_files(viz_path)
    
    # Snapshot state for Undo tracking
    history = {
        "action": action_type,
        "viz_path": viz_path,
        "img_path": img_path,
        "lbl_path": lbl_path,
        "base_stem": base_stem,
        "species_dir": species_dir
    }
    
    t = threading.Thread(target=process_action_bg, args=(action_type, viz_path, img_path, lbl_path, base_stem, DEST_IMG_DIR, DEST_LBL_DIR, data))
    t.start()
    
    global pending_cache
    if pending_cache is not None and viz_path in pending_cache:
        pending_cache.remove(viz_path)
        
    undo_stack.append(history)
    return jsonify({"status": "success"})

def process_undo_bg(last, base_stem, dest_img_dir, dest_lbl_dir):
    species_dir = last["species_dir"]
    species_name = os.path.basename(species_dir.rstrip('/'))
    
    if last["action"] == "skip":
        rev_dir = os.path.join(STAGED_AREA_DIR, species_name, "Discarded")
    elif last["action"] == "reject":
        rev_dir = os.path.join(STAGED_AREA_DIR, species_name, "Negatives")
    else:
        rev_dir = os.path.join(STAGED_AREA_DIR, species_name, "Reviewed")
    
    # Restore from Reviewed/ cache natively back into the root queue
    rev_viz = os.path.join(rev_dir, "Vizualization", os.path.basename(last["viz_path"]))
    if os.path.exists(rev_viz): shutil.move(rev_viz, last["viz_path"])
        
    rev_img = os.path.join(rev_dir, "Images", os.path.basename(last["img_path"]))
    if os.path.exists(rev_img): shutil.move(rev_img, last["img_path"])
        
    rev_lbl = os.path.join(rev_dir, "Labels", os.path.basename(last["lbl_path"]))
    if os.path.exists(rev_lbl): shutil.move(rev_lbl, last["lbl_path"])
    
    # If the user hard-accepted or hard-rejected to Retrain Dataset, sever it.
    if last["action"] in ["approve", "reject"]:
        dest_img = os.path.join(dest_img_dir, base_stem + ".jpg")
        if os.path.exists(dest_img): os.remove(dest_img)
        dest_lbl = os.path.join(dest_lbl_dir, base_stem + ".txt")
        if os.path.exists(dest_lbl): os.remove(dest_lbl)
        
    global pending_cache
    if pending_cache is not None:
        pending_cache = None

@app.route("/prepare_roboflow", methods=["POST"])
def prepare_roboflow():
    import cv2
    import numpy as np
    
    out_dir = os.path.expanduser("~/cesnet_cloud/bucket/PEG/Colorado/Roboflow_Export")
    out_img = os.path.join(out_dir, "images")
    out_lbl = os.path.join(out_dir, "labels")
    
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    
    lbl_files = glob.glob(os.path.join(STAGED_AREA_DIR, "*", "Discarded", "Labels", "*.txt"))
    
    exported = 0
    for lbl_f in lbl_files:
        base = os.path.basename(lbl_f)
        # The image path is parallel to the label path
        img_f = lbl_f.replace("/Labels/", "/Images/").replace(".txt", ".jpg")
        if not os.path.exists(img_f):
            continue
            
        with open(lbl_f, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            continue
            
        out_lbl_f = os.path.join(out_lbl, base)
        out_img_f = os.path.join(out_img, base.replace(".txt", ".jpg"))
        
        shutil.copy(img_f, out_img_f)
            
        img = cv2.imread(img_f)
        if img is None: continue
        h, w = img.shape[:2]
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 5:
                class_id = parts[0]
                coords = [float(p) for p in parts[1:]]
                x = [int(c * w) for c in coords[0::2]]
                y = [int(c * h) for c in coords[1::2]]
                
                pad = 20
                min_x = max(0, min(x) - pad)
                max_x = min(w, max(x) + pad)
                min_y = max(0, min(y) - pad)
                max_y = min(h, max(y) + pad)
                
                crop = img[min_y:max_y, min_x:max_x]
                if crop.size > 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest = max(contours, key=cv2.contourArea)
                        if len(largest) >= 3:
                            norm_pts = []
                            for pt in largest:
                                px = (pt[0][0] + min_x) / w
                                py = (pt[0][1] + min_y) / h
                                norm_pts.append(f"{px:.6f} {py:.6f}")
                            
                            new_lines.append(f"{class_id} " + " ".join(norm_pts) + "\n")
                            continue
            new_lines.append(line)
            
        with open(out_lbl_f, 'w') as f:
            f.writelines(new_lines)
        exported += 1
            
    return jsonify({"status": "success", "msg": f"Successfully snapped polygons and exported {exported} files to Roboflow_Export!"})

@app.route("/undo", methods=["POST"])
def undo():
    if not undo_stack:
        return jsonify({"status": "error", "msg": "Nothing to undo"}), 400
        
    last = undo_stack.pop()
    base_stem = last["base_stem"]
    
    t = threading.Thread(target=process_undo_bg, args=(last, base_stem, DEST_IMG_DIR, DEST_LBL_DIR))
    t.start()
    
    global pending_cache
    if pending_cache is not None:
        pending_cache.append(last["viz_path"])
        pending_cache.sort()
        
    return jsonify({"status": "success"})

def get_slide_data():
    pending = get_pending_images()
    slides = {}
    for p in pending:
        filename = os.path.basename(p)
        base = filename.replace("_viz.jpg", "").replace(".jpg", "")
        if "_x" in base and "_y" in base:
            slide_name = base.split("_x")[0]
            try:
                x = int(base.split("_x")[1].split("_y")[0])
                y = int(base.split("_y")[1])
            except:
                continue
            if slide_name not in slides:
                slides[slide_name] = []
            slides[slide_name].append({'path': p, 'x': x, 'y': y, 'filename': filename})
    return slides

OVERVIEW_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Slide Overview</title>
    <style>
        body { font-family: sans-serif; background: #121212; color: #eee; padding: 40px; }
        a { color: #4db8ff; text-decoration: none; font-size: 18px; }
        a:hover { text-decoration: underline; }
        .slide-item { background: #1e1e1e; padding: 20px; margin-bottom: 10px; border-radius: 8px; border: 1px solid #333; }
        .count { color: #888; font-size: 14px; margin-left: 10px; }
        h1 { font-weight: 300; margin-bottom: 30px; }
        .btn { background: #333; color: white; padding: 10px 20px; border-radius: 5px; margin-bottom: 20px; display: inline-block; border: 1px solid #555; text-decoration: none; }
        .btn:hover { background: #444; }
    </style>
</head>
<body>
    <a href="/" class="btn">Back to Linear Queue</a>
    <h1>Macro Slide Overview</h1>
    {% for name, tiles in slides.items() %}
        <div class="slide-item">
            <a href="/slide/{{ name }}">{{ name }}</a>
            <span class="count">({{ tiles|length }} pending sectors)</span>
        </div>
    {% endfor %}
</body>
</html>
"""

SLIDE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ slide_name }} Overview</title>
    <style>
        body { font-family: sans-serif; background: #121212; color: #eee; padding: 20px; text-align: center; }
        .grid-container { position: relative; margin: 0 auto; background: #000; border: 1px solid #333; width: {{ max_w }}px; height: {{ max_h }}px; }
        .tile { position: absolute; border: 1px solid rgba(255,255,255,0.2); box-sizing: border-box; cursor: pointer; transition: transform 0.1s; }
        .tile:hover { transform: scale(1.1); z-index: 10; border-color: #00ff00; box-shadow: 0 0 10px #00ff00; }
        .tile img { width: 100%; height: 100%; display: block; object-fit: cover; }
        h2 { font-weight: 300; margin-bottom: 20px; }
        .btn { background: #333; color: white; padding: 10px 20px; border-radius: 5px; text-decoration: none; margin-bottom: 20px; display: inline-block; border: 1px solid #555; }
        .btn:hover { background: #444; }
    </style>
</head>
<body>
    <a href="/overview" class="btn">Back to Overview List</a>
    <h2>Preview: {{ slide_name }}</h2>
    <div class="grid-container">
        {% for t in tiles %}
            <a href="/?search_id={{ slide_name|urlencode }}&exact_path={{ t.path|urlencode }}">
                <div class="tile" style="left: {{ t.x * scale }}px; top: {{ t.y * scale }}px; width: {{ 640 * scale }}px; height: {{ 640 * scale }}px;" title="{{ t.filename }}">
                    <img src="/image?path={{ t.path|urlencode }}" loading="lazy" />
                </div>
            </a>
        {% endfor %}
    </div>
</body>
</html>
"""

@app.route("/overview")
def overview():
    slides = get_slide_data()
    return render_template_string(OVERVIEW_TEMPLATE, slides=slides)

@app.route("/slide/<slide_name>")
def slide_view(slide_name):
    slides = get_slide_data()
    if slide_name not in slides:
        return "Slide not found", 404
    tiles = slides[slide_name]
    max_x = max(t['x'] for t in tiles) + 640
    max_y = max(t['y'] for t in tiles) + 640
    
    # Scale factor to fit within a reasonable viewport max width/height
    scale = min(800.0 / max_y, 1200.0 / max_x)
    if scale > 0.15: scale = 0.15
    
    return render_template_string(SLIDE_TEMPLATE, slide_name=slide_name, tiles=tiles, scale=scale, max_w=max_x*scale, max_h=max_y*scale)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Active Learning UI</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #121212; color: #eeeeee; text-align: center; margin:0; padding:0; display:flex;}
        .sidebar { width: 320px; background: #1e1e1e; height: 100vh; box-sizing: border-box; padding: 25px 20px; border-right: 1px solid #333; text-align: left; display:flex; flex-direction:column; overflow-y:auto; box-shadow: 2px 0 10px rgba(0,0,0,0.5); z-index:10;}
        .right-sidebar { width: 280px; background: #1e1e1e; height: 100vh; box-sizing: border-box; padding: 25px 15px; border-left: 1px solid #333; text-align: left; display:flex; flex-direction:column; overflow-y:auto; box-shadow: -2px 0 10px rgba(0,0,0,0.5); z-index:10;}
        .main-content { flex: 1; height: 100vh; overflow-y: auto; display:flex; flex-direction:column; align-items:center; justify-content:center;}
        
        h2 { margin-bottom: 25px; margin-top:5px; color:#bbbbbb; font-weight: 300; font-size:24px; text-align:center;}
        .image-box { padding: 10px; background: #1e1e1e; border-radius: 10px; display:inline-block; border: 1px solid #333; box-shadow: 0 4px 15px rgba(0,0,0,0.5);}
        img { max-width: 100%; max-height: 75vh; border-radius: 5px; display:block;}
        
        .bbox { position: absolute; border: 2px dashed rgba(255,255,255,0.2); background: transparent; cursor: pointer; border-radius: 4px; box-sizing: border-box; }
        .bbox.disabled { border-color: #ff0000; background: rgba(255, 0, 0, 0.4); opacity: 0.8; }
        .bbox:hover { background: rgba(0, 255, 0, 0.15); border-color: #00ff00; }
        .bbox.disabled:hover { background: rgba(255, 0, 0, 0.5); border-color: #ff0000; }
        
        .controls { margin-top: 15px; display: flex; justify-content: center; gap: 15px; }
        button { padding: 14px 20px; font-size: 15px; border: none; border-radius: 6px; cursor: pointer; color: white; font-weight: 600; white-space:nowrap; transition: all 0.2s;}
        .btn-approve { background: #22a042; box-shadow: 0 4px #126325;}
        .btn-reject { background: #d63346; box-shadow: 0 4px #8b1c28;}
        .btn-skip { background: #5a6268; box-shadow: 0 4px #363b3e;}
        
        .btn-nav { background: #2b78e4; padding: 10px 15px; font-size:14px; box-shadow: 0 3px #18488e; width: 48%; }
        .nav-container { display:flex; justify-content:space-between; margin-top: 15px; width: 100%; max-width: 600px;}
        
        button:active { transform: translateY(4px); box-shadow: 0 0 transparent;}
        button:hover { filter: brightness(1.1); }
        .shortcut { color: #ccc; background:#333; padding: 3px 8px; border-radius: 4px; font-size: 13px; font-family: monospace; display:block; margin-top:5px;}
        
        .panel-section { background: #2a2a2a; border-radius: 8px; padding: 15px; margin-bottom: 15px; border-left: 4px solid #2b78e4;}
        .panel-label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; font-weight: bold;}
        .panel-value { font-size: 18px; color: #fff; font-weight: 500; word-break: break-all;}
        
        select { width: 100%; padding: 10px; background: #333; color: white; border: 1px solid #555; border-radius: 5px; font-size: 16px; margin-top: 5px; outline:none;}
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>Data Inspector</h2>
        
        <div class="panel-section">
            <div class="panel-label">Prediction Type</div>
            <select id="pred-type-select" onchange="applyFilters()">
                <option value="all" {% if req_pred_type == 'all' %}selected{% endif %}>All (Mixed)</option>
                <option value="positive" {% if req_pred_type == 'positive' %}selected{% endif %}>Suspected Pollen</option>
                <option value="negative" {% if req_pred_type == 'negative' %}selected{% endif %}>Hard Negatives</option>
            </select>
        </div>

        <div class="panel-section">
            <div class="panel-label">Filter by Species Prediction</div>
            <select id="species-select" onchange="applyFilters()">
                <option value="">-- All Species --</option>
                {% for sp in all_species %}
                <option value="{{ sp }}" {% if req_species == sp %}selected{% endif %}>{{ sp }}</option>
                {% endfor %}
            </select>
        </div>

        <div class="panel-section">
            <div class="panel-label">Filter by True Source Species</div>
            <select id="target-species-select" onchange="applyFilters()">
                <option value="">-- All True Species (From Filename) --</option>
                {% for sp in global_species %}
                <option value="{{ sp }}" {% if req_target_species == sp %}selected{% endif %}>{{ sp }}</option>
                {% endfor %}
            </select>
        </div>

        <div class="panel-section">
            <div class="panel-label">Target ID (Find exact region)</div>
            <input type="text" id="search-id-input" value="{{ req_search_id }}" placeholder="e.g. x000100_y000200" style="width: 100%; box-sizing: border-box; padding: 10px; background:#444; color:white; border: 1px solid #2b78e4; border-radius: 5px; font-size: 14px; margin-top: 5px; outline:none;" onkeydown="if(event.key === 'Enter') applyFilters()">
            <div style="font-size:11px; color:#aaa; margin-top:4px;">(Hit ENTER to isolate exact tile preview)</div>
        </div>

        <div class="panel-section">
            <div class="panel-label">Current Species</div>
            <div class="panel-value" style="color: #4CAF50;">{{ species_name }}</div>
        </div>

        <div class="panel-section">
            <div class="panel-label">Objects Detected</div>
            <div class="panel-value" style="color: #FFC107; font-size: 24px;"><span id="active-count">{{ num_objs }}</span> / {{ num_objs }} targets</div>
            <div style="font-size:11px; color:#777; margin-top:4px;">(Click bounding boxes on the image to toggle/correct labels)</div>
        </div>
        
        <button style="margin-top:auto; background: #e67e22; box-shadow: 0 4px #b8651b; width:100%; margin-bottom: 20px;" onclick="prepareRoboflow()">
            ✨ Prepare for Roboflow (Snap Polygons)
        </button>

        <div class="panel-section">
            <div class="panel-label">Correct Species Identity</div>
            <input list="override-species-list" id="override-species" value="{{ target_override }}" style="width: 100%; box-sizing: border-box; padding: 10px; background:#444; color:white; border: 1px solid #2b78e4; border-radius: 5px; font-size: 16px; margin-top: 5px; font-weight:bold; outline:none;">
            <datalist id="override-species-list">
                {% for sp in global_species %}
                <option value="{{ sp }}">
                {% endfor %}
            </datalist>
            <div style="font-size:11px; color:#aaa; margin-top:4px;">(Auto-detected from Source Filename if match exists)</div>
        </div>

        <div class="panel-section">
            <div class="panel-label">Source Identity</div>
            <div class="panel-value" style="font-size:14px; color:#aaa;">{{ base_stem }}</div>
        </div>
        
        <div class="panel-section">
            <div class="panel-label">Queue Framework</div>
            <div class="panel-value"><span style="color:#d63346">{{ remaining }}</span> images in view <br><span style="font-size:12px;color:#888">({{ total_remaining }} total dynamically remaining)</span></div>
        </div>
        
    </div>
    
    <div class="main-content">
        <div class="image-box" style="display:inline-block; max-width: 70vw; max-height: 85vh; overflow: auto; position: relative;">
            <div style="text-align: center; position: sticky; top: 0; z-index: 100; background: rgba(30,30,30,0.95); padding: 10px; border-radius: 8px; margin-bottom: 10px; display:flex; justify-content:center; gap:30px; border-bottom: 1px solid #444; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                <div style="display:flex; align-items:center;">
                    <label for="zoom-slider" style="color:#ddd; font-size:14px; margin-right:8px; font-weight:bold;">🔍 Zoom:</label>
                    <input type="range" id="zoom-slider" min="100" max="600" step="10" value="100" style="width: 150px; cursor: pointer;" oninput="let w = this.value + '%'; document.getElementById('raw-image').style.width = w; document.getElementById('raw-image').style.maxWidth = 'none'; document.getElementById('raw-image').style.maxHeight = 'none';">
                </div>
                <div style="font-weight: 600; color: #4db8ff; margin-bottom: 10px; font-size:18px;">
                    ACTIVE LEARNING UI
                </div>
                <a href="/overview" style="display:block; padding: 8px; background: #333; color: white; border-radius: 4px; text-align: center; text-decoration: none; border: 1px solid #555; margin-bottom: 20px; font-size: 14px;">
                    🔍 Slide Overview Grid
                </a>
                <div style="display:flex; align-items:center;">
                    <label for="opacity-slider" style="color:#ddd; font-size:14px; margin-right:8px; font-weight:bold;">👁️ Bounding Box Opacity:</label>
                    <input type="range" id="opacity-slider" min="0" max="1" step="0.05" value="1" style="width: 150px; cursor: pointer;" oninput="document.getElementById('main-image').style.opacity = this.value; document.getElementById('labels-container').style.opacity = this.value;">
                </div>
            </div>
            
            <div id="zoom-container" style="position:relative; display:inline-block; transform-origin: top center;">
                <img src="/image?path={{img_path}}" alt="Raw Image" id="raw-image" style="display:block; width:100%; transition: width 0.1s ease-out;">
                <img src="/image?path={{viz_path}}" alt="Visualization" id="main-image" style="position:absolute; top:0; left:0; width:100%; height:100%; max-width:none; max-height:none;">
                <div id="labels-container" style="position:absolute; top:0; left:0; width:100%; height:100%;"></div>
            </div>
        </div>
        
        <div class="nav-container" style="flex-wrap: wrap; gap: 10px; justify-content: center;">
            <button class="btn-nav" onclick="navigate(-1)">⬅ Previous Frame (Left Arrow)</button>
            <button class="btn-nav" onclick="navigate(1)">Next Frame ➡ (Up Arrow)</button>
            
            <div style="width: 100%; display: flex; align-items: center; justify-content: center; gap: 10px; margin-top: 5px; background: #2a2a2a; padding: 10px; border-radius: 8px;">
                <span style="color:#aaa; font-size: 14px;">Jump to image:</span>
                <input type="number" id="jump-idx" value="{{ idx + 1 }}" min="1" max="{{ remaining }}" style="width: 80px; padding: 8px; background: #333; color: white; border: 1px solid #555; border-radius: 5px; text-align: center; font-size: 14px; outline: none;" onkeydown="if(event.key === 'Enter') jumpToIdx()">
                <span style="color:#aaa; font-size: 14px;">of {{ remaining }}</span>
                <button onclick="jumpToIdx()" style="background: #2b78e4; padding: 8px 15px; font-size: 14px; box-shadow: none; width: auto; color: white; border: none; border-radius: 6px; cursor: pointer;">GO</button>
            </div>
        </div>
        
        <div class="controls" style="margin-top:25px;">
            <button class="btn-approve" onclick="sendAction('approve')">Valid Pollen <span class="shortcut">Right Arrow</span></button>
            <button class="btn-reject" onclick="sendAction('reject')">Reject Background <span class="shortcut">Down Arrow</span></button>
            <button class="btn-skip" onclick="sendAction('skip')">Delete from UI Queue<span class="shortcut">Spacebar</span></button>
        </div>
    </div>

    <div class="right-sidebar">
        <h2 style="font-size:18px;">History Queue</h2>
        {% if undo_stack|length == 0 %}
            <div style="color:#888; font-size:13px; text-align:center; margin-top:20px;">No recent decisions.</div>
        {% else %}
            {% for item in undo_stack|reverse %}
            <div class="panel-section" style="border-left: 4px solid {% if item.action == 'approve' %}#22a042{% elif item.action == 'reject' %}#d63346{% else %}#5a6268{% endif %}; padding: 10px; margin-bottom: 10px;">
                <div style="font-size:11px; color:#aaa; margin-bottom: 3px; font-weight:bold; text-transform:uppercase;">{{ item.action }}</div>
                <div style="font-size:12px; color:#ddd; word-break: break-all; margin-bottom: 5px;" title="{{ item.base_stem }}">{{ item.base_stem }}</div>
                <img src="/image?path={{ item.viz_path }}" style="width:100%; height:auto; border-radius:4px; margin-bottom:8px; border:1px solid #444;">
                {% if loop.index0 == 0 %}
                <button onclick="sendUndo()" style="background: #e67e22; padding: 6px 12px; font-size:12px; border-radius:4px; border:none; color:white; cursor:pointer; width:100%; box-shadow:0 2px #a85812;">Undo Latest</button>
                {% else %}
                <div style="font-size:10px; color:#666; text-align:center;">(Undo previous first)</div>
                {% endif %}
            </div>
            {% endfor %}
        {% endif %}
    </div>

    <script>
        let isProcessing = false;
        const labelsData = {{ labels_data|tojson|safe }};
        let activeLabels = new Array(labelsData.length).fill(true);
        
        // Smart Auto-Reject: If the label is extremely small, assume it is an artifact/false positive.
        // The user can still manually click it to re-enable it if it was a mistake.
        for (let i = 0; i < labelsData.length; i++) {
            if (labelsData[i].w * 100 < 1.5 && labelsData[i].h * 100 < 1.5) {
                activeLabels[i] = false;
            }
        }

        function renderBoxes() {
            const container = document.getElementById("labels-container");
            container.innerHTML = "";
            let activeCount = 0;
            
            labelsData.forEach((lbl, idx) => {
                const box = document.createElement("div");
                box.className = "bbox";
                if (!activeLabels[idx]) {
                    box.classList.add("disabled");
                } else {
                    activeCount++;
                }
                
                let origWidth = lbl.w * 100;
                let origHeight = lbl.h * 100;
                
                let width = Math.max(origWidth, 3.0);
                let height = Math.max(origHeight, 3.0);
                
                let left = (lbl.x_min * 100) - (width - origWidth) / 2;
                let top = (lbl.y_min * 100) - (height - origHeight) / 2;
                
                box.style.left = left + "%";
                box.style.top = top + "%";
                box.style.width = width + "%";
                box.style.height = height + "%";
                
                if (lbl.conf) {
                    box.title = "Conf: " + lbl.conf;
                }
                
                box.onclick = function() {
                    activeLabels[idx] = !activeLabels[idx];
                    renderBoxes();
                };
                container.appendChild(box);
            });
            
            const countEl = document.getElementById("active-count");
            if(countEl) countEl.innerText = activeCount;
        }
        
        window.addEventListener('load', renderBoxes);
        
        function applyFilters() {
            const pt = document.getElementById("pred-type-select").value;
            const sp = document.getElementById("species-select").value;
            const tsp = document.getElementById("target-species-select").value;
            const searchId = document.getElementById("search-id-input").value;
            window.location.href = "/?species=" + encodeURIComponent(sp) + "&target_species=" + encodeURIComponent(tsp) + "&search_id=" + encodeURIComponent(searchId) + "&pred_type=" + encodeURIComponent(pt);
        }
        
        function jumpToIdx() {
            const val = parseInt(document.getElementById("jump-idx").value, 10);
            if (!isNaN(val) && val >= 1) {
                const nextIdx = val - 1;
                const pt = document.getElementById("pred-type-select").value;
                const sp = document.getElementById("species-select").value;
                const tsp = document.getElementById("target-species-select").value;
                const searchId = document.getElementById("search-id-input").value;
                window.location.href = "/?species=" + encodeURIComponent(sp) + "&target_species=" + encodeURIComponent(tsp) + "&search_id=" + encodeURIComponent(searchId) + "&pred_type=" + encodeURIComponent(pt) + "&idx=" + nextIdx;
            }
        }
        
        function navigate(dir) {
            let nextIdx = {{ idx }} + dir;
            if(nextIdx < 0) nextIdx = 0;
            const pt = document.getElementById("pred-type-select").value;
            const sp = document.getElementById("species-select").value;
            const tsp = document.getElementById("target-species-select").value;
            const searchId = document.getElementById("search-id-input").value;
            window.location.href = "/?species=" + encodeURIComponent(sp) + "&target_species=" + encodeURIComponent(tsp) + "&search_id=" + encodeURIComponent(searchId) + "&pred_type=" + encodeURIComponent(pt) + "&idx=" + nextIdx;
        }

        function sendUndo() {
            if(isProcessing) return;
            isProcessing = true;
            fetch("/undo", { method: "POST" })
            .then(r => r.json())
            .then(res => {
                if (res.status === "success") {
                    const pt = document.getElementById("pred-type-select").value;
                    const sp = document.getElementById("species-select").value;
                    const tsp = document.getElementById("target-species-select").value;
                    const searchId = document.getElementById("search-id-input").value;
                    window.location.href = "/?species=" + encodeURIComponent(sp) + "&target_species=" + encodeURIComponent(tsp) + "&search_id=" + encodeURIComponent(searchId) + "&pred_type=" + encodeURIComponent(pt);
                } else {
                    isProcessing = false;
                    alert(res.msg);
                }
            });
        }
        
        function prepareRoboflow() {
            if(isProcessing) return;
            if(!confirm("This will process all approved and rejected images across all species, snap their polygons to physical edges, and export them to Roboflow_Export. This may take a minute. Continue?")) return;
            isProcessing = true;
            document.body.style.cursor = "wait";
            fetch("/prepare_roboflow", { method: "POST" })
            .then(r => r.json())
            .then(res => {
                isProcessing = false;
                document.body.style.cursor = "default";
                alert(res.msg);
            }).catch(err => {
                isProcessing = false;
                document.body.style.cursor = "default";
                alert("An error occurred during export.");
            });
        }

        function sendAction(action) {
            if(isProcessing) return;
            isProcessing = true;
            document.getElementById("main-image").style.opacity = "0.3";
            const keepLabels = labelsData.filter((_, idx) => activeLabels[idx]).map(l => l.raw);
            const overrideSpecies = document.getElementById("override-species").value || "{{species_name}}";
            fetch("/action", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({
                    action: action, 
                    path: "{{viz_path}}", 
                    keep_labels: keepLabels,
                    override_species: overrideSpecies
                })
            }).then(() => {
                const pt = document.getElementById("pred-type-select").value;
                const sp = document.getElementById("species-select").value;
                const tsp = document.getElementById("target-species-select").value;
                const searchId = document.getElementById("search-id-input").value;
                window.location.href = "/?species=" + encodeURIComponent(sp) + "&target_species=" + encodeURIComponent(tsp) + "&search_id=" + encodeURIComponent(searchId) + "&pred_type=" + encodeURIComponent(pt) + "&idx={{ idx }}";
            });
        }

        document.addEventListener("keydown", function(e) {
            if (e.key === "ArrowRight") { e.preventDefault(); sendAction('approve'); }
            if (e.key === "ArrowDown") { e.preventDefault(); sendAction('reject'); }
            if (e.key === " " || e.key === "Spacebar") { e.preventDefault(); sendAction('skip'); }
            if (e.key === "ArrowLeft") { e.preventDefault(); navigate(-1); }
            if (e.key === "ArrowUp") { e.preventDefault(); navigate(1); }
        });
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    print(f"\\n✅ Local Active Learning Server Started! ✨")
    print(f"👉 OPEN YOUR SYSTEM BROWSER TO: http://127.0.0.1:5001")
    print(f"--- Press CTRL+C to safely exit anytime ---")
    app.run(host="0.0.0.0", port=5001)
