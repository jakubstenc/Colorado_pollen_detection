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

os.makedirs(DEST_IMG_DIR, exist_ok=True)
os.makedirs(DEST_LBL_DIR, exist_ok=True)

pending_cache = None

def get_pending_images():
    global pending_cache
    if pending_cache is None:
        print("⏳ Scraping metadata directory structure from S3... (this might take a few seconds initially)")
        all_viz = glob.glob(os.path.join(BASE_DIR, "**", "Vizualization", "*_viz.jpg"), recursive=True)
        pending_cache = sorted([v for v in all_viz if "/Reviewed/" not in v and "/Discarded/" not in v])
    return pending_cache

def resolve_source_files(viz_path):
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
    else:
        target_folder_name = "Reviewed"
        
    rev_dir = os.path.join(species_dir, target_folder_name)
    os.makedirs(os.path.join(rev_dir, "Images"), exist_ok=True)
    os.makedirs(os.path.join(rev_dir, "Labels"), exist_ok=True)
    os.makedirs(os.path.join(rev_dir, "Vizualization"), exist_ok=True)
    
    if os.path.exists(viz_path): shutil.move(viz_path, os.path.join(rev_dir, "Vizualization", os.path.basename(viz_path)))
    if os.path.exists(img_path): shutil.move(img_path, os.path.join(rev_dir, "Images", os.path.basename(img_path)))
    if os.path.exists(lbl_path): shutil.move(lbl_path, os.path.join(rev_dir, "Labels", os.path.basename(lbl_path)))

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
    filtered_pending = [p for p in pending if f"/{req_species}/" in p] if req_species else pending
    
    req_target_species = request.args.get("target_species", "")
    if req_target_species:
        filtered_pending = [p for p in filtered_pending if req_target_species.lower() in os.path.basename(p).lower()]
    
    if not filtered_pending:
        filtered_pending = pending
        req_species = ""
        
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
                                  target_override=target_override,
                                  global_species=GLOBAL_SPECIES,
                                  idx=idx,
                                  undo_available=len(undo_stack) > 0)

@app.route("/image")
def serve_image():
    path = request.args.get("path")
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
    
    if last["action"] == "skip":
        rev_dir = os.path.join(species_dir, "Discarded")
    else:
        rev_dir = os.path.join(species_dir, "Reviewed")
    
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


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Active Learning UI</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #121212; color: #eeeeee; text-align: center; margin:0; padding:0; display:flex;}
        .sidebar { width: 320px; background: #1e1e1e; height: 100vh; box-sizing: border-box; padding: 25px 20px; border-right: 1px solid #333; text-align: left; display:flex; flex-direction:column; overflow-y:auto; box-shadow: 2px 0 10px rgba(0,0,0,0.5); z-index:10;}
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
            <div class="panel-label">Current Species</div>
            <div class="panel-value" style="color: #4CAF50;">{{ species_name }}</div>
        </div>

        <div class="panel-section">
            <div class="panel-label">Objects Detected</div>
            <div class="panel-value" style="color: #FFC107; font-size: 24px;"><span id="active-count">{{ num_objs }}</span> / {{ num_objs }} targets</div>
            <div style="font-size:11px; color:#777; margin-top:4px;">(Click bounding boxes on the image to toggle/correct labels)</div>
        </div>

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
        <div class="image-box" style="display:inline-block;">
            <div style="position:relative; display:inline-block;">
                <img src="/image?path={{viz_path}}" alt="Visualization" id="main-image">
                <div id="labels-container" style="position:absolute; top:0; left:0; width:100%; height:100%;"></div>
            </div>
        </div>
        
        <div class="nav-container">
            <button class="btn-nav" onclick="navigate(-1)">⬅ Previous Frame (Left Arrow)</button>
            <button class="btn-nav" onclick="navigate(1)">Next Frame ➡ (Up Arrow)</button>
        </div>
        
        <div class="controls" style="margin-top:25px;">
            <button class="btn-approve" onclick="sendAction('approve')">Valid Pollen <span class="shortcut">Right Arrow</span></button>
            <button class="btn-reject" onclick="sendAction('reject')">Reject Background <span class="shortcut">Down Arrow</span></button>
            <button class="btn-skip" onclick="sendAction('skip')">Delete from UI Queue<span class="shortcut">Spacebar</span></button>
        </div>
    </div>

    <script>
        let isProcessing = false;
        const labelsData = {{ labels_data|tojson|safe }};
        let activeLabels = new Array(labelsData.length).fill(true);

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
                
                const left = lbl.x_min * 100;
                const top = lbl.y_min * 100;
                const width = lbl.w * 100;
                const height = lbl.h * 100;
                
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
            const sp = document.getElementById("species-select").value;
            const tsp = document.getElementById("target-species-select").value;
            window.location.href = "/?species=" + encodeURIComponent(sp) + "&target_species=" + encodeURIComponent(tsp);
        }
        
        function navigate(dir) {
            let nextIdx = {{ idx }} + dir;
            if(nextIdx < 0) nextIdx = 0;
            const sp = document.getElementById("species-select").value;
            const tsp = document.getElementById("target-species-select").value;
            window.location.href = "/?species=" + encodeURIComponent(sp) + "&target_species=" + encodeURIComponent(tsp) + "&idx=" + nextIdx;
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
                const sp = document.getElementById("species-select").value;
                const tsp = document.getElementById("target-species-select").value;
                window.location.href = "/?species=" + encodeURIComponent(sp) + "&target_species=" + encodeURIComponent(tsp) + "&idx={{ idx }}";
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
