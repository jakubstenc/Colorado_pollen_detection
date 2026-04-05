import os
import shutil
import glob
from flask import Flask, render_template_string, send_file, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.expanduser("~/Desktop/Species_model")
DEST_DIR = os.path.expanduser("~/Desktop/Retrain_Dataset")

DEST_IMG_DIR = os.path.join(DEST_DIR, "images")
DEST_LBL_DIR = os.path.join(DEST_DIR, "labels")

os.makedirs(DEST_IMG_DIR, exist_ok=True)
os.makedirs(DEST_LBL_DIR, exist_ok=True)

def get_pending_images():
    """Find all Visualization tiles that haven't been reviewed yet."""
    all_viz = glob.glob(os.path.join(BASE_DIR, "**", "Vizualization", "*_viz.jpg"), recursive=True)
    return [v for v in all_viz if "/Reviewed/" not in v]

def resolve_source_files(viz_path):
    """Derive the paths of the associated raw Image and raw Label automatically."""
    viz_dir = os.path.dirname(viz_path)
    species_dir = os.path.dirname(viz_dir)
    
    filename = os.path.basename(viz_path)
    base_stem = filename.replace("_viz.jpg", "")
    
    img_path = os.path.join(species_dir, "Images", base_stem + ".jpg")
    lbl_path = os.path.join(species_dir, "Labels", base_stem + ".txt")
    
    return img_path, lbl_path, base_stem, species_dir

def mark_as_reviewed(viz_path):
    """Move processed items out of the live queue so they don't reappear on reload."""
    img_path, lbl_path, base_stem, species_dir = resolve_source_files(viz_path)
    
    rev_dir = os.path.join(species_dir, "Reviewed")
    os.makedirs(os.path.join(rev_dir, "Images"), exist_ok=True)
    os.makedirs(os.path.join(rev_dir, "Labels"), exist_ok=True)
    os.makedirs(os.path.join(rev_dir, "Vizualization"), exist_ok=True)
    
    if os.path.exists(viz_path): shutil.move(viz_path, os.path.join(rev_dir, "Vizualization", os.path.basename(viz_path)))
    if os.path.exists(img_path): shutil.move(img_path, os.path.join(rev_dir, "Images", os.path.basename(img_path)))
    if os.path.exists(lbl_path): shutil.move(lbl_path, os.path.join(rev_dir, "Labels", os.path.basename(lbl_path)))

@app.route("/")
def index():
    pending = get_pending_images()
    if not pending:
        return "<h1 style='color:white; font-family:sans-serif; text-align:center; padding-top:20%'>✅ All items reviewed! The Retrain_Dataset is ready.</h1>"
    
    curr_viz = pending[0]
    return render_template_string(HTML_TEMPLATE, viz_path=curr_viz, remaining=len(pending))

@app.route("/image")
def serve_image():
    path = request.args.get("path")
    return send_file(path)

@app.route("/action", methods=["POST"])
def action():
    data = request.json
    action_type = data.get("action")
    viz_path = data.get("path")
    
    if not os.path.exists(viz_path):
        return jsonify({"status": "error", "msg": "File not found"}), 404
        
    img_path, lbl_path, base_stem, _ = resolve_source_files(viz_path)
    
    if action_type in ["approve", "reject"]:
        if os.path.exists(img_path):
            shutil.copy(img_path, os.path.join(DEST_IMG_DIR, base_stem + ".jpg"))
        
        dest_lbl = os.path.join(DEST_LBL_DIR, base_stem + ".txt")
        if action_type == "approve":
            if os.path.exists(lbl_path):
                shutil.copy(lbl_path, dest_lbl)
        elif action_type == "reject":
            # Strip YOLO labels to treat as pure HARD NEGATIVE Background.
            open(dest_lbl, 'w').close()
            
    mark_as_reviewed(viz_path)
    return jsonify({"status": "success"})


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Active Learning UI</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #121212; color: #eeeeee; text-align: center; margin:0; padding:0;}
        .container { max-width: 1000px; margin: auto; padding-top: 2vh;}
        h2 { margin-bottom: 10px; color:#bbbbbb; font-weight: 300;}
        .image-box { margin: 15px auto; padding: 10px; background: #1e1e1e; border-radius: 10px; display:inline-block; border: 1px solid #333; box-shadow: 0 4px 15px rgba(0,0,0,0.5);}
        img { max-width: 100%; max-height: 70vh; border-radius: 5px; display:block;}
        .controls { margin-top: 15px; display: flex; justify-content: center; gap: 20px; }
        button { padding: 16px 32px; font-size: 17px; border: none; border-radius: 6px; cursor: pointer; color: white; font-weight: 600; white-space:nowrap; transition: all 0.2s;}
        .btn-approve { background: #22a042; box-shadow: 0 4px #126325;}
        .btn-reject { background: #d63346; box-shadow: 0 4px #8b1c28;}
        .btn-skip { background: #5a6268; box-shadow: 0 4px #363b3e;}
        button:active { transform: translateY(4px); box-shadow: 0 0 transparent;}
        button:hover { filter: brightness(1.1); }
        .stats { margin-top: 25px; color: #888; letter-spacing: 0.5px;}
        .shortcut { color: #ccc; background:#333; padding: 3px 8px; border-radius: 4px; font-size: 13px; font-family: monospace}
    </style>
</head>
<body>
    <div class="container">
        <h2>Rapid Image Verification <span style="color:#d63346; font-weight:bold;">({{remaining}} remaining)</span></h2>
        
        <div class="image-box">
            <img src="/image?path={{viz_path}}" alt="Visualization Visualization">
        </div>
        
        <div class="controls">
            <button class="btn-approve" onclick="sendAction('approve')">Positive Pollen <br><span class="shortcut">Right Arrow ➔</span></button>
            <button class="btn-reject" onclick="sendAction('reject')">Hard Negative (Dirt) <br><span class="shortcut">Down Arrow ⬇</span></button>
            <button class="btn-skip" onclick="sendAction('skip')">Skip Box <br><span class="shortcut">Spacebar</span></button>
        </div>
        
        <p class="stats"><b>Active Learning Routing:</b><br/> Approve will preserve labels. Reject will blank the labels to strictly teach background traits.</p>
    </div>

    <script>
        let isProcessing = false;
        function sendAction(action) {
            if(isProcessing) return;
            isProcessing = true;
            document.body.style.opacity = "0.7";
            fetch("/action", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({action: action, path: "{{viz_path}}"})
            }).then(() => window.location.reload());
        }

        document.addEventListener("keydown", function(e) {
            if (e.key === "ArrowRight") { e.preventDefault(); sendAction('approve'); }
            if (e.key === "ArrowDown") { e.preventDefault(); sendAction('reject'); }
            if (e.key === " " || e.key === "Spacebar") { e.preventDefault(); sendAction('skip'); }
        });
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    print(f"\\n✅ Local Active Learning Server Started! ✨")
    print(f"👉 OPEN YOUR SYSTEM BROWSER TO: http://127.0.0.1:5000")
    print(f"--- Press CTRL+C to safely exit anytime ---")
    app.run(host="0.0.0.0", port=5000)
