import glob, os, shutil, cv2
STAGED_AREA_DIR = os.path.expanduser("~/cesnet_cloud/bucket/PEG/Colorado/Staged_area/Species_curated")
img_files = glob.glob(os.path.join(STAGED_AREA_DIR, "*", "Discarded", "Images", "*.jpg"))
out_dir = os.path.expanduser("~/cesnet_data/PEG/Colorado/Roboflow_Export")
out_img = os.path.join(out_dir, "images")
out_lbl = os.path.join(out_dir, "labels")

exported = 0
for img_f in img_files:
    base = os.path.basename(img_f)
    lbl_f = img_f.replace("/Images/", "/Labels/").replace(".jpg", ".txt")
    out_img_f = os.path.join(out_img, base)
    out_lbl_f = os.path.join(out_lbl, base.replace(".jpg", ".txt"))
    if os.path.exists(lbl_f):
        with open(lbl_f, 'r') as f:
            lines = f.readlines()
    else:
        lines = []
    if not lines:
        exported += 1
        continue
    exported += 1

print("Total files:", len(img_files))
print("Exported:", exported)
