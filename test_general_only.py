import sys
import os
sys.path.append('src')
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Inline get_mip_rgb to avoid aicsimageio errors
def get_mip_rgb(local_czi):
    from aicsimageio import AICSImage
    img = AICSImage(local_czi)
    dask_czyx = img.get_image_dask_data("CZYX")
    
    def normalize_to_uint8(img):
        img = img.astype(np.float32)
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = 255.0 * (img - img_min) / (img_max - img_min)
        return img.astype(np.uint8)
        
    mips = []
    num_c = dask_czyx.shape[0]
    for c in range(min(3, num_c)):
        mip = dask_czyx[c].max(axis=0).compute()
        if len(mip.shape) == 3 and mip.shape[-1] == 3:
            return normalize_to_uint8(mip)
        mips.append(normalize_to_uint8(mip))
        
    if len(mips) == 1:
        return np.stack([mips[0], mips[0], mips[0]], axis=-1)
    elif len(mips) == 2:
        return np.stack([mips[0], mips[1], np.zeros_like(mips[0])], axis=-1)
    else:
        return np.stack(mips[:3], axis=-1)

def tile_image(image, size=640, overlap=0.15):
    h, w = image.shape[:2]
    step = int(size * (1 - overlap))
    for y in range(0, h, step):
        for x in range(0, w, step):
            y_end = min(y + size, h)
            x_end = min(x + size, w)
            tile = np.zeros((size, size, 3), dtype=image.dtype)
            patch = image[y:y_end, x:x_end]
            tile[:patch.shape[0], :patch.shape[1]] = patch
            yield tile, x, y

def extract_general_pollen(tile, model, conf_thresh):
    results = model(tile, verbose=False, retina_masks=True)
    detections = []
    if results[0].masks is None or results[0].boxes is None:
        return detections
    H, W = tile.shape[:2]
    for mask_xy, box in zip(results[0].masks.xy, results[0].boxes):
        if mask_xy.shape[0] < 3:
            continue
        c_conf = float(box.conf[0])
        if c_conf < conf_thresh:
            continue
        poly_px = []
        for point in mask_xy:
            x, y = point
            poly_px.append((x, y))
        detections.append({
            'poly_px': np.array(poly_px, dtype=np.int32),
            'conf': c_conf
        })
    return detections

def main():
    general_model = YOLO("models/general_pollen/latest.pt")
    local_czi = Path("test.czi")
    
    rgb = get_mip_rgb(local_czi)
    
    overview_scale = 0.1
    overview_img = cv2.resize(rgb, (0, 0), fx=overview_scale, fy=overview_scale)
    
    n_tiles = 0
    total_detections = 0
    for tile, tx, ty in tile_image(rgb, size=640, overlap=0.15):
        n_tiles += 1
        # Pass directly to general model
        detections = extract_general_pollen(tile, general_model, conf_thresh=0.1) # LOWER CONFIDENCE
        
        for d in detections:
            total_detections += 1
            poly_px = d['poly_px']
            
            global_poly = poly_px.copy()
            global_poly[:, 0] += tx
            global_poly[:, 1] += ty
            
            overview_poly = (global_poly * overview_scale).astype(int)
            cv2.polylines(overview_img, [overview_poly.reshape((-1, 1, 2))], True, (0, 0, 255), 2) # RED bounding box
    
    print(f"Total tiles processed: {n_tiles}")
    print(f"Total general pollen detected: {total_detections}")
    
    # Save the output image
    overview_bgr = cv2.cvtColor(overview_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("overview_general_only.jpg", overview_bgr)

if __name__ == "__main__":
    main()
