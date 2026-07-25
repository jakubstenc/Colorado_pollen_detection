# 🌲 Colorado Pollen Detection (YOLOv8)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green) ![Status](https://img.shields.io/badge/Status-Active-success)

An automated two-stage computer vision pipeline for processing large `.czi` microscopy scans and detecting pollen using deep learning. This project leverages YOLOv8 to locate pollen grains from complex microscopic environments, unifying various species under a single detection class before parsing them via a second-tier species classifier.

---

## 📌 Table of Contents

- [🔭 Project Overview](#-project-overview)
- [🔬 Methodology](#-methodology)
- [⚙️ Workflow Setup](#️-workflow-setup)
- [🏷️ Labeling Guide (Roboflow)](#️-labeling-guide-roboflow)
- [🏋️ Training Pipeline](#️-training-pipeline)
- [🔍 Routine Detection](#-routine-detection)
- [📊 Model Performance](#-model-performance)

---

## 🔭 Project Overview

This tool automates the extraction and detection of pollen grains from massive microscopy files. It is designed to handle multi-gigabyte `.czi` files by breaking them into manageable, high-resolution overlapping tiles for inference and training.

- **Input:** High-resolution microscopy scans (`.czi`).
- **Output:** Annotated full-image visual projections, CSV tables with pollen metrics (count, $\mu m^2$ size), and YOLOv8 dataset blocks.
- **Performance:** Optimized for extreme image sizes using active tiling, non-maximum suppression, and automated dataset packaging.

---

## 🔬 Methodology

The project operates locally against S3 datasets or on the **CESNET MetaCentrum** high-performance computing cluster using Kubernetes Jobs. 

### Key Strategies:
1. **Two-Stage Architecture:** 
   - **Stage 1 (General Pollen Segmentation):** Finds *any* pollen grain anywhere using the massive 40M parameter `yolov8l-seg.pt` Instance Segmentation model. All annotations are mapped to class `0` to build a highly generalized contour detector.
   - **Stage 2 (Species Classifier):** Uses the extracted dataset from Stage 1 to strictly build and refine a multi-class species discriminator.
2. **Dynamic Dataset Compilation:** Zero-maintenance data ingestion. The script passively consumes un-annotated images from `Staged_negatives` (automatically generating empty label files) and parses an exact 80/10/10 Train/Val/Test random split to guarantee unbiased post-training evaluation.
3. **High-Res CZI Tiling & Overview:** Scans are automatically segmented into overlapping 640px tiles, processed through YOLO, and finally reconstructed into a downscaled 2000px single-panel visual mosaic with detection polygons drawn directly on top.
4. **Optimized Training:** The A100 pipeline forces a massive 500-epoch hyper-optimized run with `batch=32` and `patience=0`. We disable mosaic augmentation for the final 20 epochs to stabilize training loss and permanently eliminate ghost detections.

---

## ⚙️ Workflow Setup

Deployment is managed natively via Python scripts connecting directly to your mapped `cesnet_cloud` bucket, or via matched Kubernetes Jobs in the `stenc-ns` namespace that inherit HostPath volumes natively.

### Storage Architecture
- **Staging Areas (Roboflow Datasets):** `/home/meow/cesnet_cloud/bucket/PEG/Colorado/Staged_area/`
- **True Negatives:** `/home/meow/cesnet_cloud/bucket/PEG/Colorado/Staged_negatives/`
- **Routine Detection (Input):** `/home/meow/cesnet_cloud/bucket/PEG/Colorado/detect_images/`
- **Detection Results (Output):** `/home/meow/cesnet_cloud/bucket/PEG/Colorado/detection_results/`

---

## 🏷️ Labeling Guide (Roboflow)

We use **Roboflow** for semantic annotation and dataset versioning. 

1. **Polygon Masks:** Draw tight, precise polygons around the perimeter of the pollen grain. Do not use generic square boxes.
2. **Ghosting/Debris:** Ensure that background artifacts and air bubbles are ignored to reduce false positives.
3. **Automated Exports:** The detection pipeline (`czi_ingest.py`) physically writes strictly formatted multi-class `.txt` files on detection, outputting native YOLO datasets immediately usable for further training.

---

## 🧑‍💻 Continuous Active Learning Loop

Machine learning models improve continuously through feedback. We use a local web-based UI (`active_learning_ui.py`) to grade the model's predictions on brand-new `.czi` scans and feed that data back into the next training cycle.

**The Loop:**
1. **Run Inference:** Scan a new `.czi` stigma slide through the two-stage pipeline to generate predictions.
2. **Grade Predictions in UI:** Open `active_learning_ui.py` to review the model's performance:
    - **Accept (Right Arrow):** The polygon prediction is perfect. The image is saved to `Curated_Retrain_Data`.
    - **Reject (Down Arrow):** The model hallucinated (e.g., an air bubble). This is saved as a **Hard Negative** (an empty label file) to `Curated_Retrain_Data` so the model explicitly learns to ignore it.
    - **Skip/Discard (Spacebar):** The model missed the pollen or the polygon is messy. The image is moved to the `Discarded` folder for manual correction.
3. **Roboflow Correction:** Click the "Prepare for Roboflow" button in the UI. This safely exports your `Discarded` images into `Roboflow_Export`. Upload this folder to your Roboflow project, manually redraw the bad polygons, and download the dataset zip back into `Staged_area`.
4. **Retrain on CESNET:** Trigger the training jobs on the cluster. The `train_general.py` script automatically merges your new Roboflow zip, your perfectly `Accepted` images, and your `Hard Negatives` to create a significantly smarter Stage 1 detector!

---

## 🏋️ Training Pipeline

Training executes directly against the dynamically generated dataset folders.

**Workflow:**
1. **Model Directories:** The system maintains local weights logic at `models/general_pollen/latest.pt` and `models/species_classifier/latest.pt`.
2. **Execution:** 
   - Use `kubectl apply -f k8s/pollen-general-train-job.yaml` to deploy natively.
   - Alternatively: run natively using `python src/train_general.py` and `python src/train_species.py`.
3. **Notification:** A Python script intercepts the end sequence and emails the summary metrics immediately to the researchers.

---

## 🔍 Routine Detection

Routine `.czi` inferences are handled by `czi_ingest.py` orchestrating native models.

**Pipeline Breakdown:**
1. Place raw scans into your bucket's `detect_images` folder.
2. Execute the inference:
   ```bash
   kubectl apply -f k8s/pollen-single-detect-job.yaml
   ```
3. Locate Outputs in `detection_results`:
   - `overview_filename.jpg` (Macro stitch visualizing all polygons).
   - `summary_results.csv` (Tabular database of all found pollen limits).

---

## 📊 Model Performance

*(Training outputs pending)*

### Reference Documentation
For a deeper dive into methodology, consult the Quarto docs:
- 👉 [**Main Presentation**](docs/index.html)
- 👉 [**Ingestion Tutorial**](docs/tutorials/ingestion.html)

---
**License:** MIT
