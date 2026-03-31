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
   - **Stage 1 (General Pollen Model):** Finds *any* pollen grain anywhere, unifying all annotations mathematically into class `0`.
   - **Stage 2 (Species Classifier):** Uses the extracted dataset from Stage 1 to strictly build and refine a multi-class species discriminator.
2. **High-Res CZI Tiling & Overview:** Scans are automatically segmented into overlapping 640px tiles, processed through YOLO, and finally reconstructed into a downscaled 2000px single-panel visual mosaic with detection polygons drawn directly on top.
3. **Mosaic Disabling:** We disable mosaic augmentation for the final 20 epochs to stabilize training loss and eliminate ghost detections.

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
