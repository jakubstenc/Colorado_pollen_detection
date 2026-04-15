#!/bin/bash
# wait_and_rebuild.sh
# Autonomously polls the Kubernetes cluster for the completion of the training job.
# Upon success, immediately dynamically regenerates the training ground-truth data pipeline natively.

TARGET_POD="pollen-train-general"
NAMESPACE="stenc-ns"

echo "[$(date)] Waiting for Pod $TARGET_POD to initialize..."
sleep 30

while true; do
  STATUS=$(kubectl --kubeconfig=/home/meow/Documents/Antigravity/Colorado_pollen_detection/kubeconfig.yaml get pod $TARGET_POD -n $NAMESPACE -o jsonpath='{.status.phase}' 2>/dev/null)
  
  if [ "$STATUS" == "Succeeded" ]; then
    echo "[$(date)] ✅ Training completed successfully across Kubernetes cluster!"
    break
  elif [ "$STATUS" == "Failed" ]; then
    echo "[$(date)] ❌ Training Pod encountered a distinct failure!"
    exit 1
  fi
  
  # Poll interval
  sleep 45
done

echo "[$(date)] 🚀 Triggering Native Dataset Extraction (local_dataset_builder) against newest S3 Weights..."
cd /home/meow/Documents/Antigravity/Colorado_pollen_detection

# Run the updated, edge-snapping, comprehensive local dataset builder natively
.venv/bin/python -u local_dataset_builder.py > /tmp/master_dataset_build_output.txt 2>&1

echo "[$(date)] ✅ Complete Dataset Generator executed."
