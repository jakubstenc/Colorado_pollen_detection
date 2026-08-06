#!/bin/bash
set -e

NAMESPACE="stenc-ns"

echo "─────────────────────────────────────────────────"
echo "🚀 YOLOv8 SPECIES Training Deployment to K8s"
echo "─────────────────────────────────────────────────"

echo "Bypassing ttl.sh (outage). Mounting scripts via ConfigMap."

# Locate kubectl
if command -v kubectl &>/dev/null; then
    KUBECTL="kubectl"
elif [ -f "./kubectl" ]; then
    KUBECTL="./kubectl"
else
    echo "❌ kubectl not found!"
    exit 1
fi

[ -f "./kubeconfig.yaml" ] && export KUBECONFIG="$(pwd)/kubeconfig.yaml"

echo "1. Cleaning up old species training jobs and configmaps..."
$KUBECTL delete pod pollen-train-species -n $NAMESPACE --ignore-not-found
$KUBECTL delete configmap species-train-scripts -n $NAMESPACE --ignore-not-found

echo "2. Creating ConfigMap for training scripts..."
$KUBECTL create configmap species-train-scripts -n $NAMESPACE --from-file=src/extract_crops.py --from-file=src/train_species.py

echo "3. Deploying new Species Classifier training job..."
$KUBECTL apply -f k8s/pollen-species-train-job.yaml -n $NAMESPACE

echo "─────────────────────────────────────────────────"
echo "✅ Job successfully submitted!"
echo "Data is natively syncing from S3 into the pod bypassing the hostPath security."
echo "To monitor logs, run:"
echo "kubectl logs -f pod/pollen-train-species -n $NAMESPACE"
