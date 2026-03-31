#!/bin/bash
set -e

# Set kubeconfig if it exists in the current directory
[ -f "./kubeconfig.yaml" ] && export KUBECONFIG="$(pwd)/kubeconfig.yaml"

# Find kubectl
if command -v kubectl &>/dev/null; then
    KUBECTL="kubectl"
elif [ -f "./kubectl" ]; then
    KUBECTL="./kubectl"
else
    echo "❌ kubectl not found! Please download it or run deploy_czi.sh first."
    exit 1
fi

NAMESPACE="stenc-ns"

echo "─────────────────────────────────────────────────"
echo "🚀 1. Updating ConfigMap with modified python script..."
$KUBECTL create configmap pollen-extract-script \
    --from-file=src/extract_300.py \
    -n $NAMESPACE -o yaml --dry-run=client | $KUBECTL apply -f -

echo "─────────────────────────────────────────────────"
echo "🚀 2. Starting Unseen Ran_adu Extraction Job..."
$KUBECTL apply -f k8s/pollen-extract-ran-adu-job.yaml

echo "─────────────────────────────────────────────────"
echo "✅ Extraction job submitted!"
echo "To watch the progress, run:"
echo "  $KUBECTL get po -n $NAMESPACE -w"
echo ""
echo "Once that pod completes successfully, start the pseudo-labeling and upload script:"
echo "  $KUBECTL apply -f k8s/pollen-pseudo-label-ran-adu-job.yaml"
