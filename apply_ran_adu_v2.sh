#!/bin/bash
set -e

[ -f "./kubeconfig.yaml" ] && export KUBECONFIG="$(pwd)/kubeconfig.yaml"

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
echo "🚀 1. Cleaning up previous jobs & updating ConfigMaps..."
$KUBECTL delete job pollen-extract-ran-adu-v2 -n $NAMESPACE --ignore-not-found
$KUBECTL delete job pollen-pseudo-label-ran-adu-v2 -n $NAMESPACE --ignore-not-found

$KUBECTL create configmap pollen-extract-script \
    --from-file=src/extract_300.py \
    -n $NAMESPACE -o yaml --dry-run=client | $KUBECTL apply -f -

$KUBECTL create configmap pseudo-label-script \
    --from-file=src/pseudo_label.py \
    -n $NAMESPACE -o yaml --dry-run=client | $KUBECTL apply -f -

echo "─────────────────────────────────────────────────"
echo "🚀 2. Starting V2 Unseen Ran_adu Extraction Job (1500 candidate tiles)..."
$KUBECTL apply -f k8s/pollen-extract-ran-adu-v2-job.yaml

echo "─────────────────────────────────────────────────"
echo "✅ Extraction job submitted!"
echo "To watch the progress, run:"
echo "  $KUBECTL get po -n $NAMESPACE -w"
echo ""
echo "Once that pod completes successfully, start the positive-filtering pseudo-labeler:"
echo "  $KUBECTL apply -f k8s/pollen-pseudo-label-ran-adu-v2-job.yaml"
