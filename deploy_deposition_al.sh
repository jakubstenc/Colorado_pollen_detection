#!/bin/bash
set -e

RANDOM_ID=$(openssl rand -hex 4)
IMAGE_NAME="ttl.sh/pollen-extract-${RANDOM_ID}:24h"
NAMESPACE="stenc-ns"

echo "─────────────────────────────────────────────────"
echo "🚀 YOLOv8 Deposition Active Learning K8s Deployment"
echo "─────────────────────────────────────────────────"

echo "1. Skipping Docker build due to ttl.sh outage (installing dependencies at runtime)"

echo "2. Skipping Docker push due to ttl.sh outage"


if command -v kubectl &>/dev/null; then
    KUBECTL="kubectl"
elif [ -f "./kubectl" ]; then
    KUBECTL="./kubectl"
else
    echo "❌ kubectl not found!"
    exit 1
fi

[ -f "./kubeconfig.yaml" ] && export KUBECONFIG="$(pwd)/kubeconfig.yaml"

echo "3. Creating/Updating the configmap for the python script..."
$KUBECTL create configmap deposition-al-script --from-file=k8s_deposition_al.py --from-file=src/focus_check.py --from-file=src/build_species_dataset.py -n $NAMESPACE --dry-run=client -o yaml | $KUBECTL apply -f -

echo "4. Cleaning up old deposition-al job..."
$KUBECTL delete job pollen-deposition-al -n $NAMESPACE --ignore-not-found

echo "5. Deploying new deposition-al job..."
sed "s|IMAGE_PLACEHOLDER|${IMAGE_NAME}|g" k8s/deposition-al-job.yaml | $KUBECTL apply -f -

echo "─────────────────────────────────────────────────"
echo "✅ Job successfully submitted!"
echo "To monitor logs, run:"
echo "kubectl logs -f job/pollen-deposition-al -n $NAMESPACE"
