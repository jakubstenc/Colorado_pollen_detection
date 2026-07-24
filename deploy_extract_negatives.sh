#!/bin/bash
set -e

RANDOM_ID=$(openssl rand -hex 4)
IMAGE_NAME="ttl.sh/pollen-extract-${RANDOM_ID}:24h"
NAMESPACE="stenc-ns"

echo "─────────────────────────────────────────────────"
echo "🚀 YOLOv8 Extract Negatives Deployment to K8s"
echo "─────────────────────────────────────────────────"

echo "1. Building Docker image: $IMAGE_NAME"
# Assuming Dockerfile.train has all the dependencies we need
docker build -f Dockerfile.train -t "$IMAGE_NAME" .

echo "2. Pushing image..."
docker push "$IMAGE_NAME"

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

echo "3. Creating/Updating the configmap for the python script..."
$KUBECTL create configmap extract-script --from-file=extract_negatives.py -n $NAMESPACE --dry-run=client -o yaml | $KUBECTL apply -f -

echo "4. Cleaning up old extract job..."
$KUBECTL delete job pollen-extract-negatives -n $NAMESPACE --ignore-not-found

echo "5. Deploying new extract job..."
sed "s|IMAGE_PLACEHOLDER|${IMAGE_NAME}|g" k8s/extract-negatives-job.yaml | $KUBECTL apply -f -

echo "─────────────────────────────────────────────────"
echo "✅ Job successfully submitted!"
echo "To monitor logs, run:"
echo "kubectl logs -f job/pollen-extract-negatives -n $NAMESPACE"
