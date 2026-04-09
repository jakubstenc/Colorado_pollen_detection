#!/bin/bash
set -e

RANDOM_ID=$(openssl rand -hex 4)
IMAGE_NAME="ttl.sh/pollen-species-train-${RANDOM_ID}:24h"
NAMESPACE="stenc-ns"

echo "─────────────────────────────────────────────────"
echo "🚀 YOLOv8 SPECIES Training Deployment to K8s"
echo "─────────────────────────────────────────────────"

echo "1. Building Docker image with new classification pipelines: $IMAGE_NAME"
docker build -f Dockerfile.train -t "$IMAGE_NAME" .

echo "2. Pushing image to ttl.sh repository..."
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

echo "3. Cleaning up old species training jobs..."
$KUBECTL delete pod pollen-train-species -n $NAMESPACE --ignore-not-found

echo "4. Deploying new Species Classifier training job..."
sed "s|IMAGE_PLACEHOLDER|${IMAGE_NAME}|g" k8s/pollen-species-train-job.yaml | $KUBECTL apply -f - -n $NAMESPACE

echo "─────────────────────────────────────────────────"
echo "✅ Job successfully submitted!"
echo "Data is natively syncing from S3 into the pod bypassing the hostPath security."
echo "To monitor logs, run:"
echo "kubectl logs -f pod/pollen-train-species -n $NAMESPACE"
