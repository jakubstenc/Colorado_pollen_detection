#!/bin/bash
set -e

RANDOM_ID=$(openssl rand -hex 4)
IMAGE_NAME="ttl.sh/pollen-train-${RANDOM_ID}:24h"
NAMESPACE="stenc-ns"

echo "─────────────────────────────────────────────────"
echo "🚀 YOLOv8 Training Deployment to K8s"
echo "─────────────────────────────────────────────────"

echo "1. Building Docker image: $IMAGE_NAME"
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

echo "3. Cleaning up old training job..."
$KUBECTL delete pod pollen-train-general -n $NAMESPACE --ignore-not-found

echo "4. Deploying new training job..."
sed "s|IMAGE_PLACEHOLDER|${IMAGE_NAME}|g" k8s/pollen-general-train-job.yaml | $KUBECTL apply -f - -n $NAMESPACE

echo "─────────────────────────────────────────────────"
echo "✅ Job successfully submitted!"
echo "To monitor logs, run:"
echo "kubectl logs -f pod/pollen-train-general -n $NAMESPACE"
