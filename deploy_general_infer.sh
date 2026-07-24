#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: ./deploy_general_infer.sh <species_folder_in_source>"
    echo "Example: ./deploy_general_infer.sh Pollen_deposition/Cal_chi"
    exit 1
fi

TARGET_SPECIES=$1
DOCKER_IMAGE="ttl.sh/pollen-czi-$(uuidgen | cut -d'-' -f1):24h"
NAMESPACE="stenc-ns"

echo "─────────────────────────────────────────────────"
echo "🔬 Generic Inference Deployment: $TARGET_SPECIES"
echo "─────────────────────────────────────────────────"

echo "🚀 1. Building Docker image: $DOCKER_IMAGE"
docker build -f Dockerfile.czi -t "$DOCKER_IMAGE" .

echo "☁️  2. Pushing image..."
docker push "$DOCKER_IMAGE"

echo "🧹 3. Cleaning up old generic inference job..."
kubectl delete job pollen-general-infer -n $NAMESPACE --ignore-not-found=true

echo "🚀 4. Deploying K8s workflow..."
sed -e "s|IMAGE_PLACEHOLDER|$DOCKER_IMAGE|g" \
    -e "s|SPECIES_PLACEHOLDER|$TARGET_SPECIES|g" \
    k8s/pollen-general-infer-job.yaml | kubectl apply -f - -n $NAMESPACE

echo "─────────────────────────────────────────────────"
echo "✅ Job successfully submitted!"
echo "To monitor logs, run:"
echo "kubectl logs -f job/pollen-general-infer -n $NAMESPACE"
