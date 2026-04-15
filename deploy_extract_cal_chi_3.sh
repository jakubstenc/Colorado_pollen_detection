#!/bin/bash
# deploy_extract_cal_chi_3.sh — Build and deploy the targeted extraction job to Kubernetes
set -e

RANDOM_ID=$(openssl rand -hex 4)
IMAGE_NAME="ttl.sh/pollen-species-builder-${RANDOM_ID}:24h"
NAMESPACE="stenc-ns"

echo "─────────────────────────────────────────────────"
echo "🔬 Targeted Species Extraction Deployment"
echo "─────────────────────────────────────────────────"

echo "🚀 1. Building Docker image: $IMAGE_NAME"
# Use Dockerfile.species which contains the environment for dataset building
docker build -f Dockerfile.species -t "$IMAGE_NAME" .

echo "☁️  2. Pushing image..."
n=0
until [ "$n" -ge 5 ]; do
    docker push "$IMAGE_NAME" && break
    n=$((n+1))
    echo "   ⚠️ Push failed. Retry $n/5 in 5s..."
    sleep 5
done
[ "$n" -ge 5 ] && { echo "❌ Push failed after 5 attempts."; exit 1; }

if command -v kubectl &>/dev/null; then
    KUBECTL="kubectl"
elif [ -f "./kubectl" ]; then
    KUBECTL="./kubectl"
fi

[ -f "./kubeconfig.yaml" ] && export KUBECONFIG="$(pwd)/kubeconfig.yaml"

echo "🧹 4. Cleaning up previous targeted extraction job..."
$KUBECTL delete job pollen-extract-cal-chi-3 -n $NAMESPACE --ignore-not-found

echo "🚀 5. Deploying targeted extraction job..."
sed "s|IMAGE_PLACEHOLDER|${IMAGE_NAME}|g" k8s/pollen-extract-cal-chi-3-job.yaml \
    | $KUBECTL apply -f - -n $NAMESPACE

echo "⏳ Wait for pod to start..."
sleep 5

echo "👀 7. Streaming Python logs (Ctrl+C to detach)..."
$KUBECTL logs -f job/pollen-extract-cal-chi-3 -n $NAMESPACE
