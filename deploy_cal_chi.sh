#!/bin/bash
# deploy_cal_chi.sh — Build and deploy the Cal_chi inference job to Kubernetes
# Usage: ./deploy_cal_chi.sh
set -e

RANDOM_ID=$(openssl rand -hex 4)
IMAGE_NAME="ttl.sh/pollen-czi-${RANDOM_ID}:24h"
NAMESPACE="stenc-ns"

echo "─────────────────────────────────────────────────"
echo "🔬 Cal_chi Processing Deployment"
echo "─────────────────────────────────────────────────"

# 1. Build Docker image with updated inference code
echo "🚀 1. Building Docker image: $IMAGE_NAME"
docker build -f Dockerfile.czi -t "$IMAGE_NAME" .

# 2. Push to ephemeral registry
echo "☁️  2. Pushing image..."
n=0
until [ "$n" -ge 5 ]; do
    docker push "$IMAGE_NAME" && break
    n=$((n+1))
    echo "   ⚠️ Push failed. Retry $n/5 in 5s..."
    sleep 5
done
[ "$n" -ge 5 ] && { echo "❌ Push failed after 5 attempts."; exit 1; }

# 3. Locate kubectl
if command -v kubectl &>/dev/null; then
    KUBECTL="kubectl"
elif [ -f "./kubectl" ]; then
    KUBECTL="./kubectl"
fi

[ -f "./kubeconfig.yaml" ] && export KUBECONFIG="$(pwd)/kubeconfig.yaml"

# 4. Clean up old job
echo "🧹 4. Cleaning up previous Cal_chi infer job..."
$KUBECTL delete job pollen-cal-chi-infer -n $NAMESPACE --ignore-not-found

# 5. Deploy
echo "🚀 5. Deploying Cal_chi inference job..."
sed "s|IMAGE_PLACEHOLDER|${IMAGE_NAME}|g" k8s/pollen-cal-chi-infer-job.yaml \
    | $KUBECTL apply -f - -n $NAMESPACE

# 6. Stream logs
echo "⏳ 6. Waiting for pod to start (it will sync 76 CZI files first! This might take 10+ minutes)..."
echo "   It's safe to Ctrl+C here if you don't want to wait on the init container."
count=0
while ! $KUBECTL logs job/pollen-cal-chi-infer -n $NAMESPACE > /dev/null 2>&1; do
    echo -n "."
    sleep 10
    count=$((count+1))
    [ $count -ge 120 ] && { echo ""; echo "❌ Timeout waiting for logs. Check: $KUBECTL describe job pollen-cal-chi-infer -n $NAMESPACE"; exit 1; }
done
echo ""
echo "👀 7. Streaming Python logs (Ctrl+C to detach)..."
$KUBECTL logs -f job/pollen-cal-chi-infer -n $NAMESPACE
