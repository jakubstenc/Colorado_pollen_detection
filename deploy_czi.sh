#!/bin/bash
# deploy_czi.sh — Build and deploy the CZI preprocessing job to Kubernetes (CESNET stenc-ns)
# Usage: ./deploy_czi.sh
set -e

RANDOM_ID=$(openssl rand -hex 4)
IMAGE_NAME="ttl.sh/pollen-czi-${RANDOM_ID}:24h"
NAMESPACE="stenc-ns"

echo "─────────────────────────────────────────────────"
echo "🔬 CZI Preprocessing Deployment"
echo "─────────────────────────────────────────────────"

# 1. Build Docker image with CZI dependencies
echo "🚀 1. Building Docker image: $IMAGE_NAME"
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker not running or permission denied. Try: sudo ./deploy_czi.sh"
    exit 1
fi
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
else
    echo "⬇️  Downloading kubectl..."
    curl -LO "https://dl.k8s.io/release/$(curl -Ls https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    chmod +x kubectl
    KUBECTL="./kubectl"
fi

[ -f "./kubeconfig.yaml" ] && export KUBECONFIG="$(pwd)/kubeconfig.yaml"

# 4. Ensure S3 credentials secret exists
echo "🔑 3. Checking s3-credentials secret..."
if ! $KUBECTL get secret s3-credentials -n $NAMESPACE &>/dev/null; then
    echo "   Creating s3-credentials secret from .env file..."
    source .env 2>/dev/null || true
    $KUBECTL create secret generic s3-credentials \
        --from-literal=access-key="${AWS_ACCESS_KEY_ID}" \
        --from-literal=secret-key="${AWS_SECRET_ACCESS_KEY}" \
        -n $NAMESPACE
fi

# 5. Clean up old job
echo "🧹 4. Cleaning up old preprocessing job..."
$KUBECTL delete job pollen-czi-preprocess -n $NAMESPACE --ignore-not-found

# 6. Deploy
echo "🚀 5. Deploying CZI preprocessing job..."
sed "s|IMAGE_PLACEHOLDER|${IMAGE_NAME}|g" k8s/pollen-czi-preprocess-job.yaml \
    | $KUBECTL apply -f - -n $NAMESPACE

# 7. Stream logs
echo "⏳ 6. Waiting for pod to start..."
count=0
while ! $KUBECTL logs job/pollen-czi-preprocess -n $NAMESPACE > /dev/null 2>&1; do
    echo -n "."
    sleep 5
    count=$((count+1))
    [ $count -ge 120 ] && { echo ""; echo "❌ Timeout. Check: $KUBECTL describe job pollen-czi-preprocess -n $NAMESPACE"; exit 1; }
done
echo ""
echo "👀 7. Streaming logs (Ctrl+C to detach, job continues)..."
$KUBECTL logs -f job/pollen-czi-preprocess -n $NAMESPACE
