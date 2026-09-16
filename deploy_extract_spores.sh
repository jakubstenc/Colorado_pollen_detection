#!/bin/bash
set -e

RANDOM_ID=$(openssl rand -hex 4)
IMAGE_NAME="ttl.sh/pollen-extract-spores-${RANDOM_ID}:24h"
NAMESPACE="stenc-ns"

echo "─────────────────────────────────────────────────"
echo "🍄 Lycopodium Spore Extraction Deployment to K8s"
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

echo "3. Creating/Updating configmap with both extract_spores.py and build_species_dataset.py..."
$KUBECTL create configmap extract-spores-script \
    --from-file=extract_spores.py \
    --from-file=build_species_dataset.py=src/build_species_dataset.py \
    -n $NAMESPACE --dry-run=client -o yaml | $KUBECTL apply -f -

echo "4. Cleaning up old extract-spores job if present..."
$KUBECTL delete job pollen-extract-spores -n $NAMESPACE --ignore-not-found

echo "5. Deploying new extract-spores job..."
sed "s|IMAGE_PLACEHOLDER|${IMAGE_NAME}|g" k8s/extract-spores-job.yaml | $KUBECTL apply -f -

echo "─────────────────────────────────────────────────"
echo "✅ Job submitted!"
echo ""
echo "Monitor with:"
echo "  ./kubectl logs -f job/pollen-extract-spores -n $NAMESPACE --kubeconfig kubeconfig.yaml"
echo ""
echo "After it completes, open the Active Learning UI."
echo "The 'Lyc_spo' folder will appear with orange-outlined spore detections to review."
