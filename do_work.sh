#!/bin/bash
set -e
KUBECONFIG="$(pwd)/kubeconfig.yaml"
KUBECTL="./kubectl"

echo "1/4: Resetting S3 datastores via Wiper Job..."
$KUBECTL --kubeconfig=$KUBECONFIG delete job wipe-s3-fast -n stenc-ns --ignore-not-found
$KUBECTL --kubeconfig=$KUBECONFIG apply -f k8s/wipe-s3-job.yaml
$KUBECTL --kubeconfig=$KUBECONFIG wait --for=condition=complete job/wipe-s3-fast -n stenc-ns --timeout=300s || true

echo "2/4: Triggering new Species Generation Job with the updated general model..."
$KUBECTL --kubeconfig=$KUBECONFIG delete job pollen-build-species-dataset -n stenc-ns --ignore-not-found
$KUBECTL --kubeconfig=$KUBECONFIG apply -f k8s/pollen-build-species-dataset-job.yaml

echo "3/4: Waiting for Dataset Extraction Job on the active GPU cluster... (~5-10 minutes)"
$KUBECTL --kubeconfig=$KUBECONFIG wait --for=condition=complete job/pollen-build-species-dataset -n stenc-ns --timeout=3600s

echo "4/4: Downloading newly packaged datasets to local environment..."
rm -rf ~/Desktop/Species_model/*
echo "Local staging area wiped. Tracing S3 outputs..."
.venv/bin/python download_s3.py

echo "✅ ALL PIPELINE STEPS SUCCESSFULLY COMPLETED!"
