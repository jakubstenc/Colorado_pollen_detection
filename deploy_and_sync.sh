#!/bin/bash
echo "Waiting for S3 wipe to complete..."
while true; do
    POD_STATUS=$(./kubectl --kubeconfig=kubeconfig.yaml get pods -l job-name=wipe-s3-fast -n stenc-ns -o jsonpath='{.items[0].status.phase}')
    if [ "$POD_STATUS" = "Succeeded" ]; then 
        echo "Wipe Succeeded!"
        break
    fi
    if [ "$POD_STATUS" = "Failed" ]; then 
        echo "Wipe failed! Aborting."
        exit 1
    fi
    sleep 10
done

echo "Deploying Species Dataset Builder..."
./kubectl --kubeconfig=kubeconfig.yaml delete job pollen-build-species-dataset -n stenc-ns --ignore-not-found
./kubectl --kubeconfig=kubeconfig.yaml apply -f k8s/pollen-build-species-dataset-job.yaml

echo "Waiting for Dataset Builder..."
while true; do
    POD_STATUS=$(./kubectl --kubeconfig=kubeconfig.yaml get pods -l job-name=pollen-build-species-dataset -n stenc-ns -o jsonpath='{.items[0].status.phase}' 2>/dev/null)
    if [ "$POD_STATUS" = "Succeeded" ]; then 
        echo "Dataset builder finished perfectly!"
        break
    fi
    if [ "$POD_STATUS" = "Failed" ]; then 
        echo "Dataset builder failed!"
        exit 1
    fi
    sleep 15
done

echo "Syncing new images to Desktop..."
export AWS_ACCESS_KEY_ID="1Y920BKC0SAWPNDE8RD6"
export AWS_SECRET_ACCESS_KEY="SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD"
mkdir -p ~/Desktop/Species_model
# Running aws s3 sync robustly
aws --endpoint-url https://s3.cl4.du.cesnet.cz s3 sync s3://bucket/PEG/Colorado/Species_model/ ~/Desktop/Species_model --no-verify-ssl

echo "All complete! The files are on the Desktop."
