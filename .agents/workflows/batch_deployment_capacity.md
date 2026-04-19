---
description: Pre-flight storage and capacity checklist for Kubernetes batch deployments
---

# Batch Deployment Capacity Process

Whenever deploying a Kubernetes batch workload across an entire directory or scaling an inference pipeline (e.g. running `deploy_general_infer.sh` or processing a full dataset):

1. **Calculate the Source Target Size**: Always calculate or request the total scale (in GBs) of the dataset that will be fetched from S3 or mounted into the environment. 
2. **Expand the Scratch Volume**: Ensure the respective Kubernetes job manifests (like `pollen-general-infer-job.yaml`) define an `emptyDir` or persistent `scratch` volume with a `sizeLimit` parameter set to be *significantly* larger than the raw source payload (at least 5-10x the raw data size to accommodate image overlays, uncompressed inference arrays, and staging artifact copies).
3. **Bind Container Output to Scratch**: Ensure the inference script's output flag (e.g. `--out`) points rigidly into the expanded volume path (like `/mnt/czi_data/output`) rather than the container’s default local filesystem (like `/app/output`). Container ephemeral limits natively sit at roughly 10-20Gi on most cloud nodes and will abruptly trigger unexpected Pod Evictions without trailing error stacks if overrun.

By formalizing this inspection pipeline, unexpected single-image cutoffs or `Evicted` disk exhaustion crashes are completely preempted.
