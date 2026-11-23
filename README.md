# Serve Qwen3-32B with vLLM (TorchTPU) on Trillium TPU with Google Cloud Managed Lustre (Single-Host)

In this guide, we show how to serve Qwen3-32B (`Qwen/Qwen3-32B`, BF16 weights + FP8 KV Cache) with vLLM using the **TorchTPU** backend on Trillium (TPU v6e) using Google Kubernetes Engine (GKE), with the model weights and compilation caches stored on [Google Cloud Managed Lustre](https://docs.cloud.google.com/managed-lustre/docs/overview).

We deploy it on a single **ct6e-standard-8t** node pool (`2x4` topology, 8 TPU v6e chips, 256 GB HBM total) using **Tensor Parallelism (TP) size of 8**.

This recipe is the Managed Lustre variant of the [Hyperdisk-based Qwen3-32B single-host recipe](../single-host/README.md). Compared with that recipe:

| | Hyperdisk recipe | This recipe (Managed Lustre) |
| :---- | :---- | :---- |
| Model weights | Downloaded from Hugging Face by every pod into its own `ReadWriteOnce` Hyperdisk Balanced PVC | Downloaded **once** into Lustre by a CPU Job, then read by every pod from a `ReadWriteMany` PVC |
| XLA / torch.compile caches | Per-pod, on Hyperdisk | Shared across all replicas on Lustre and kept across restarts |
| Scaling `replicas` beyond 1 | Each replica re-downloads and re-compiles | Replicas reuse the same weights and warm compile cache |
| Hugging Face access at serve time | Required | Not required (`HF_HUB_OFFLINE=1`) |

## Verified Models

| Model | Parameters | Min TPUs (Chips) | Topology | HuggingFace |
| :---- | :---- | :---- | :---- | :---- |
| Qwen3-32B | 32.8B Dense | 8× (`ct6e-standard-8t`) | `2x4` (1 host) | [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) |

> [!TIP]
> `Qwen/Qwen3-32B` (~65 GB BF16 weights) can also fit on a 4-chip `ct6e-standard-4t` (`2x2`) node by changing `cloud.google.com/gke-tpu-topology` to `"2x2"`, `google.com/tpu` to `"4"`, and `--tensor-parallel-size=4`. If you do, also change the cache directory suffix (`qwen3-32b-v6e-2x4-tp8`) in `qwen3-32b-server.yaml` so the two configurations do not share a compile cache.

## Prerequisites

1. A GKE cluster with TPU v6e support, running GKE `1.33.2-gke.4780000` or later (for the Managed Lustre CSI driver add-on).
2. At least one CPU node pool in the cluster (used by the one-time model download Job).
3. `gcloud` CLI and `kubectl` installed and configured.
4. A Hugging Face account and access token with read access to `Qwen/Qwen3-32B` (used only for the download Job).
5. IAM roles: in addition to the GKE / TPU roles used by the Hyperdisk recipe, **Google Cloud Managed Lustre Admin** (`roles/lustre.admin`) to create the Lustre instance.

---

## Step 1: Define Parameters

Export the necessary environment variables for your Google Cloud project, GKE cluster and Lustre instance:

```shell
export PROJECT_ID="<YOUR_PROJECT_ID>"
export CLUSTER_NAME="<YOUR_CLUSTER_NAME>"
export REGION="<YOUR_REGION>"
export ZONE="<YOUR_ZONE>"
export NODEPOOL_NAME="qwen3-32b-v6e-pool"

export LUSTRE_INSTANCE_NAME="<YOUR_LUSTRE_INSTANCE_NAME>"
export LUSTRE_ZONE="${ZONE}"   # Keep Lustre in the same zone as the TPU nodes
```

---

## Step 2: Prepare the Cluster and Managed Lustre

### Enable the Managed Lustre CSI driver

For GKE clusters running `1.33.2-gke.4780000` or later:

```shell
gcloud container clusters update ${CLUSTER_NAME} \
    --location=${REGION} \
    --project=${PROJECT_ID} \
    --update-addons=LustreCsiDriver=ENABLED
```

For clusters running an earlier version, or to use an existing Managed Lustre instance that was created with the `gke-support-enabled` flag, enable the legacy port instead:

```shell
gcloud container clusters update ${CLUSTER_NAME} \
    --location=${REGION} \
    --project=${PROJECT_ID} \
    --enable-legacy-lustre-port
```

> [!NOTE]
> A node upgrade may be required for existing clusters. See [Node upgrade required](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/lustre-csi-driver-existing-instance#node-upgrade-required).

### Create the Managed Lustre instance

Create a Managed Lustre instance by following [Create an instance](https://docs.cloud.google.com/managed-lustre/docs/create-instance). Create it:

- on the **same VPC network** as the GKE cluster, and
- in the **same zone** as the TPU v6e node pool, to keep model-load latency low.

The weights for Qwen3-32B are ~65 GB and the compile caches add a few GB, so the smallest instance size and throughput tier allowed by Managed Lustre is enough for this recipe. Choose a higher throughput tier if you plan to scale to many replicas that start up at the same time.

Collect the instance details needed for the PersistentVolume:

```shell
gcloud lustre instances describe ${LUSTRE_INSTANCE_NAME} \
    --project=${PROJECT_ID} \
    --location=${LUSTRE_ZONE}
```

The output looks similar to the following:

```text
capacityGib: '18000'
filesystem: lfs
gkeSupportEnabled: true
mountPoint: 10.90.1.4@tcp:/lfs
name: projects/my-project/locations/us-east5-b/instances/my-lustre
network: projects/my-project/global/networks/default
perUnitStorageThroughput: '125'
state: ACTIVE
```

---

## Step 3: Create the Single-Host TPU Node Pool and Kubernetes Resources

1. Create a TPU v6e (Trillium) node pool with the `ct6e-standard-8t` machine type and `2x4` topology (1 host, 8 chips):

```shell
gcloud container node-pools create ${NODEPOOL_NAME} \
    --project=${PROJECT_ID} \
    --location=${REGION} \
    --node-locations=${ZONE} \
    --num-nodes=1 \
    --machine-type=ct6e-standard-8t \
    --tpu-topology=2x4 \
    --cluster=${CLUSTER_NAME}
```

> [!TIP]
> To serve multiple replicas, set `--num-nodes` to the number of replicas you plan to run (one `2x4` host per replica).

2. Configure `kubectl` to communicate with your cluster:

```shell
gcloud container clusters get-credentials ${CLUSTER_NAME} --location=${REGION} --project=${PROJECT_ID}
```

3. Create the namespace `vllm-qwen3-32b`:

```shell
kubectl create namespace vllm-qwen3-32b
```

4. Create a Kubernetes secret for your Hugging Face token (used only by the download Job):

```shell
export HF_TOKEN="YOUR_HUGGING_FACE_API_TOKEN"
kubectl create secret generic hf-secret \
    --namespace=vllm-qwen3-32b \
    --from-literal=hf_api_token=${HF_TOKEN}
```

5. Create the Lustre PersistentVolume and PersistentVolumeClaim. Edit [lustre_pvc.yaml](./lustre_pvc.yaml) and replace the `<...>` placeholders with the values from `gcloud lustre instances describe`:

| Placeholder | Source | Example |
| :---- | :---- | :---- |
| `<LUSTRE_CAPACITY>` | `capacityGib`, with a `Gi` suffix | `18000Gi` |
| `<LUSTRE_PROJECT_ID>` | Project that owns the instance | `my-project` |
| `<LUSTRE_ZONE>` | Zone of the instance | `us-east5-b` |
| `<LUSTRE_INSTANCE_NAME>` | Instance name | `my-lustre` |
| `<LUSTRE_IP_ADDRESS>` | IP part of `mountPoint` | `10.90.1.4` |
| `<LUSTRE_FILE_SYSTEM>` | `filesystem` | `lfs` |

Then apply it and confirm the claim is `Bound`:

```shell
cd tpu-recipes/inference/trillium/vLLM/vLLM-TorchTPU/Qwen3-32B/single-host-lustre
kubectl apply -f lustre_pvc.yaml
kubectl get pvc -n vllm-qwen3-32b lustre-qwen3-pvc
```

Creating the Lustre instance and its PV/PVC is a one-time setup.

---

## Step 4: Download the Model to Lustre

Run the one-time download Job in [model-download-job.yaml](./model-download-job.yaml). It runs on a CPU node and writes the weights to `/models/Qwen/Qwen3-32B` on the Lustre file system:

```shell
kubectl apply -f model-download-job.yaml
kubectl logs -n vllm-qwen3-32b job/qwen3-32b-download -f
```

Wait for the Job to complete:

```shell
kubectl wait -n vllm-qwen3-32b --for=condition=complete job/qwen3-32b-download --timeout=60m
```

> [!NOTE]
> If the weights are already in a Cloud Storage bucket, you can instead import them into Lustre with [Transfer data](https://docs.cloud.google.com/managed-lustre/docs/transfer-data). Place them at `/models/Qwen/Qwen3-32B` on the file system, or update `MODEL_DIR` in `qwen3-32b-server.yaml` to match.

Layout of the Lustre file system used by this recipe:

```text
/                                         (mounted at /data in the pods)
├── models/Qwen/Qwen3-32B/                (weights, tokenizer, config)
└── cache/qwen3-32b-v6e-2x4-tp8/
    ├── xla_cache/                        (VLLM_XLA_CACHE_PATH)
    └── torch_compile_cache/              (VLLM_TORCH_COMPILE_CACHE_DIR)
```

---

## Step 5: Apply the vLLM Server Manifest

Apply the vLLM server manifest using [qwen3-32b-server.yaml](./qwen3-32b-server.yaml):

```shell
kubectl apply -f qwen3-32b-server.yaml
```

Monitor the progress of the server deployment:

```shell
kubectl get pods -n vllm-qwen3-32b -w
```

Follow the server logs during model loading and XLA compilation:

```shell
kubectl logs -n vllm-qwen3-32b deployment/vllm-qwen3-32b -c vllm-torchtpu -f
```

Once the server has loaded the model and completed compilation, you will see:

```text
(APIServer pid=1) INFO:     Started server process [1]
(APIServer pid=1) INFO:     Waiting for application startup.
(APIServer pid=1) INFO:     Application startup complete.
```

The first start compiles and writes the XLA / torch.compile caches to Lustre. Later starts, including pods rescheduled onto a different node, load the weights from Lustre and reuse the warm caches, so they skip both the Hugging Face download and most of the compilation.

---

## Step 6: Scale Out (Optional)

Because the Lustre volume is `ReadWriteMany`, additional replicas can share the same weights and caches. After the first replica is `Ready` (so the compile cache is fully populated), scale the Deployment:

```shell
kubectl scale deployment/vllm-qwen3-32b -n vllm-qwen3-32b --replicas=4
```

Each replica needs its own `2x4` v6e host; make sure the node pool has enough nodes. The `vllm-qwen3-32b-service` Service load-balances requests across all ready replicas.

> [!IMPORTANT]
> Scale only after the first replica has finished compiling. Starting many replicas against an empty cache makes each of them compile in parallel and write to the same cache directory.

---

## Step 7: Interact with the Model (Optional)

1. Port-forward the vLLM service to port `8000` on your local machine:

```shell
kubectl port-forward -n vllm-qwen3-32b service/vllm-qwen3-32b-service 8000:8000
```

2. Send a test chat completion request. The server registers the model under `--served-model-name=Qwen/Qwen3-32B`, so clients use the same model name as in the Hyperdisk recipe:

```shell
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen3-32B",
        "messages": [
            {"role": "user", "content": "Explain how tensor parallelism works on TPU."}
        ],
        "max_tokens": 128,
        "temperature": 0.7
    }'
```

---

## Step 8: Run Serving Benchmarks

The namespace, Service name and served model name are the same as in the Hyperdisk recipe, so the three benchmark manifests from [Step 6 of that recipe](../single-host/README.md#step-6-run-serving-benchmarks) (`1k/1k`, `1k/8k`, `8k/1k`) work against this deployment unchanged. For example, for `1k/1k`, save the following manifest as `qwen3-32b-benchmark-1k1k.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: qwen3-32b-bench-1k1k
  namespace: vllm-qwen3-32b
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 60
  containers:
    - name: vllm-bench
      image: vllm/vllm-torchtpu:v0.29.0
      command: ["/bin/bash", "-c"]
      args:
        - |
          set -euo pipefail
          apt-get update && apt-get install -y git curl
          while ! curl -sf http://vllm-qwen3-32b-service:8000/health; do
            sleep 15
            echo "Waiting for vllm-qwen3-32b-service:8000/health..."
          done
          git clone https://github.com/SemiAnalysisAI/InferenceX.git /workspace/inferencex && \
          cd /workspace/inferencex && \
          git checkout f6c1f5b5d122bc4a62b93c9bd2919dfef68ccbcd && \
          python3 /workspace/inferencex/utils/bench_serving/benchmark_serving.py \
            --backend=vllm \
            --request-rate=inf \
            --percentile-metrics='ttft,tpot,itl,e2el' \
            --host=vllm-qwen3-32b-service \
            --port=8000 \
            --model=Qwen/Qwen3-32B \
            --tokenizer=Qwen/Qwen3-32B \
            --dataset-name=random \
            --random-input-len=1024 \
            --random-output-len=1024 \
            --random-range-ratio=0.8 \
            --num-prompts=320 \
            --max-concurrency=64 \
            --num-warmups=16 \
            --ignore-eos
      env:
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              key: hf_api_token
              name: hf-secret
              optional: true
```

Run it and follow the logs:

```shell
kubectl apply -f qwen3-32b-benchmark-1k1k.yaml
kubectl logs -n vllm-qwen3-32b qwen3-32b-bench-1k1k -f
kubectl delete -f qwen3-32b-benchmark-1k1k.yaml
```

Serving throughput and latency should match the Hyperdisk recipe, because weights are in HBM once loaded. The difference shows up in **startup time**: measure it with

```shell
kubectl get pod -n vllm-qwen3-32b -l app=vllm-qwen3-32b \
    -o jsonpath='{range .items[*]}{.metadata.name}{"  created="}{.metadata.creationTimestamp}{"  ready="}{.status.conditions[?(@.type=="Ready")].lastTransitionTime}{"\n"}{end}'
```

and compare a cold start (first pod, empty cache) with a warm start (`kubectl rollout restart deployment/vllm-qwen3-32b -n vllm-qwen3-32b`).

---

## Step 9: Cleanup

1. Delete the vLLM server and the download Job:

```shell
kubectl delete -f qwen3-32b-server.yaml
kubectl delete -f model-download-job.yaml --ignore-not-found
```

2. Delete the Lustre PV/PVC. The PV uses `persistentVolumeReclaimPolicy: Retain`, so the data on the Lustre instance is kept:

```shell
kubectl delete -f lustre_pvc.yaml
```

3. (Optional) Delete the Managed Lustre instance. **This permanently deletes the weights and caches stored on it:**

```shell
gcloud lustre instances delete ${LUSTRE_INSTANCE_NAME} \
    --project=${PROJECT_ID} \
    --location=${LUSTRE_ZONE}
```

4. (Optional) Delete the GKE TPU node pool:

```shell
gcloud container node-pools delete ${NODEPOOL_NAME} \
    --cluster=${CLUSTER_NAME} \
    --location=${REGION} \
    --project=${PROJECT_ID}
```
