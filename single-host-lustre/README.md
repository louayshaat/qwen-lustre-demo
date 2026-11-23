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

1. A Google Cloud project with billing enabled and TPU v6e quota (or a reservation) in your chosen zone.
2. `gcloud` CLI and `kubectl` installed, and `gcloud auth login` completed.
3. IAM roles: Kubernetes Engine Admin, Compute Network Admin (for Private Services Access), Service Usage Admin, and **Google Cloud Managed Lustre Admin** (`roles/lustre.admin`).
4. (Optional) A Hugging Face access token. `Qwen/Qwen3-32B` is public, so the token only helps with rate limits.

The zone must support **both** TPU v6e and Managed Lustre. The values below use `us-east5-a`, which supports both.

---

## Step 1: Define Parameters

```shell
# Project and network
export PROJECT_ID="tpu-launchpad-playground"
export NETWORK="default"
export ZONE="us-east5-a"
export REGION="${ZONE%-*}"            # us-east5

# GKE
export CLUSTER_NAME="qwen-demo-cluster"
export NODEPOOL_NAME="qwen3-32b-v6e-pool"

# Managed Lustre
export LUSTRE_INSTANCE_NAME="qwen-demo"
export LUSTRE_ZONE="${ZONE}"           # same zone as the TPU nodes
export LUSTRE_FILE_SYSTEM="qwendemo"   # <= 8 characters, letters and numbers only
export LUSTRE_CAPACITY_GIB="9000"      # smallest instance size
export LUSTRE_THROUGHPUT="1000"        # MBps per TiB; 9,000 GiB is only allowed on the 1000 tier

gcloud config set project ${PROJECT_ID}
```

---

## Step 2: Enable the Required APIs

```shell
gcloud services enable \
    compute.googleapis.com \
    container.googleapis.com \
    lustre.googleapis.com \
    servicenetworking.googleapis.com \
    --project=${PROJECT_ID}
```

---

## Step 3: Set Up Private Services Access on the VPC

Managed Lustre is reached over [Private Services Access](https://docs.cloud.google.com/vpc/docs/private-services-access). The VPC needs a peering connection to `servicenetworking.googleapis.com` with an allocated range of at least `/20`.

1. Check whether the network already has a Private Services Access connection:

```shell
gcloud services vpc-peerings list \
    --network=${NETWORK} \
    --project=${PROJECT_ID}
```

If a connection is listed, skip to [Step 4](#step-4-create-the-managed-lustre-instance).

2. Otherwise, allocate a `/20` range and create the connection:

```shell
gcloud compute addresses create lustre-psa-range \
    --global \
    --purpose=VPC_PEERING \
    --prefix-length=20 \
    --network=${NETWORK} \
    --project=${PROJECT_ID}

gcloud services vpc-peerings connect \
    --service=servicenetworking.googleapis.com \
    --ranges=lustre-psa-range \
    --network=${NETWORK} \
    --project=${PROJECT_ID}
```

---

## Step 4: Create the Managed Lustre Instance

```shell
gcloud lustre instances create ${LUSTRE_INSTANCE_NAME} \
    --project=${PROJECT_ID} \
    --location=${LUSTRE_ZONE} \
    --network=projects/${PROJECT_ID}/global/networks/${NETWORK} \
    --filesystem=${LUSTRE_FILE_SYSTEM} \
    --capacity-gib=${LUSTRE_CAPACITY_GIB} \
    --per-unit-storage-throughput=${LUSTRE_THROUGHPUT} \
    --async
```

`--async` returns immediately, so you can create the GKE cluster (Steps 5 and 6) while the instance is provisioning. Check its state with:

```shell
gcloud lustre instances describe ${LUSTRE_INSTANCE_NAME} \
    --location=${LUSTRE_ZONE} \
    --project=${PROJECT_ID} \
    --format="value(state)"
```

Wait for `ACTIVE` before Step 8.

> [!NOTE]
> The performance tier can't be changed after creation. Capacity can be increased later.

---

## Step 5: Create the GKE Cluster

Create a regional cluster with its nodes in `${ZONE}` and the **Managed Lustre CSI driver** add-on enabled. The default node pool is a small CPU pool that runs system pods and the one-time model download Job.

```shell
gcloud container clusters create ${CLUSTER_NAME} \
    --project=${PROJECT_ID} \
    --location=${REGION} \
    --node-locations=${ZONE} \
    --network=${NETWORK} \
    --subnetwork=${NETWORK} \
    --release-channel=regular \
    --workload-pool=${PROJECT_ID}.svc.id.goog \
    --machine-type=e2-standard-8 \
    --num-nodes=1 \
    --addons=LustreCsiDriver,GcePersistentDiskCsiDriver,HttpLoadBalancing
```

> [!NOTE]
> The `LustreCsiDriver` add-on needs GKE `1.33.2-gke.4780000` or later. To add it to an **existing** cluster instead, run `gcloud container clusters update ${CLUSTER_NAME} --location=${REGION} --update-addons=LustreCsiDriver=ENABLED`. A node upgrade may be required; see [Node upgrade required](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/persistent-volumes/lustre-csi-driver-existing-instance#node-upgrade-required).

Verify that the add-on is enabled (look for the Lustre CSI driver entry with `enabled: true`):

```shell
gcloud container clusters describe ${CLUSTER_NAME} \
    --location=${REGION} \
    --project=${PROJECT_ID} \
    --format="yaml(addonsConfig)"
```

---

## Step 6: Create the TPU v6e Node Pool

Create a TPU v6e (Trillium) node pool with the `ct6e-standard-8t` machine type and `2x4` topology (1 host, 8 chips):

```shell
gcloud container node-pools create ${NODEPOOL_NAME} \
    --project=${PROJECT_ID} \
    --cluster=${CLUSTER_NAME} \
    --location=${REGION} \
    --node-locations=${ZONE} \
    --num-nodes=1 \
    --machine-type=ct6e-standard-8t \
    --tpu-topology=2x4
```

> [!TIP]
> If you have a TPU reservation, add `--reservation-affinity=specific --reservation=<RESERVATION_NAME>`. To serve multiple replicas later, set `--num-nodes` to the number of replicas (one `2x4` host per replica).

---

## Step 7: Configure kubectl, Namespace and Secret

1. Fetch cluster credentials and check the nodes:

```shell
gcloud container clusters get-credentials ${CLUSTER_NAME} \
    --location=${REGION} \
    --project=${PROJECT_ID}

kubectl get nodes -L cloud.google.com/gke-tpu-accelerator,cloud.google.com/gke-tpu-topology
```

2. Create the namespace:

```shell
kubectl create namespace vllm-qwen3-32b
```

3. (Optional) Create the Hugging Face token secret used by the download Job:

```shell
export HF_TOKEN="YOUR_HUGGING_FACE_API_TOKEN"
kubectl create secret generic hf-secret \
    --namespace=vllm-qwen3-32b \
    --from-literal=hf_api_token=${HF_TOKEN}
```

---

## Step 8: Connect Lustre to the Cluster (PV/PVC)

1. Once the Lustre instance is `ACTIVE`, read its details:

```shell
gcloud lustre instances describe ${LUSTRE_INSTANCE_NAME} \
    --location=${LUSTRE_ZONE} \
    --project=${PROJECT_ID}
```

The output looks similar to the following:

```text
capacityGib: '9000'
filesystem: qwendemo
mountPoint: 10.90.1.4@tcp:/qwendemo
name: projects/tpu-launchpad-playground/locations/us-east5-a/instances/qwen-demo
network: projects/tpu-launchpad-playground/global/networks/default
perUnitStorageThroughput: '1000'
state: ACTIVE
```

2. Edit [lustre_pvc.yaml](./lustre_pvc.yaml) and replace the placeholders:

| Placeholder | Value |
| :---- | :---- |
| `<LUSTRE_CAPACITY>` | `9000Gi` |
| `<LUSTRE_PROJECT_ID>` | `tpu-launchpad-playground` |
| `<LUSTRE_ZONE>` | `us-east5-a` |
| `<LUSTRE_INSTANCE_NAME>` | `qwen-demo` |
| `<LUSTRE_IP_ADDRESS>` | IP part of `mountPoint`, e.g. `10.90.1.4` |
| `<LUSTRE_FILE_SYSTEM>` | `qwendemo` |

3. Apply it and confirm the claim is `Bound`:

```shell
cd tpu-recipes/inference/trillium/vLLM/vLLM-TorchTPU/Qwen3-32B/single-host-lustre
kubectl apply -f lustre_pvc.yaml
kubectl get pv lustre-qwen3-pv
kubectl get pvc -n vllm-qwen3-32b lustre-qwen3-pvc
```

---

## Step 9: Download the Model to Lustre

Run the one-time download Job in [model-download-job.yaml](./model-download-job.yaml). It runs on the CPU node pool and writes the weights to `/models/Qwen/Qwen3-32B` on the Lustre file system:

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

## Step 10: Deploy the vLLM Server

Apply the vLLM server manifest using [qwen3-32b-server.yaml](./qwen3-32b-server.yaml):

```shell
kubectl apply -f qwen3-32b-server.yaml
```

Monitor the deployment and follow the logs during model loading and XLA compilation:

```shell
kubectl get pods -n vllm-qwen3-32b -w
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

## Step 11: Interact with the Model

1. Port-forward the vLLM service to port `8000` on your local machine:

```shell
kubectl port-forward -n vllm-qwen3-32b service/vllm-qwen3-32b-service 8000:8000
```

2. Send a test chat completion request. The server registers the model under `--served-model-name=Qwen/Qwen3-32B`:

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

## Step 12: Show the Warm Start (Demo)

Restart the Deployment. The new pod loads the weights from Lustre and reuses the compile cache that the first pod wrote:

```shell
kubectl rollout restart deployment/vllm-qwen3-32b -n vllm-qwen3-32b
kubectl get pods -n vllm-qwen3-32b -w
```

Compare cold and warm start times (pod creation to `Ready`):

```shell
kubectl get pod -n vllm-qwen3-32b -l app=vllm-qwen3-32b \
    -o jsonpath='{range .items[*]}{.metadata.name}{"  created="}{.metadata.creationTimestamp}{"  ready="}{.status.conditions[?(@.type=="Ready")].lastTransitionTime}{"\n"}{end}'
```

---

## Step 13: Scale Out (Optional)

Because the Lustre volume is `ReadWriteMany`, more replicas can share the same weights and caches. Add TPU nodes first (one `2x4` host per replica), then scale the Deployment:

```shell
gcloud container clusters resize ${CLUSTER_NAME} \
    --node-pool=${NODEPOOL_NAME} \
    --num-nodes=2 \
    --location=${REGION} \
    --project=${PROJECT_ID}

kubectl scale deployment/vllm-qwen3-32b -n vllm-qwen3-32b --replicas=2
```

The `vllm-qwen3-32b-service` Service load-balances requests across all ready replicas.

> [!IMPORTANT]
> Scale only after the first replica has finished compiling. Starting many replicas against an empty cache makes each of them compile in parallel and write to the same cache directory.

---

## Step 14: Run Serving Benchmarks (Optional)

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

Serving throughput and latency should match the Hyperdisk recipe, because the weights are in HBM once loaded. The Lustre benefit shows up in startup time (Step 12) and in scale-out (Step 13).

---

## Step 15: Cleanup

1. Delete the Kubernetes resources:

```shell
kubectl delete -f qwen3-32b-server.yaml
kubectl delete -f model-download-job.yaml --ignore-not-found
kubectl delete -f lustre_pvc.yaml
```

The PV uses `persistentVolumeReclaimPolicy: Retain`, so the data on the Lustre instance is kept until you delete the instance.

2. Delete the GKE cluster (this also deletes the TPU node pool):

```shell
gcloud container clusters delete ${CLUSTER_NAME} \
    --location=${REGION} \
    --project=${PROJECT_ID}
```

3. Delete the Managed Lustre instance. **This permanently deletes the weights and caches stored on it:**

```shell
gcloud lustre instances delete ${LUSTRE_INSTANCE_NAME} \
    --location=${LUSTRE_ZONE} \
    --project=${PROJECT_ID}
```

4. (Optional) Remove the Private Services Access connection and range created in Step 3, **only if nothing else in the VPC uses them** (for example Cloud SQL, Memorystore or Filestore):

```shell
gcloud services vpc-peerings delete \
    --service=servicenetworking.googleapis.com \
    --network=${NETWORK} \
    --project=${PROJECT_ID}

gcloud compute addresses delete lustre-psa-range --global --project=${PROJECT_ID}
```
