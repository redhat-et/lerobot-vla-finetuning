# LeRobot VLA Fine-Tuning Container for OpenShift

> [!WARNING]
> **Experimental / Work in Progress**
>
> This setup is not finished yet and has not been fully validated for production use.
> The resolver logic, package versions, container dependencies, GPU runtime behavior,
> and OpenShift deployment flow may still change.

This repository contains an experimental workflow for building a Red Hat UBI-based
container image for LeRobot VLA fine-tuning.

The workflow:

1. Inspects the GPU hardware available in the OpenShift cluster.
2. Generates a `versions.env` manifest with exact compatible package versions.
3. Builds the container image locally with Podman.
4. Produces an image intended to run on an OpenShift GPU node.

## Current status

Implemented:

- GPU node inspection in OpenShift.
- Automatic generation of exact Python, CUDA, PyTorch, TorchVision, and LeRobot versions.
- Red Hat UBI-based container image.
- GPU-enabled PyTorch installation.
- LeRobot training dependencies.
- Podman-based local image build.

Still experimental:

- Full end-to-end fine-tuning validation.
- OpenShift deployment manifests.
- Multi-GPU and distributed training.
- Persistent dataset and model storage.
- Final dependency minimization.
- Production security and reproducibility checks.

## Prerequisites

The following tools are expected to be available:

- `oc`
- `podman`
- `bash`
- Python 3
- Access to an OpenShift cluster with NVIDIA GPU nodes

## 1. Inspect GPU nodes in OpenShift

Log in to the cluster and inspect the GPU pool, model, memory, driver, maximum CUDA
version, and compute capability:

```bash
oc get nodes -l nvidia.com/gpu.present=true -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.labels.gpu-pool-size}{"\t"}{.metadata.labels.nvidia\.com/gpu\.product}{"\t"}{.metadata.labels.nvidia\.com/gpu\.memory}{"\t"}{.metadata.labels.nvidia\.com/gpu\.count}{"\t"}{.metadata.labels.nvidia\.com/cuda\.driver-version\.full}{"\t"}{.metadata.labels.nvidia\.com/cuda\.runtime-version\.full}{"\t"}{.metadata.labels.nvidia\.com/gpu\.compute\.major}{"\t"}{.metadata.labels.nvidia\.com/gpu\.compute\.minor}{"\n"}{end}' | awk -F'\t' 'BEGIN{print "NODE\tPOOL\tMODEL\tVRAM_PER_GPU_MB\tCOUNT\tTOTAL_VRAM_GB\tDRIVER\tCUDA_MAX\tCOMPUTE_CAP"} {printf "%s\t%s\t%s\t%s\t%s\t%.0f\t%s\t%s\t%s.%s\n", $1,$2,$3,$4,$5,($4*$5)/1024,$6,$7,$8,$9}'
```

Example values used below:

```text
GPU model: NVIDIA L40S
Driver: 580.126.20
CUDA maximum: 13.0
Compute capability: 8.9
GPU pool: xlarge
```

## 2. Generate `versions.env`

Make the resolver executable:
Generate the version manifest:
```bash
chmod +x resolve_build_manifest_lerobot.sh

./resolve_build_manifest_lerobot.sh \
  --cuda-max 13.0 \
  --compute-cap 8.9 \
  --gpu-model "NVIDIA L40S" \
  --driver 580.126.20 \
  --pool xlarge \
  --lerobot-ref main \
  --out versions.env
```

The generated file contains the exact versions selected for the build, for example:

```text
PYTHON_VERSION=...
CUDA_TOOLKIT_VERSION=...
PYTORCH_CUDA_BRANCH=...
TORCH_VERSION=...
TORCHVISION_VERSION=...
LEROBOT_VERSION=...
```

Review the manifest before building:

```bash
cat versions.env
```

## 3. Load the version manifest

```bash
set -a
source versions.env
set +a

printf 'Python: %s\n' "$PYTHON_VERSION"
printf 'CUDA branch: %s\n' "$PYTORCH_CUDA_BRANCH"
printf 'PyTorch: %s\n' "$TORCH_VERSION"
printf 'TorchVision: %s\n' "$TORCHVISION_VERSION"
printf 'LeRobot: %s\n' "$LEROBOT_VERSION"
```

## 4. Build the container image

The example below installs LeRobot training dependencies for the `pi` policy:

```bash
podman build \
  --build-arg UBI_PYTHON_IMAGE="$UBI_PYTHON_IMAGE" \
  --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
  --build-arg PYTHON_ABI="$PYTHON_ABI" \
  --build-arg CUDA_TOOLKIT_VERSION="$CUDA_TOOLKIT_VERSION" \
  --build-arg PYTORCH_CUDA_BRANCH="$PYTORCH_CUDA_BRANCH" \
  --build-arg PYTORCH_INDEX_URL="$PYTORCH_INDEX_URL" \
  --build-arg TORCH_VERSION="$TORCH_VERSION" \
  --build-arg TORCHVISION_VERSION="$TORCHVISION_VERSION" \
  --build-arg LEROBOT_VERSION="$LEROBOT_VERSION" \
  --build-arg TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
  --build-arg GPU_COMPUTE_CAPABILITY="$GPU_COMPUTE_CAPABILITY" \
  --build-arg GPU_SM="$GPU_SM" \
  --build-arg TARGET_ARCH="$TARGET_ARCH" \
  --build-arg LEROBOT_EXTRAS="training,pi" \
  --file Containerfile \
  --tag lerobot-vla-finetuning:latest \
  .
```

## 5. Verify the image

List the built image:

```bash
podman images
```

Check the installed package versions:

```bash
podman run \
  --rm \
  lerobot-vla-finetuning:latest \
  python -c '
import importlib.metadata as metadata
import torch
import torchvision

print("lerobot:", metadata.version("lerobot"))
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("wheel CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
'
```

During `podman build`, `torch.cuda.is_available()` may be `False` because the build
container normally does not have access to the GPU.

## 6. Push container image to Quay.io
```bash
podman push quay.io/<username>/lerobot-vla-finetuning:latest
```

## 7. Create secrets
```bash
oc create secret docker-registry quay-pull-secret \
  --docker-server=quay.io \
  --docker-username='<QUAY_USERNAME>' \
  --docker-password='<QUAY_ROBOT_TOKEN>' \
  --docker-email='<EMAIL>'
oc get secret quay-pull-secret

oc create secret generic huggingface-credentials \
  --from-literal=token='<HF_TOKEN>'
oc get secret huggingface-credentials
```

## 8. Run smoke test
```bash
oc apply --dry-run=server -f ./k8s/storage-pvcs.yaml
oc apply --dry-run=server -f ./k8s/download-job.yaml
```

## 9. Create PVC
```bash
oc apply -f ./k8s/storage-pvcs.yaml      # creating
oc get pvc                               # checking
```

## 10. Run download job
```bash
oc apply -f ./k8s/download-job.yaml      # creating 
oc get jobs                              # checking
```

## 11. Run fine-tuning job
```bash
oc apply -f ./k8s/training-job.yaml      # creating 
oc get jobs                              # checking
```

The final dot is the Podman build context. In the current setup it can be the local
project directory containing the `Containerfile`.

## Notes

- `versions.env` is generated output and should contain exact selected versions.
- The resolver may query external package sources when determining compatibility.
- The container image is designed for GPU workloads, but the NVIDIA driver must remain
  on the OpenShift node rather than inside the image.
- The `LEROBOT_EXTRAS` argument controls which optional LeRobot components are installed.
- The example uses `training,pi`; other policies may require different extras.
- Do not treat the current image as production-ready until fine-tuning has been validated
  end to end on the target OpenShift cluster.

## Known limitations

- No completed OpenShift Job or KubeFlow Pipeline manifest is included yet.
- Dataset mounts, object storage, checkpoints, secrets, and Hugging Face authentication
  are not configured here.
- Image size and dependency selection may still be optimized.
- The exact training command is not yet defined in this README.
- Multi-GPU behavior has not yet been validated.
