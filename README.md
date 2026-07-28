## INSPECTION OF CLUSTER'S NODE GPU HARDWARE AND
## GENERATING requitements.txt FOR CONTAINER PACKAGES

## LOGIN TO THE CLUSTER AND INSPECT HARDWARE PROPERTIES
## LIKE POOL/DRIVER/CUDA_MAX/COMPUTE_CAP
```bash
oc get nodes -l nvidia.com/gpu.present=true -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.labels.gpu-pool-size}{"\t"}{.metadata.labels.nvidia\.com/gpu\.product}{"\t"}{.metadata.labels.nvidia\.com/gpu\.memory}{"\t"}{.metadata.labels.nvidia\.com/gpu\.count}{"\t"}{.metadata.labels.nvidia\.com/cuda\.driver-version\.full}{"\t"}{.metadata.labels.nvidia\.com/cuda\.runtime-version\.full}{"\t"}{.metadata.labels.nvidia\.com/gpu\.compute\.major}{"\t"}{.metadata.labels.nvidia\.com/gpu\.compute\.minor}{"\n"}{end}' | awk -F'\t' 'BEGIN{print "NODE\tPOOL\tMODEL\tVRAM_PER_GPU_MB\tCOUNT\tTOTAL_VRAM_GB\tDRIVER\tCUDA_MAX\tCOMPUTE_CAP"} {printf "%s\t%s\t%s\t%s\t%s\t%.0f\t%s\t%s\t%s.%s\n", $1,$2,$3,$4,$5,($4*$5)/1024,$6,$7,$8,$9}'
```

## GENERATING MANIFEST FILE FOR CONTAINER BUILD SETUP
```bash
chmod +x resolve_build_manifest_lerobot.sh

./resolve_build_manifest_lerobot.sh \
  --cuda-max 13.0 \
  --compute-cap 8.9 \
  --gpu-model "NVIDIA L40S" \
  --driver 580.126.20 \
  --pool xlarge \
  --lerobot-ref main \
  -o versions.env
```


## BUILD CONTAINER IMAGE
```bash
set -a
source versions.env
set +a

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
  -f Containerfile \
  -t lerobot-vla-finetuning:latest .
```
