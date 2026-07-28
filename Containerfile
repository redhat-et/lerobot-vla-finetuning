# syntax=docker/dockerfile:1
#
# LeRobot VLA fine-tuning image for Podman/OpenShift.
# Exact installable versions are supplied from versions.env.
# The LeRobot Git repository is not required in the build context.

ARG UBI_PYTHON_IMAGE
FROM ${UBI_PYTHON_IMAGE}

ARG PYTHON_VERSION
ARG PYTHON_ABI
ARG PYTORCH_CUDA_BRANCH
ARG PYTORCH_INDEX_URL
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCH_CUDA_ARCH_LIST
ARG GPU_COMPUTE_CAPABILITY
ARG GPU_SM
ARG TARGET_ARCH=x86_64
ARG LEROBOT_VERSION
ARG LEROBOT_EXTRAS=training

USER 0

# Runtime-only RHEL libraries. CUDA runtime libraries are supplied by the
# selected PyTorch CUDA wheels, so no CUDA Toolkit RPMs are installed here.
RUN set -eux; \
    : "${PYTHON_VERSION:?PYTHON_VERSION build argument is required}"; \
    : "${PYTHON_ABI:?PYTHON_ABI build argument is required}"; \
    : "${PYTORCH_INDEX_URL:?PYTORCH_INDEX_URL build argument is required}"; \
    : "${TORCH_VERSION:?TORCH_VERSION build argument is required}"; \
    : "${TORCHVISION_VERSION:?TORCHVISION_VERSION build argument is required}"; \
    : "${LEROBOT_VERSION:?LEROBOT_VERSION build argument is required}"; \
    case "${TARGET_ARCH}" in \
      x86_64|aarch64) ;; \
      *) echo "Unsupported TARGET_ARCH=${TARGET_ARCH}" >&2; exit 1 ;; \
    esac; \
    dnf install -y --setopt=install_weak_deps=False \
      ca-certificates \
      libusb1 \
      alsa-lib \
      libsndfile \
      mesa-libEGL \
      mesa-libGL \
      libglvnd-egl \
      libglvnd-glx \
      libjpeg-turbo \
      libpng; \
    dnf clean all; \
    rm -rf /var/cache/dnf

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=0 \
    DEVICE=cuda \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HF_LEROBOT_HOME=/workspace/.cache/huggingface/lerobot \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    TORCH_HOME=/workspace/.cache/torch \
    TRITON_CACHE_DIR=/workspace/.cache/triton \
    XDG_CACHE_HOME=/workspace/.cache

WORKDIR /workspace

# Validate the Python version provided by the selected UBI image.
RUN PYTHON_VERSION="${PYTHON_VERSION}" python - <<'PYVERIFY'
import os
import platform

expected = os.environ["PYTHON_VERSION"]
actual = ".".join(platform.python_version_tuple()[:2])
if actual != expected:
    raise SystemExit(
        f"Python version mismatch: versions.env={expected}, base image={actual}"
    )
print("Resolved Python validated:", actual)
PYVERIFY

# Install the exact resolved GPU stack and only the requested LeRobot extras.
# This is the only Python dependency installation step in the image.
RUN python -m pip install \
      --extra-index-url "${PYTORCH_INDEX_URL}" \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      "lerobot[${LEROBOT_EXTRAS}]==${LEROBOT_VERSION}" && \
    find "$(python -c 'import sys; print(sys.prefix)')" \
      -type f \
      -path '*/triton/backends/nvidia/bin/ptxas' \
      -exec chmod +x {} + && \
    TORCH_VERSION="${TORCH_VERSION}" \
    TORCHVISION_VERSION="${TORCHVISION_VERSION}" \
    LEROBOT_VERSION="${LEROBOT_VERSION}" \
    python - <<'PYVERIFY'
import importlib.metadata as metadata
import os
import torch
import torchvision

expected_torch = os.environ["TORCH_VERSION"]
expected_torchvision = os.environ["TORCHVISION_VERSION"]
expected_lerobot = os.environ["LEROBOT_VERSION"]
actual_lerobot = metadata.version("lerobot")

if torch.__version__ != expected_torch:
    raise SystemExit(f"torch mismatch: {torch.__version__} != {expected_torch}")
if torchvision.__version__ != expected_torchvision:
    raise SystemExit(
        f"torchvision mismatch: {torchvision.__version__} != {expected_torchvision}"
    )
if actual_lerobot != expected_lerobot:
    raise SystemExit(f"lerobot mismatch: {actual_lerobot} != {expected_lerobot}")

print("LeRobot environment installed")
print("lerobot:", actual_lerobot)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("wheel CUDA runtime:", torch.version.cuda)
print("compiled CUDA architectures:", torch.cuda.get_arch_list())
PYVERIFY

# OpenShift commonly starts the image with an arbitrary UID in group 0.
RUN mkdir -p \
      /workspace/data \
      /workspace/output \
      /workspace/.cache/huggingface/transformers \
      /workspace/.cache/huggingface/lerobot \
      /workspace/.cache/torch \
      /workspace/.cache/triton && \
    chgrp -R 0 /workspace /opt/app-root && \
    chmod -R g=u /workspace /opt/app-root

ENV PYTHON_VERSION=${PYTHON_VERSION} \
    PYTHON_ABI=${PYTHON_ABI} \
    PYTORCH_CUDA_BRANCH=${PYTORCH_CUDA_BRANCH} \
    PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL} \
    TORCH_VERSION=${TORCH_VERSION} \
    TORCHVISION_VERSION=${TORCHVISION_VERSION} \
    LEROBOT_VERSION=${LEROBOT_VERSION} \
    LEROBOT_EXTRAS=${LEROBOT_EXTRAS} \
    GPU_COMPUTE_CAPABILITY=${GPU_COMPUTE_CAPABILITY} \
    GPU_SM=${GPU_SM}

LABEL org.opencontainers.image.title="LeRobot VLA Fine-Tuning Runtime" \
      org.opencontainers.image.description="Red Hat UBI with exact PyTorch CUDA wheels and LeRobot for OpenShift GPU fine-tuning" \
      org.opencontainers.image.source="https://github.com/huggingface/lerobot"

USER 1001

CMD ["python", "-c", "import importlib.metadata as m, torch; print('lerobot', m.version('lerobot')); print('torch', torch.__version__); print('CUDA available', torch.cuda.is_available())"]
