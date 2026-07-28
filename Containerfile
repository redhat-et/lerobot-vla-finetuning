# syntax=docker/dockerfile:1
#
# VLA fine-tuning image for Podman/OpenShift.
# Version values are passed from versions.env as Podman build arguments.
# The NVIDIA driver is NOT installed in the image; OpenShift GPU nodes provide it.

# The resolver writes the exact UBI Python image to versions.env.
# Example: PYTHON_VERSION=3.12 -> ubi9/python-312.
ARG UBI_PYTHON_IMAGE
FROM ${UBI_PYTHON_IMAGE}

# Values resolved in versions.env.
ARG PYTHON_VERSION
ARG PYTHON_ABI
ARG CUDA_TOOLKIT_VERSION
ARG PYTORCH_CUDA_BRANCH
ARG PYTORCH_INDEX_URL
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCH_CUDA_ARCH_LIST
ARG GPU_COMPUTE_CAPABILITY
ARG GPU_SM
ARG TARGET_ARCH=x86_64

USER 0

# Install only the headless CUDA development components pinned to the selected
# major/minor release. The full cuda-toolkit meta-package pulls Nsight/GUI
# dependencies that are not appropriate for a minimal UBI/OpenShift image.
RUN set -eux; \
    test -n "${PYTHON_VERSION}"; \
    test -n "${PYTHON_ABI}"; \
    test -n "${CUDA_TOOLKIT_VERSION}"; \
    test -n "${PYTORCH_INDEX_URL}"; \
    test -n "${TORCH_VERSION}"; \
    test -n "${TORCHVISION_VERSION}"; \
    case "${TARGET_ARCH}" in \
      x86_64) NVIDIA_REPO_ARCH="x86_64" ;; \
      aarch64) NVIDIA_REPO_ARCH="sbsa" ;; \
      *) echo "Unsupported TARGET_ARCH=${TARGET_ARCH}" >&2; exit 1 ;; \
    esac; \
    CUDA_PACKAGE_VERSION="$(printf '%s' "${CUDA_TOOLKIT_VERSION}" | tr '.' '-')"; \
    dnf install -y --setopt=install_weak_deps=False curl-minimal ca-certificates; \
    curl --fail --location --retry 5 \
      "https://developer.download.nvidia.com/compute/cuda/repos/rhel9/${NVIDIA_REPO_ARCH}/cuda-rhel9.repo" \
      --output /etc/yum.repos.d/cuda-rhel9.repo; \
    dnf install -y --setopt=install_weak_deps=False \
      "cuda-compiler-${CUDA_PACKAGE_VERSION}" \
      "cuda-libraries-devel-${CUDA_PACKAGE_VERSION}" \
      gcc \
      gcc-c++ \
      git \
      make \
      ninja-build \
      which; \
    dnf clean all; \
    rm -rf /var/cache/dnf

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CUDA_TOOLKIT_VERSION=${CUDA_TOOLKIT_VERSION} \
    GPU_SM=${GPU_SM} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    XDG_CACHE_HOME=/workspace/.cache

# Install exactly the wheel versions selected by the resolver.
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
      --index-url "${PYTORCH_INDEX_URL}" \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" && \
    python - <<'PY'
import os
import torch
import torchvision

expected_cuda = os.environ.get("CUDA_TOOLKIT_VERSION")
expected_arch = os.environ.get("GPU_SM")

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torch wheel CUDA runtime:", torch.version.cuda)
print("compiled CUDA architectures:", torch.cuda.get_arch_list())
print("target GPU architecture:", expected_arch)
PY

# OpenShift normally runs containers with an arbitrary non-root UID whose
# primary group is 0. Group ownership and g=u permissions make the workspace
# writable without requiring a fixed UID.
RUN mkdir -p \
      /workspace \
      /workspace/.cache/huggingface/transformers \
      /workspace/data \
      /workspace/output && \
    chgrp -R 0 /workspace /opt/app-root && \
    chmod -R g=u /workspace /opt/app-root

WORKDIR /workspace

# Retain useful resolved values as OCI image metadata/environment variables.
ENV PYTHON_VERSION=${PYTHON_VERSION} \
    PYTHON_ABI=${PYTHON_ABI} \
    CUDA_TOOLKIT_VERSION=${CUDA_TOOLKIT_VERSION} \
    PYTORCH_CUDA_BRANCH=${PYTORCH_CUDA_BRANCH} \
    PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL} \
    TORCH_VERSION=${TORCH_VERSION} \
    TORCHVISION_VERSION=${TORCHVISION_VERSION} \
    GPU_COMPUTE_CAPABILITY=${GPU_COMPUTE_CAPABILITY} \
    GPU_SM=${GPU_SM}

LABEL org.opencontainers.image.title="VLA Fine-Tuning Runtime" \
      org.opencontainers.image.description="Red Hat UBI 9, Python, CUDA Toolkit and PyTorch environment for OpenShift GPU workloads"

# The UBI Python image uses an unprivileged application user. OpenShift may
# replace it with another arbitrary UID under its SecurityContextConstraints.
USER 1001

CMD ["python", "-c", "import torch; print('torch', torch.__version__); print('CUDA available', torch.cuda.is_available())"]
