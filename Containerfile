# syntax=docker/dockerfile:1
#
# LeRobot VLA fine-tuning image for Podman/OpenShift.
# - UBI 9 / Python 3.12 runtime
# - PyTorch CUDA 12.8
# - TorchCodec CUDA wheel
# - FFmpeg 7.1.1 built as shared libraries in a UBI 9 builder stage
#
# No Conda/Mamba, EPEL, or RPM Fusion repositories are used.

ARG UBI_PYTHON_IMAGE
ARG FFMPEG_VERSION=7.1.1

# Build FFmpeg against the same UBI 9 glibc as the final image.
FROM registry.access.redhat.com/ubi9/ubi:latest AS ffmpeg-builder

ARG FFMPEG_VERSION

USER 0

RUN set -eux; \
    dnf install -y --setopt=install_weak_deps=False \
      ca-certificates \
      curl-minimal \
      diffutils \
      gcc \
      make \
      tar \
      xz; \
    dnf clean all; \
    rm -rf /var/cache/dnf; \
    curl -fL --retry 5 --retry-delay 2 \
      "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" \
      -o /tmp/ffmpeg.tar.xz; \
    mkdir -p /tmp/ffmpeg-src; \
    tar -xJf /tmp/ffmpeg.tar.xz \
      --strip-components=1 \
      -C /tmp/ffmpeg-src; \
    cd /tmp/ffmpeg-src; \
    ./configure \
      --prefix=/opt/ffmpeg \
      --libdir=/opt/ffmpeg/lib64 \
      --shlibdir=/opt/ffmpeg/lib64 \
      --enable-shared \
      --disable-static \
      --disable-debug \
      --disable-doc \
      --disable-htmlpages \
      --disable-manpages \
      --disable-podpages \
      --disable-txtpages \
      --disable-x86asm; \
    make -j"$(getconf _NPROCESSORS_ONLN)"; \
    make install; \
    printf '%s\n' /opt/ffmpeg/lib64 > /etc/ld.so.conf.d/ffmpeg.conf; \
    ldconfig; \
    /opt/ffmpeg/bin/ffmpeg -version; \
    find /opt/ffmpeg/lib64 -maxdepth 1 \( -type f -o -type l \) | sort

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
ARG TORCHCODEC_VERSION=0.11.1
ARG NVIDIA_NPP_VERSION=12.3.3.100
ARG FFMPEG_VERSION

USER 0

COPY --from=ffmpeg-builder /opt/ffmpeg /opt/ffmpeg

# Runtime-only RHEL libraries. CUDA runtime libraries are supplied by the
# selected PyTorch CUDA wheels. NPP is installed below as an NVIDIA wheel
# because TorchCodec's CUDA wheel requires libnppicc.so.12.
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
    rm -rf /var/cache/dnf; \
    printf '%s\n' /opt/ffmpeg/lib64 > /etc/ld.so.conf.d/ffmpeg.conf; \
    ldconfig; \
    /opt/ffmpeg/bin/ffmpeg -version; \
    ldconfig -p | grep -E 'libav(codec|format|util)|libsw(scale|resample)'

ENV PATH=/opt/ffmpeg/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/ffmpeg/lib64:${LD_LIBRARY_PATH} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
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

# Install the exact GPU stack, LeRobot, CUDA TorchCodec, and the missing
# CUDA NPP runtime required by TorchCodec.
RUN set -eux; \
    python -m pip install \
      --extra-index-url "${PYTORCH_INDEX_URL}" \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      "lerobot[${LEROBOT_EXTRAS}]==${LEROBOT_VERSION}"; \
    python -m pip install \
      --force-reinstall \
      --no-deps \
      --index-url "${PYTORCH_INDEX_URL}" \
      "torchcodec==${TORCHCODEC_VERSION}+cu128"; \
    python -m pip install \
      --no-deps \
      "nvidia-npp-cu12==${NVIDIA_NPP_VERSION}"; \
    find "$(python -c 'import sys; print(sys.prefix)')" \
      -type f \
      -path '*/triton/backends/nvidia/bin/ptxas' \
      -exec chmod +x {} +; \
    SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"; \
    test -d "${SITE_PACKAGES}/nvidia/npp/lib"; \
    printf '%s\n' "${SITE_PACKAGES}/nvidia/npp/lib" \
      > /etc/ld.so.conf.d/nvidia-npp-python.conf; \
    find "${SITE_PACKAGES}/nvidia/npp/lib" \
      -maxdepth 1 \
      -type f \
      -name 'libnpp*.so*' \
      -print; \
    ldconfig; \
    ldconfig -p | grep 'libnppicc.so.12'; \
    TORCH_VERSION="${TORCH_VERSION}" \
    TORCHVISION_VERSION="${TORCHVISION_VERSION}" \
    LEROBOT_VERSION="${LEROBOT_VERSION}" \
    TORCHCODEC_VERSION="${TORCHCODEC_VERSION}" \
    python - <<'PYVERIFY'
import importlib.metadata as metadata
import os

import torch
import torchvision
import torchcodec
from torchcodec.decoders import VideoDecoder

expected_torch = os.environ["TORCH_VERSION"]
expected_torchvision = os.environ["TORCHVISION_VERSION"]
expected_lerobot = os.environ["LEROBOT_VERSION"]
expected_torchcodec = os.environ["TORCHCODEC_VERSION"]

actual_lerobot = metadata.version("lerobot")
actual_torchcodec = metadata.version("torchcodec")

if torch.__version__ != expected_torch:
    raise SystemExit(f"torch mismatch: {torch.__version__} != {expected_torch}")
if torchvision.__version__ != expected_torchvision:
    raise SystemExit(
        f"torchvision mismatch: {torchvision.__version__} != {expected_torchvision}"
    )
if actual_lerobot != expected_lerobot:
    raise SystemExit(f"lerobot mismatch: {actual_lerobot} != {expected_lerobot}")
if not actual_torchcodec.startswith(expected_torchcodec):
    raise SystemExit(
        f"torchcodec mismatch: {actual_torchcodec} does not start with "
        f"{expected_torchcodec}"
    )

print("LeRobot environment installed")
print("lerobot:", actual_lerobot)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchcodec:", actual_torchcodec)
print("wheel CUDA runtime:", torch.version.cuda)
print("compiled CUDA architectures:", torch.cuda.get_arch_list())
print("TorchCodec VideoDecoder:", VideoDecoder)
PYVERIFY

# End-to-end CPU decode test. This catches missing FFmpeg shared libraries
# during image build instead of failing after the model has loaded.
RUN set -eux; \
    ffmpeg \
      -hide_banner \
      -loglevel error \
      -f lavfi \
      -i testsrc2=size=128x96:rate=10 \
      -t 1 \
      -c:v mpeg4 \
      -y \
      /tmp/torchcodec-smoke.mp4

RUN python - <<'PYVERIFY'
from torchcodec.decoders import VideoDecoder

decoder = VideoDecoder("/tmp/torchcodec-smoke.mp4")
frame = decoder[0]
print("TorchCodec decoded frame shape:", tuple(frame.shape))
PYVERIFY

RUN rm -f /tmp/torchcodec-smoke.mp4

# OpenShift commonly starts the image with an arbitrary UID in group 0.
RUN mkdir -p \
      /workspace/data \
      /workspace/output \
      /workspace/.cache/huggingface/transformers \
      /workspace/.cache/huggingface/lerobot \
      /workspace/.cache/torch \
      /workspace/.cache/triton && \
    chgrp -R 0 /workspace /opt/app-root /opt/ffmpeg && \
    chmod -R g=u /workspace /opt/app-root /opt/ffmpeg

ENV PYTHON_VERSION=${PYTHON_VERSION} \
    PYTHON_ABI=${PYTHON_ABI} \
    PYTORCH_CUDA_BRANCH=${PYTORCH_CUDA_BRANCH} \
    PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL} \
    TORCH_VERSION=${TORCH_VERSION} \
    TORCHVISION_VERSION=${TORCHVISION_VERSION} \
    LEROBOT_VERSION=${LEROBOT_VERSION} \
    LEROBOT_EXTRAS=${LEROBOT_EXTRAS} \
    TORCHCODEC_VERSION=${TORCHCODEC_VERSION} \
    NVIDIA_NPP_VERSION=${NVIDIA_NPP_VERSION} \
    FFMPEG_VERSION=${FFMPEG_VERSION} \
    GPU_COMPUTE_CAPABILITY=${GPU_COMPUTE_CAPABILITY} \
    GPU_SM=${GPU_SM}

LABEL org.opencontainers.image.title="LeRobot VLA Fine-Tuning Runtime" \
      org.opencontainers.image.description="Red Hat UBI with PyTorch CUDA, TorchCodec, CUDA NPP and FFmpeg shared libraries" \
      org.opencontainers.image.source="https://github.com/huggingface/lerobot"

USER 1001

CMD ["python", "-c", "import importlib.metadata as m, torch; from torchcodec.decoders import VideoDecoder; print('lerobot', m.version('lerobot')); print('torch', torch.__version__); print('torchcodec', m.version('torchcodec')); print('CUDA available', torch.cuda.is_available()); print('VideoDecoder', VideoDecoder)"]
