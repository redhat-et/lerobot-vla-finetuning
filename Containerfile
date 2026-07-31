# syntax=docker/dockerfile:1
#
# LeRobot VLA fine-tuning image for Podman/OpenShift.
# - UBI 9 / Python 3.12 runtime
# - PyTorch CUDA 12.8
# - TorchCodec CUDA wheel
# - FFmpeg 7.1.1 built as shared libraries in a UBI 9 builder stage
# - /opt/lerobot-tools/compute_quantile_stats.py: fast q01/q99 dataset
#   stats for non-video features, bypassing video decoding (see the
#   script's own docstring for why this exists)
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
      xz \
      zlib-devel; \
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
      --enable-zlib \
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
ARG TORCHCODEC_VERSION
ARG NVIDIA_NPP_PACKAGE
ARG NVIDIA_NPP_VERSION
ARG FFMPEG_VERSION

USER 0

COPY --from=ffmpeg-builder /opt/ffmpeg /opt/ffmpeg
RUN mkdir -p /opt/lerobot-tools && cat > /opt/lerobot-tools/compute_quantile_stats.py <<'COMPUTE_QUANTILE_STATS_EOF'
#!/usr/bin/env python3
"""Compute q01/q10/q50/q90/q99 (and min/max/mean/std/count) statistics for
selected non-video features of a LeRobot v3.0 dataset, reading directly from
the parquet data shards -- bypassing video decoding entirely.

Why this exists: LeRobot's own `augment_dataset_quantile_stats` computes
stats for every feature in the dataset, including video ones. When video
keys are present it falls back to fully sequential per-episode processing
"for thread safety", which does not parallelize and can take many hours on
datasets with thousands of episodes. This script only reads the numeric
feature columns that actually need QUANTILES normalization (typically
`action` and `observation.state`, with VISUAL: IDENTITY in
policy.normalization_mapping) -- meaning no dataset stats are ever
consulted for video features at all (see
lerobot/processor/normalize_processor.py: NormalizationMode.IDENTITY
short-circuits before any stats lookup). Skipping video is therefore not a
shortcut but the same outcome the training pipeline gets anyway, computed
without ever touching a single video frame.

Assumes the dataset is already in LeRobot v3.0 layout (parquet shards
under <root>/data/**/*.parquet, feature metadata in meta/info.json). Run
the v2.1->v3.0 conversion first if needed.

Usage:
    python compute_quantile_stats.py \
        --root /path/to/dataset/root \
        --features action,observation.state

Idempotent and non-destructive: if meta/stats.json already exists, only
the requested feature entries are added/overwritten -- every other entry
(including any existing video stats) is left untouched.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

QUANTILE_LEVELS = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}


def find_parquet_shards(dataset_root: Path) -> list[Path]:
    data_dir = dataset_root / "data"
    if not data_dir.is_dir():
        raise SystemExit(f"no 'data' directory found under {dataset_root}")
    shards = sorted(data_dir.glob("**/*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet files found under {data_dir}")
    return shards


def load_feature_columns(
    shards: list[Path], features: list[str]
) -> dict[str, np.ndarray]:
    """Read only the requested columns across all shards and stack them
    into one (num_rows, feature_dim) float64 array per feature name."""
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in features}

    for shard_path in shards:
        pf = pq.ParquetFile(shard_path)
        available = set(pf.schema_arrow.names)
        missing = [f for f in features if f not in available]
        if missing:
            raise SystemExit(
                f"{shard_path} is missing requested column(s): "
                + ", ".join(missing)
            )
        for batch in pf.iter_batches(columns=features):
            for name in features:
                col = batch.column(name)
                arr = col.to_numpy(zero_copy_only=False)
                if arr.dtype == object:
                    # Fixed-length numeric vector feature: Arrow list
                    # column, one list/array per row -> stack to (N, D).
                    stacked = np.stack(
                        [np.asarray(row, dtype=np.float64) for row in arr]
                    )
                else:
                    # Plain scalar column -> treat as (N, 1).
                    stacked = np.asarray(arr, dtype=np.float64).reshape(-1, 1)
                chunks[name].append(stacked)

    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}


def compute_stats(values: np.ndarray) -> dict:
    """values: shape (N, D). Returns a dict matching LeRobot's stats.json
    per-feature schema: min/max/mean/std/count/q01/q10/q50/q90/q99, each a
    flat list of length D (count is a length-1 list)."""
    stats = {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0, ddof=0).tolist(),
        "count": [int(values.shape[0])],
    }
    quantiles = np.quantile(values, list(QUANTILE_LEVELS.values()), axis=0)
    for key, row in zip(QUANTILE_LEVELS.keys(), quantiles):
        stats[key] = np.atleast_1d(row).tolist()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", required=True, help="Dataset root directory (LeRobot v3.0 layout)"
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Comma-separated list of non-video feature names to compute stats for",
    )
    args = parser.parse_args()

    dataset_root = Path(args.root)
    features = [f.strip() for f in args.features.split(",") if f.strip()]
    if not features:
        raise SystemExit("--features must list at least one feature name")

    info_path = dataset_root / "meta" / "info.json"
    with info_path.open() as stream:
        info = json.load(stream)

    dataset_features = info.get("features", {})
    for name in features:
        feature_info = dataset_features.get(name)
        if feature_info is None:
            raise SystemExit(f"feature '{name}' not found in {info_path}")
        if feature_info.get("dtype") == "video":
            raise SystemExit(
                f"feature '{name}' is a video feature; this script is only "
                "for numeric (non-video) features"
            )

    shards = find_parquet_shards(dataset_root)
    print(
        f"Found {len(shards)} parquet shard(s) under {dataset_root / 'data'}",
        file=sys.stderr,
    )

    columns = load_feature_columns(shards, features)

    stats_path = dataset_root / "meta" / "stats.json"
    if stats_path.exists():
        with stats_path.open() as stream:
            all_stats = json.load(stream)
    else:
        all_stats = {}

    for name in features:
        values = columns[name]
        print(
            f"Computing stats for '{name}': {values.shape[0]} rows x "
            f"{values.shape[1]} dim(s)",
            file=sys.stderr,
        )
        all_stats[name] = compute_stats(values)

    tmp_path = stats_path.with_suffix(".json.tmp")
    with tmp_path.open("w") as stream:
        json.dump(all_stats, stream, indent=4)
    tmp_path.replace(stats_path)

    print(f"Wrote stats for {', '.join(features)} to {stats_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
COMPUTE_QUANTILE_STATS_EOF

# Runtime-only RHEL libraries. CUDA runtime libraries are supplied by the
# selected PyTorch CUDA wheels. NPP is installed below as an NVIDIA wheel
# (NVIDIA_NPP_PACKAGE/NVIDIA_NPP_VERSION from versions.env) because
# TorchCodec's CUDA wheel dlopen's libnppicc at runtime and does not bundle
# it itself.
RUN set -eux; \
    : "${PYTHON_VERSION:?PYTHON_VERSION build argument is required}"; \
    : "${PYTHON_ABI:?PYTHON_ABI build argument is required}"; \
    : "${PYTORCH_INDEX_URL:?PYTORCH_INDEX_URL build argument is required}"; \
    : "${TORCH_VERSION:?TORCH_VERSION build argument is required}"; \
    : "${TORCHVISION_VERSION:?TORCHVISION_VERSION build argument is required}"; \
    : "${LEROBOT_VERSION:?LEROBOT_VERSION build argument is required}"; \
    : "${PYTORCH_CUDA_BRANCH:?PYTORCH_CUDA_BRANCH build argument is required}"; \
    : "${TORCHCODEC_VERSION:?TORCHCODEC_VERSION build argument is required}"; \
    : "${NVIDIA_NPP_PACKAGE:?NVIDIA_NPP_PACKAGE build argument is required}"; \
    : "${NVIDIA_NPP_VERSION:?NVIDIA_NPP_VERSION build argument is required}"; \
    case "${TARGET_ARCH}" in \
      x86_64) ;; \
      aarch64) echo "TARGET_ARCH=aarch64 unsupported: TorchCodec has no Linux ARM wheels (see https://huggingface.co/docs/lerobot/en/installation); use versions.env from the resolver, which already rejects aarch64" >&2; exit 1 ;; \
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
      libpng \
      pkgconf-pkg-config; \
    dnf clean all; \
    rm -rf /var/cache/dnf; \
    printf '%s\n' /opt/ffmpeg/lib64 > /etc/ld.so.conf.d/ffmpeg.conf; \
    ldconfig; \
    /opt/ffmpeg/bin/ffmpeg -version; \
    ldconfig -p | grep -E 'libav(codec|format|util)|libsw(scale|resample)'

# REMOVED ARGUMENT "CUDA_VISIBLE_DEVICES=0" TO ALLOW THIS IMAGE WORK WITH ANY NUMBER OF GPUS
ENV PATH=/opt/ffmpeg/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/ffmpeg/lib64:${LD_LIBRARY_PATH} \
    PKG_CONFIG_PATH=/opt/ffmpeg/lib64/pkgconfig \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
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
      "torchcodec==${TORCHCODEC_VERSION}"; \
    python -m pip install \
      --no-deps \
      "${NVIDIA_NPP_PACKAGE}==${NVIDIA_NPP_VERSION}"; \
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
    NPP_CUDA_MAJOR="${NVIDIA_NPP_PACKAGE##*-cu}"; \
    ldconfig -p | grep "libnppicc.so.${NPP_CUDA_MAJOR}"; \
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

# Reusable tool: compute q01/q10/q50/q99 dataset stats for non-video
# features directly from parquet, bypassing video decoding entirely (the
# official lerobot.scripts.augment_dataset_quantile_stats decodes every
# video frame and falls back to fully sequential per-episode processing
# whenever video keys are present, which can take many hours on
# datasets with thousands of episodes). Needs pyarrow/numpy, both pulled
# in transitively by lerobot[dataset] (via lerobot[training] above).
RUN chmod +x /opt/lerobot-tools/compute_quantile_stats.py && \
    python -c "import pyarrow, numpy; print('pyarrow', pyarrow.__version__, '/ numpy', numpy.__version__)" && \
    python /opt/lerobot-tools/compute_quantile_stats.py --help >/dev/null && \
    echo "compute_quantile_stats.py OK"

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
    chgrp -R 0 /workspace /opt/app-root /opt/ffmpeg /opt/lerobot-tools && \
    chmod -R g=u /workspace /opt/app-root /opt/ffmpeg /opt/lerobot-tools

ENV PYTHON_VERSION=${PYTHON_VERSION} \
    PYTHON_ABI=${PYTHON_ABI} \
    PYTORCH_CUDA_BRANCH=${PYTORCH_CUDA_BRANCH} \
    PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL} \
    TORCH_VERSION=${TORCH_VERSION} \
    TORCHVISION_VERSION=${TORCHVISION_VERSION} \
    LEROBOT_VERSION=${LEROBOT_VERSION} \
    LEROBOT_EXTRAS=${LEROBOT_EXTRAS} \
    TORCHCODEC_VERSION=${TORCHCODEC_VERSION} \
    NVIDIA_NPP_PACKAGE=${NVIDIA_NPP_PACKAGE} \
    NVIDIA_NPP_VERSION=${NVIDIA_NPP_VERSION} \
    FFMPEG_VERSION=${FFMPEG_VERSION} \
    GPU_COMPUTE_CAPABILITY=${GPU_COMPUTE_CAPABILITY} \
    GPU_SM=${GPU_SM}

LABEL org.opencontainers.image.title="LeRobot VLA Fine-Tuning Runtime" \
      org.opencontainers.image.description="Red Hat UBI with PyTorch CUDA, TorchCodec, CUDA NPP and FFmpeg shared libraries" \
      org.opencontainers.image.source="https://github.com/huggingface/lerobot"

USER 1001

CMD ["python", "-c", "import importlib.metadata as m, torch; from torchcodec.decoders import VideoDecoder; print('lerobot', m.version('lerobot')); print('torch', torch.__version__); print('torchcodec', m.version('torchcodec')); print('CUDA available', torch.cuda.is_available()); print('VideoDecoder', VideoDecoder)"]
