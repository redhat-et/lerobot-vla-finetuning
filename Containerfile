# syntax=docker/dockerfile:1
#
# LeRobot VLA fine-tuning image for Podman/OpenShift.
# - UBI 9 / Python 3.12 runtime
# - PyTorch CUDA 12.8
# - TorchCodec CUDA wheel
# - FFmpeg 7.1.1 built as shared libraries in a UBI 9 builder stage, with
#   libdav1d (AV1) and libvpx (VP9) decode support -- FFmpeg's own
#   built-in/native AV1 decoder is listed by `ffmpeg -decoders` but does
#   NOT actually decode all real-world AV1 content (confirmed failure:
#   "Could not push packet to decoder: Function not implemented" on
#   lerobot/libero, which is AV1-encoded, while nvidia's H264-encoded
#   dataset worked fine with the native decoder). dav1d is the de facto
#   standard AV1 decoder (used by browsers, VLC, etc.) and does not have
#   this gap. libaom (AV1 encoder) is used ONLY in the builder stage to
#   synthesize AV1/VP9 test fixtures for the runtime decode smoke test
#   below. If it was available at build time, FFmpeg links against it
#   (--enable-libaom) and its runtime .so is REQUIRED in the final image
#   too -- there is no such thing as an "optional" enabled codec library
#   at the shared-library level; ffmpeg fails to load AT ALL (not just
#   AV1 encoding) without it. libdav1d/libvpx/libaom availability is
#   detected at build time (this UBI9 CodeReady Builder mirror does not
#   always carry the same package set as a full RHEL subscription); only
#   libdav1d is a hard requirement (it's the confirmed decode fix),
#   libvpx/libaom degrade gracefully to "feature not built" if missing
#   -- but whichever ARE built in must have their runtime package
#   installed below, unconditionally, no exceptions.
# - /opt/lerobot-tools/compute_quantile_stats.py: fast q01/q99 dataset
#   stats for non-video features, bypassing video decoding (see the
#   script's own docstring for why this exists)
#
# Uses EPEL9 (libdav1d/libvpx/libaom packages) and the UBI9 CodeReady
# Builder repo (-devel packages, builder stage only). No Conda/Mamba or
# RPM Fusion.

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
    dnf install -y --setopt=install_weak_deps=False \
      https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm; \
    CRB_REPO_ID="$(dnf repolist --all 2>/dev/null | awk 'tolower($0) ~ /(codeready|crb)/ && tolower($0) !~ /(debug|source)/ {print $1; exit}')"; \
    echo "Detected CodeReady Builder repo id: ${CRB_REPO_ID:-<none found>}"; \
    if [ -z "${CRB_REPO_ID}" ]; then \
      echo "No CodeReady Builder / CRB repo found -- listing all repos for debugging:" >&2; \
      dnf repolist --all >&2; \
      exit 1; \
    fi; \
    # libdav1d-devel is REQUIRED: it's the confirmed fix for a real AV1
    # decode failure (FFmpeg's own native AV1 decoder is *listed* by
    # `ffmpeg -decoders` but does not actually decode all real-world AV1
    # content -- see the top-of-file comment). Fail the build loudly if
    # this specific package is missing rather than silently shipping the
    # same broken decoder again.
    dnf install -y --setopt=install_weak_deps=False \
      --enablerepo="${CRB_REPO_ID}" \
      libdav1d-devel; \
    # libvpx-devel (VP9) and libaom-devel (AV1 encoder, used ONLY to
    # build test fixtures below) are best-effort extras: this UBI9 CRB
    # mirror does not always carry the same package set as a full RHEL
    # subscription or CentOS Stream CRB. Missing them does not block the
    # build -- FFmpeg is configured with only what actually installed.
    HAVE_LIBVPX=0; \
    if dnf install -y --setopt=install_weak_deps=False \
        --enablerepo="${CRB_REPO_ID}" libvpx-devel; then \
      HAVE_LIBVPX=1; \
    else \
      echo "WARNING: libvpx-devel unavailable -- building without VP9 support" >&2; \
    fi; \
    HAVE_LIBAOM=0; \
    if dnf install -y --setopt=install_weak_deps=False \
        --enablerepo="${CRB_REPO_ID}" libaom-devel; then \
      HAVE_LIBAOM=1; \
    else \
      echo "WARNING: libaom-devel unavailable -- no AV1 test fixture will be generated (dav1d decode support is unaffected; only the encoder needed to synthesize a test file is missing)" >&2; \
    fi; \
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
    CONFIGURE_FLAGS="--prefix=/opt/ffmpeg --libdir=/opt/ffmpeg/lib64 --shlibdir=/opt/ffmpeg/lib64 --enable-shared --enable-zlib --enable-libdav1d --disable-static --disable-debug --disable-doc --disable-htmlpages --disable-manpages --disable-podpages --disable-txtpages --disable-x86asm"; \
    [ "${HAVE_LIBVPX}" = "1" ] && CONFIGURE_FLAGS="${CONFIGURE_FLAGS} --enable-libvpx"; \
    [ "${HAVE_LIBAOM}" = "1" ] && CONFIGURE_FLAGS="${CONFIGURE_FLAGS} --enable-libaom"; \
    echo "FFmpeg configure flags: ${CONFIGURE_FLAGS}"; \
    ./configure ${CONFIGURE_FLAGS}; \
    make -j"$(getconf _NPROCESSORS_ONLN)"; \
    make install; \
    printf '%s\n' /opt/ffmpeg/lib64 > /etc/ld.so.conf.d/ffmpeg.conf; \
    ldconfig; \
    /opt/ffmpeg/bin/ffmpeg -version; \
    /opt/ffmpeg/bin/ffmpeg -decoders 2>&1 | grep -E '\bav1\b|\bvp9\b' || true; \
    find /opt/ffmpeg/lib64 -maxdepth 1 \( -type f -o -type l \) | sort; \
    mkdir -p /opt/codec-fixtures; \
    if [ "${HAVE_LIBAOM}" = "1" ]; then \
      /opt/ffmpeg/bin/ffmpeg \
        -hide_banner \
        -loglevel error \
        -f lavfi \
        -i testsrc2=size=256x256:rate=10 \
        -t 1 \
        -c:v libaom-av1 \
        -cpu-used 8 \
        -y \
        /opt/codec-fixtures/av1-test.mp4; \
    fi; \
    if [ "${HAVE_LIBVPX}" = "1" ]; then \
      /opt/ffmpeg/bin/ffmpeg \
        -hide_banner \
        -loglevel error \
        -f lavfi \
        -i testsrc2=size=256x256:rate=10 \
        -t 1 \
        -c:v libvpx-vp9 \
        -y \
        /opt/codec-fixtures/vp9-test.mp4; \
    fi; \
    printf '%s\n' "${HAVE_LIBVPX}" > /opt/codec-fixtures/.have-libvpx; \
    printf '%s\n' "${HAVE_LIBAOM}" > /opt/codec-fixtures/.have-libaom

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
COPY --from=ffmpeg-builder /opt/codec-fixtures /opt/codec-fixtures
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
      https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm; \
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
      pkgconf-pkg-config \
      libdav1d; \
    if [ "$(cat /opt/codec-fixtures/.have-libvpx 2>/dev/null)" = "1" ]; then \
      dnf install -y --setopt=install_weak_deps=False libvpx; \
    fi; \
    if [ "$(cat /opt/codec-fixtures/.have-libaom 2>/dev/null)" = "1" ]; then \
      dnf install -y --setopt=install_weak_deps=False libaom; \
    fi; \
    dnf clean all; \
    rm -rf /var/cache/dnf; \
    printf '%s\n' /opt/ffmpeg/lib64 > /etc/ld.so.conf.d/ffmpeg.conf; \
    ldconfig; \
    /opt/ffmpeg/bin/ffmpeg -version; \
    ldconfig -p | grep -E 'libav(codec|format|util)|libsw(scale|resample)'; \
    ldconfig -p | grep -E 'libdav1d|libvpx' || true; \
    /opt/ffmpeg/bin/ffmpeg -decoders 2>&1 | grep -E '\bav1\b|\bvp9\b' || true

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

# AV1 and VP9 decode -- real, non-trivial content (moving test pattern, not
# a single flat color), not just a check that the decoder is *listed*.
# `ffmpeg -decoders` listing "av1"/"vp9" is NOT sufficient evidence the
# decoder actually works: this exact gap (decoder listed, but failing on
# real AV1 content with "Could not push packet to decoder: Function not
# implemented") is what broke training on lerobot/libero before dav1d/vpx
# were added here.
RUN python - <<'PYVERIFY'
from pathlib import Path

from torchcodec.decoders import VideoDecoder

fixtures = [("AV1", "/opt/codec-fixtures/av1-test.mp4"), ("VP9", "/opt/codec-fixtures/vp9-test.mp4")]
tested_any = False

for name, path in fixtures:
    if not Path(path).exists():
        print(f"{name} fixture not present (encoder unavailable at build time) -- skipping")
        continue
    tested_any = True
    decoder = VideoDecoder(path)
    assert decoder.metadata.codec == name.lower(), f"{path}: expected codec {name.lower()}, got {decoder.metadata.codec}"
    frames = decoder.get_frames_at(indices=[0, 5, 9])
    print(f"{name} decoded OK: codec={decoder.metadata.codec}, frames shape={tuple(frames.data.shape)}")

if not tested_any:
    print("NOTE: no AV1/VP9 fixtures were available to test -- libaom/libvpx were both unavailable at build time")
PYVERIFY

RUN rm -f /tmp/torchcodec-smoke.mp4 && rm -rf /opt/codec-fixtures

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
