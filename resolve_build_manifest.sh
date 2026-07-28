#!/usr/bin/env bash
#
# Resolve a consistent version manifest for a VLA fine-tuning container.
# The script does NOT build or pull a container image.
#
# It resolves:
#   - PyTorch CUDA wheel branch
#   - CUDA Toolkit major/minor for later container construction
#   - target Python version and CPython ABI (independent of host Python)
#   - compatible torch and torchvision wheels from the official PyTorch index
#   - GPU compute architecture
#
set -euo pipefail

readonly PYTORCH_INDEX_ROOT="https://download.pytorch.org/whl"
readonly OFFICIAL_VERSIONS_URL="https://pytorch.org/get-started/previous-versions/"
readonly OFFICIAL_COMPAT_URL="https://github.com/pytorch/vision#installation"
readonly OFFICIAL_LOCAL_URL="https://pytorch.org/get-started/locally/"

usage() {
  cat <<'USAGE'
Usage:
  ./gen_requirements.sh --cuda-max <X.Y> --compute-cap <X.Y> [options]

Required:
  --cuda-max <X.Y>       Maximum CUDA version supported by the target node driver
  --compute-cap <X.Y>    GPU compute capability, for example 8.9

Optional:
  --gpu-model <name>     GPU model written to the manifest
  --driver <version>     NVIDIA driver version written to the manifest
  --pool <label>         GPU pool/node label written to the manifest
  --target-arch <arch>   Target container CPU architecture: x86_64 or aarch64
                         Default: x86_64
  -o, --out <file>       Output manifest. Default: vla-build.env

Example:
  chmod +x gen_requirements.sh
  ./gen_requirements.sh --cuda-max 13.0 --compute-cap 8.9 \
      --gpu-model "NVIDIA L40S" --driver 580.126.20 --pool xlarge \
      -o vla-build.env
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_value() {
  local option="$1"
  local value="${2:-}"
  [[ -n "$value" && "$value" != --* ]] || die "$option requires a value"
}

CUDA_MAX=""
COMPUTE_CAP=""
GPU_MODEL="unknown"
DRIVER="unknown"
POOL="unknown"
TARGET_ARCH="x86_64"
OUTFILE="vla-build.env"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-max)    require_value "$1" "${2:-}"; CUDA_MAX="$2"; shift 2 ;;
    --compute-cap) require_value "$1" "${2:-}"; COMPUTE_CAP="$2"; shift 2 ;;
    --gpu-model)   require_value "$1" "${2:-}"; GPU_MODEL="$2"; shift 2 ;;
    --driver)      require_value "$1" "${2:-}"; DRIVER="$2"; shift 2 ;;
    --pool)        require_value "$1" "${2:-}"; POOL="$2"; shift 2 ;;
    --target-arch) require_value "$1" "${2:-}"; TARGET_ARCH="$2"; shift 2 ;;
    -o|--out)      require_value "$1" "${2:-}"; OUTFILE="$2"; shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *)             usage >&2; die "unknown argument: $1" ;;
  esac
done

[[ -n "$CUDA_MAX" ]] || die "--cuda-max is required"
[[ -n "$COMPUTE_CAP" ]] || die "--compute-cap is required"
[[ "$CUDA_MAX" =~ ^[0-9]+\.[0-9]+$ ]] || die "invalid --cuda-max '$CUDA_MAX'; expected X.Y"
[[ "$COMPUTE_CAP" =~ ^[0-9]+\.[0-9]+$ ]] || die "invalid --compute-cap '$COMPUTE_CAP'; expected X.Y"
[[ "$TARGET_ARCH" == "x86_64" || "$TARGET_ARCH" == "aarch64" ]] || \
  die "unsupported --target-arch '$TARGET_ARCH'; expected x86_64 or aarch64"

SM="sm_${COMPUTE_CAP/./}"
WHEEL_PLATFORM="manylinux_2_28_${TARGET_ARCH}"

# Host Python is used only as a parser/downloader. It is NOT selected as the
# Python version for the future container.
if command -v python3 >/dev/null 2>&1; then
  HOST_PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
  HOST_PYTHON="python"
else
  die "Python is required to query and inspect the official wheel index"
fi

select_cuda_branch() {
  case "$1" in
    13.2|13.1)          echo "cu132" ;;
    13.0)               echo "cu130" ;;
    12.9)               echo "cu129" ;;
    12.8)               echo "cu128" ;;
    12.7|12.6)          echo "cu126" ;;
    12.5|12.4)          echo "cu124" ;;
    12.3|12.2|12.1)     echo "cu121" ;;
    12.0|11.9|11.8)     echo "cu118" ;;
    *) return 1 ;;
  esac
}

TORCH_CUDA_BRANCH=$(select_cuda_branch "$CUDA_MAX") || \
  die "CUDA_MAX=$CUDA_MAX is not present in the verified PyTorch CUDA branch mapping"
INDEX_URL="${PYTORCH_INDEX_ROOT}/${TORCH_CUDA_BRANCH}"

# cu130 -> 13.0, cu128 -> 12.8, cu121 -> 12.1.
CUDA_DIGITS="${TORCH_CUDA_BRANCH#cu}"
CUDA_TOOLKIT_VERSION="${CUDA_DIGITS:0:${#CUDA_DIGITS}-1}.${CUDA_DIGITS: -1}"

version_le() { printf '%s\n%s\n' "$1" "$2" | sort -V -C; }
version_ge() { version_le "$2" "$1"; }

minimum_toolkit_for_compute_cap() {
  case "$1" in
    12.0|10.0) echo "12.8" ;;
    9.0|8.9)   echo "11.8" ;;
    8.6|8.0)   echo "11.0" ;;
    7.5)       echo "10.0" ;;
    7.0)       echo "9.0" ;;
    6.1|6.0)   echo "8.0" ;;
    *)         echo "" ;;
  esac
}

MIN_GPU_TOOLKIT=$(minimum_toolkit_for_compute_cap "$COMPUTE_CAP")
if [[ -n "$MIN_GPU_TOOLKIT" ]] && ! version_ge "$CUDA_TOOLKIT_VERSION" "$MIN_GPU_TOOLKIT"; then
  die "CUDA Toolkit $CUDA_TOOLKIT_VERSION does not support compute capability $COMPUTE_CAP; minimum is $MIN_GPU_TOOLKIT"
fi
if ! version_le "$CUDA_TOOLKIT_VERSION" "$CUDA_MAX"; then
  die "selected CUDA Toolkit $CUDA_TOOLKIT_VERSION exceeds driver CUDA_MAX=$CUDA_MAX"
fi

if [[ "$DRIVER" =~ ^([0-9]+) ]]; then
  DRIVER_MAJOR="${BASH_REMATCH[1]}"
  case "${CUDA_TOOLKIT_VERSION%%.*}" in
    13) MIN_DRIVER_MAJOR=580 ;;
    12) MIN_DRIVER_MAJOR=525 ;;
    11) MIN_DRIVER_MAJOR=450 ;;
    *)  MIN_DRIVER_MAJOR=0 ;;
  esac
  (( DRIVER_MAJOR >= MIN_DRIVER_MAJOR )) || \
    die "driver $DRIVER is too old for CUDA Toolkit $CUDA_TOOLKIT_VERSION; expected branch >= $MIN_DRIVER_MAJOR"
fi

echo "==> Parameters"
echo "    pool              : $POOL"
echo "    GPU               : $GPU_MODEL"
echo "    driver            : $DRIVER"
echo "    CUDA_MAX          : $CUDA_MAX"
echo "    compute capability: $COMPUTE_CAP ($SM)"
echo "    target arch       : $TARGET_ARCH"
echo "    wheel branch      : $TORCH_CUDA_BRANCH"
echo "    CUDA Toolkit      : $CUDA_TOOLKIT_VERSION"
echo "==> Selecting target Python and compatible official wheels..."

# Python policy for VLA/ML containers:
#   3.11 first: mature, broadly supported by ML libraries and CUDA extensions.
#   3.10 second: maximum legacy compatibility.
#   3.12+ only when the preferred ABIs are unavailable.
# Availability is verified against actual official torch and torchvision wheels.
RESOLUTION_JSON=$(
  "$HOST_PYTHON" - "$INDEX_URL" "$WHEEL_PLATFORM" <<'PYCODE'
import email
import html.parser
import json
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from io import BytesIO

index_url = sys.argv[1].rstrip("/")
platform_tag = sys.argv[2]
python_preference = [ "3.12", "3.11", "3.13", "3.10", "3.14"]

class LinkParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)

def fetch_links(package):
    url = f"{index_url}/{package}/"
    req = urllib.request.Request(url, headers={"User-Agent": "vla-version-resolver/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        text = response.read().decode("utf-8", errors="replace")
    parser = LinkParser()
    parser.feed(text)
    result = []
    for href in parser.links:
        absolute = urllib.parse.urljoin(url, href)
        filename = urllib.parse.unquote(urllib.parse.urlparse(absolute).path.rsplit("/", 1)[-1])
        if filename.endswith(".whl"):
            result.append((filename, absolute))
    return result

def stable_version_key(version):
    public = version.split("+", 1)[0]
    if not re.fullmatch(r"\d+\.\d+\.\d+", public):
        return None
    return tuple(int(x) for x in public.split("."))

def wheel_records(package, links):
    records = []
    pattern = re.compile(
        rf"^{re.escape(package)}-(?P<version>.+?)-(?P<abi>cp\d+t?)-(?P<abi2>cp\d+t?)-(?P<platform>[^.]+)\.whl$",
        re.I,
    )
    for filename, url in links:
        match = pattern.match(filename)
        if not match:
            continue
        version = match.group("version")
        key = stable_version_key(version)
        if key is None:
            continue
        records.append({
            "filename": filename,
            "url": url,
            "version": version,
            "version_key": key,
            "abi": match.group("abi"),
            "platform": match.group("platform"),
        })
    return records

def public_version(version):
    return version.split("+", 1)[0]

def inspect_torchvision(url):
    req = urllib.request.Request(url, headers={"User-Agent": "vla-version-resolver/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        wheel_bytes = response.read()
    with zipfile.ZipFile(BytesIO(wheel_bytes)) as wheel:
        names = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise RuntimeError("could not uniquely locate torchvision METADATA")
        message = email.message_from_bytes(wheel.read(names[0]))
    requirements = message.get_all("Requires-Dist", [])
    torch_requirements = [r for r in requirements if re.match(r"^torch(?:\s|\(|==|>=|<=|~=|!=|>|<|$)", r, re.I)]
    for requirement in torch_requirements:
        match = re.match(r"^torch\s*(?:\(\s*)?==\s*([^;\s\)]+)", requirement, re.I)
        if match:
            return public_version(match.group(1)), requirement
    return None, " | ".join(torch_requirements)

torch_records = wheel_records("torch", fetch_links("torch"))
tv_records = wheel_records("torchvision", fetch_links("torchvision"))

attempts = []
for pyver in python_preference:
    abi = "cp" + pyver.replace(".", "")
    torch_candidates = [
        r for r in torch_records
        if r["abi"] == abi and r["platform"] == platform_tag
    ]
    tv_candidates = [
        r for r in tv_records
        if r["abi"] == abi and r["platform"] == platform_tag
    ]
    torch_candidates.sort(key=lambda r: r["version_key"], reverse=True)
    tv_candidates.sort(key=lambda r: r["version_key"], reverse=True)

    if not torch_candidates or not tv_candidates:
        attempts.append(f"Python {pyver}: missing torch or torchvision wheel")
        continue

    # Prefer the latest stable torch available for this ABI, then find the
    # torchvision wheel whose official METADATA pins that exact public version.
    for torch_rec in torch_candidates:
        torch_public = public_version(torch_rec["version"])
        for tv_rec in tv_candidates:
            required_torch, requirement = inspect_torchvision(tv_rec["url"])
            if required_torch == torch_public:
                print(json.dumps({
                    "python_version": pyver,
                    "python_abi": abi,
                    "torch_version": torch_rec["version"],
                    "torch_wheel": torch_rec["filename"],
                    "torchvision_version": tv_rec["version"],
                    "torchvision_wheel": tv_rec["filename"],
                    "compatibility_source": f"official torchvision wheel METADATA: {requirement}",
                    "selection_policy": "VLA/ML compatibility preference 3.11, 3.10, 3.12, 3.13, 3.14; verified official wheels",
                }))
                raise SystemExit(0)
        attempts.append(f"Python {pyver}: no torchvision METADATA matched torch {torch_public}")

raise SystemExit("No compatible wheel set found. " + "; ".join(attempts))
PYCODE
) || die "could not resolve a target Python/torch/torchvision wheel set from $INDEX_URL"

readarray -t RESOLVED < <(
  "$HOST_PYTHON" - "$RESOLUTION_JSON" <<'PYCODE'
import json, sys
v = json.loads(sys.argv[1])
for key in (
    "python_version",
    "python_abi",
    "torch_version",
    "torch_wheel",
    "torchvision_version",
    "torchvision_wheel",
    "compatibility_source",
    "selection_policy",
):
    print(v[key])
PYCODE
)

PYTHON_VERSION="${RESOLVED[0]}"
PYTHON_ABI="${RESOLVED[1]}"

# Red Hat UBI Python image names use a compact version suffix:
# Python 3.12 -> ubi9/python-312, Python 3.11 -> ubi9/python-311.
PYTHON_COMPACT="${PYTHON_VERSION/./}"
UBI_PYTHON_IMAGE="registry.access.redhat.com/ubi9/python-${PYTHON_COMPACT}:latest"

TORCH_VERSION="${RESOLVED[2]}"
TORCH_WHEEL_FILENAME="${RESOLVED[3]}"
TORCHVISION_VERSION="${RESOLVED[4]}"
TORCHVISION_WHEEL_FILENAME="${RESOLVED[5]}"
RESOLUTION_SOURCE="${RESOLVED[6]}"
PYTHON_SELECTION_POLICY="${RESOLVED[7]}"

mkdir -p "$(dirname "$OUTFILE")"
TMP_OUT=$(mktemp "${OUTFILE}.tmp.XXXXXX")
trap 'rm -f "$TMP_OUT"' EXIT

cat > "$TMP_OUT" <<EOF_MANIFEST
# Auto-generated by gen_requirements.sh on $(date -u +%FT%TZ)
# Unified version manifest for a future VLA fine-tuning container build.
# This script resolved versions only; it did not pull or build a container.

GPU_POOL=$(printf '%q' "$POOL")
GPU_MODEL=$(printf '%q' "$GPU_MODEL")
NVIDIA_DRIVER_VERSION=$(printf '%q' "$DRIVER")
GPU_COMPUTE_CAPABILITY=$COMPUTE_CAP
GPU_SM=$SM
TORCH_CUDA_ARCH_LIST=$COMPUTE_CAP

TARGET_OS=linux
TARGET_ARCH=$TARGET_ARCH
WHEEL_PLATFORM=$WHEEL_PLATFORM

CUDA_MAX=$CUDA_MAX
CUDA_TOOLKIT_VERSION=$CUDA_TOOLKIT_VERSION
PYTORCH_CUDA_BRANCH=$TORCH_CUDA_BRANCH
PYTORCH_INDEX_URL=$INDEX_URL

PYTHON_VERSION=$PYTHON_VERSION
PYTHON_ABI=$PYTHON_ABI
UBI_PYTHON_IMAGE=$UBI_PYTHON_IMAGE

TORCH_VERSION=$TORCH_VERSION
TORCH_WHEEL_FILENAME=$(printf '%q' "$TORCH_WHEEL_FILENAME")
TORCHVISION_VERSION=$TORCHVISION_VERSION
TORCHVISION_WHEEL_FILENAME=$(printf '%q' "$TORCHVISION_WHEEL_FILENAME")

# Python selection policy: $PYTHON_SELECTION_POLICY
# Compatibility source: $RESOLUTION_SOURCE
# Official references:
#   $OFFICIAL_LOCAL_URL
#   $OFFICIAL_VERSIONS_URL
#   $OFFICIAL_COMPAT_URL
EOF_MANIFEST

mv "$TMP_OUT" "$OUTFILE"
trap - EXIT

printf '\n==> Generated version manifest: %s\n' "$OUTFILE"
printf '    Python          : %s (%s)\n' "$PYTHON_VERSION" "$PYTHON_ABI"
printf '    UBI Python image: %s\n' "$UBI_PYTHON_IMAGE"
printf '    CUDA Toolkit    : %s\n' "$CUDA_TOOLKIT_VERSION"
printf '    PyTorch branch  : %s\n' "$TORCH_CUDA_BRANCH"
printf '    torch           : %s\n' "$TORCH_VERSION"
printf '    torchvision     : %s\n' "$TORCHVISION_VERSION"
printf '    wheel platform  : %s\n' "$WHEEL_PLATFORM"
printf '    source          : %s\n\n' "$RESOLUTION_SOURCE"
cat "$OUTFILE"
