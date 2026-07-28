#!/usr/bin/env bash
#
# Resolve a LeRobot-compatible version manifest for a future VLA fine-tuning container.
# The script resolves versions only. It does not pull or build a container image.
#
# Sources of truth:
#   1. LeRobot pyproject.toml for Python / torch / torchvision constraints and preferred CUDA index.
#   2. Official PyTorch wheel indexes for actual wheel availability.
#   3. torchvision wheel METADATA for the exact torch compatibility relation.
#
set -euo pipefail

readonly PYTORCH_INDEX_ROOT="https://download.pytorch.org/whl"
readonly LEROBOT_REPO_ROOT="https://raw.githubusercontent.com/huggingface/lerobot"
readonly LEROBOT_API_ROOT="https://api.github.com/repos/huggingface/lerobot"
readonly OFFICIAL_LEROBOT_URL="https://github.com/huggingface/lerobot/blob/main/pyproject.toml"
readonly OFFICIAL_PYTORCH_URL="https://pytorch.org/get-started/locally/"
readonly OFFICIAL_VISION_URL="https://github.com/pytorch/vision#installation"

usage() {
  cat <<'USAGE'
Usage:
  ./resolve_build_manifest.sh --cuda-max <X.Y> --compute-cap <X.Y> [options]

Required:
  --cuda-max <X.Y>       Maximum CUDA version supported by the target node driver
  --compute-cap <X.Y>    GPU compute capability, for example 8.9

Optional:
  --gpu-model <name>     GPU model written to the manifest
  --driver <version>     NVIDIA driver version written to the manifest
  --pool <label>         GPU pool/node label written to the manifest
  --target-arch <arch>   Target container CPU architecture: x86_64 or aarch64
                         Default: x86_64
  --lerobot-ref <ref>    LeRobot Git ref whose pyproject.toml defines constraints
                         Default: main
  -o, --out <file>       Output manifest
                         Default: versions.env

Example:
  chmod +x resolve_build_manifest.sh
  ./resolve_build_manifest.sh --cuda-max 13.0 --compute-cap 8.9 \
      --gpu-model "NVIDIA L40S" --driver 580.126.20 --pool xlarge \
      --lerobot-ref main -o versions.env
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
LEROBOT_REF="main"
OUTFILE="versions.env"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda-max)    require_value "$1" "${2:-}"; CUDA_MAX="$2"; shift 2 ;;
    --compute-cap) require_value "$1" "${2:-}"; COMPUTE_CAP="$2"; shift 2 ;;
    --gpu-model)   require_value "$1" "${2:-}"; GPU_MODEL="$2"; shift 2 ;;
    --driver)      require_value "$1" "${2:-}"; DRIVER="$2"; shift 2 ;;
    --pool)        require_value "$1" "${2:-}"; POOL="$2"; shift 2 ;;
    --target-arch) require_value "$1" "${2:-}"; TARGET_ARCH="$2"; shift 2 ;;
    --lerobot-ref) require_value "$1" "${2:-}"; LEROBOT_REF="$2"; shift 2 ;;
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

if command -v python3 >/dev/null 2>&1; then
  HOST_PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
  HOST_PYTHON="python"
else
  die "Python is required to inspect LeRobot metadata and PyTorch wheel indexes"
fi

SM="sm_${COMPUTE_CAP/./}"
WHEEL_PLATFORM="manylinux_2_28_${TARGET_ARCH}"

echo "==> Target hardware"
echo "    pool              : $POOL"
echo "    GPU               : $GPU_MODEL"
echo "    driver            : $DRIVER"
echo "    CUDA_MAX          : $CUDA_MAX"
echo "    compute capability: $COMPUTE_CAP ($SM)"
echo "    target arch       : $TARGET_ARCH"
echo "==> Resolving the published LeRobot release and its constraints from PyPI/GitHub..."
echo "==> Resolving a LeRobot-compatible Python / CUDA / torch / torchvision set..."

RESOLUTION_JSON=$(
  "$HOST_PYTHON" - \
    "$PYTORCH_INDEX_ROOT" \
    "$LEROBOT_REPO_ROOT" \
    "$LEROBOT_REF" \
    "$CUDA_MAX" \
    "$COMPUTE_CAP" \
    "$WHEEL_PLATFORM" \
    "$TARGET_ARCH" <<'PYCODE'
import email
import html.parser
import json
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from functools import lru_cache
from io import BytesIO

index_root, lerobot_repo_root, lerobot_ref, cuda_max, compute_cap, platform_tag, target_arch = sys.argv[1:]

def fetch_text(url, timeout=30):
    request = urllib.request.Request(url, headers={"User-Agent": "lerobot-build-manifest-resolver/2.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")

# Installation is performed from PyPI, so the source of truth must be a
# version that is actually published there. For main/latest, use the latest
# published stable release. A numeric --lerobot-ref selects that exact release.
pypi = json.loads(fetch_text("https://pypi.org/pypi/lerobot/json"))
if lerobot_ref in {"main", "latest"}:
    lerobot_version = pypi["info"]["version"]
else:
    candidate = lerobot_ref[1:] if lerobot_ref.startswith("v") else lerobot_ref
    if candidate not in pypi.get("releases", {}) or not pypi["releases"][candidate]:
        raise RuntimeError(
            f"LeRobot {candidate} is not published on PyPI; "
            "use main/latest or an exact published version"
        )
    lerobot_version = candidate

# Read constraints from the source tree corresponding to that published release.
pyproject = None
release_ref = None
for candidate_ref in (f"v{lerobot_version}", lerobot_version):
    try:
        pyproject = fetch_text(f"{lerobot_repo_root}/{candidate_ref}/pyproject.toml")
        release_ref = candidate_ref
        break
    except Exception:
        pass
if pyproject is None:
    raise RuntimeError(f"could not fetch pyproject.toml for published LeRobot {lerobot_version}")

def first_match(pattern, text, description):
    match = re.search(pattern, text, re.M | re.S)
    if not match:
        raise RuntimeError(f"could not parse {description} from LeRobot pyproject.toml")
    return match.group(1).strip()

python_constraint = first_match(
    r'^\s*requires-python\s*=\s*"([^"]+)"',
    pyproject,
    "requires-python",
)
torch_constraint = first_match(
    r'^\s*"torch([^"]*)"\s*,?\s*$',
    pyproject,
    "torch dependency",
)
torchvision_constraint = first_match(
    r'^\s*"torchvision([^"]*)"\s*,?\s*$',
    pyproject,
    "torchvision dependency",
)

# Prefer the CUDA index explicitly configured by LeRobot's uv section.
uv_indexes = re.findall(
    r'\[\[tool\.uv\.index\]\](.*?)(?=^\s*\[\[|^\s*\[tool\.|\Z)',
    pyproject,
    re.M | re.S,
)
preferred_branch = None
for block in uv_indexes:
    url_match = re.search(r'^\s*url\s*=\s*"https://download\.pytorch\.org/whl/(cu\d+)"', block, re.M)
    if url_match:
        preferred_branch = url_match.group(1)
        break

commit_request = urllib.request.Request(
    f"https://api.github.com/repos/huggingface/lerobot/commits/{urllib.parse.quote(release_ref, safe='')}",
    headers={
        "User-Agent": "lerobot-build-manifest-resolver/2.0",
        "Accept": "application/vnd.github+json",
    },
)
with urllib.request.urlopen(commit_request, timeout=30) as response:
    lerobot_commit = json.load(response)["sha"]

def version_tuple(value):
    public = value.split("+", 1)[0]
    parts = public.split(".")
    if not all(part.isdigit() for part in parts):
        raise ValueError(f"unsupported version format: {value}")
    return tuple(int(part) for part in parts)

def normalize_tuple(value, width=3):
    parts = list(version_tuple(value))
    return tuple((parts + [0] * width)[:width])

def satisfies(version, spec):
    current = normalize_tuple(version)
    spec = spec.strip()
    if not spec:
        return True
    for clause in spec.split(","):
        clause = clause.strip()
        match = re.fullmatch(r"(>=|<=|==|!=|>|<|~=)\s*([0-9]+(?:\.[0-9]+){0,2})", clause)
        if not match:
            raise ValueError(f"unsupported constraint clause: {clause!r}")
        operator, expected_text = match.groups()
        expected = normalize_tuple(expected_text)
        if operator == ">=" and not (current >= expected):
            return False
        if operator == "<=" and not (current <= expected):
            return False
        if operator == ">" and not (current > expected):
            return False
        if operator == "<" and not (current < expected):
            return False
        if operator == "==" and not (current == expected):
            return False
        if operator == "!=" and not (current != expected):
            return False
        if operator == "~=":
            # PEP 440 compatible-release approximation for X.Y[.Z].
            lower_ok = current >= expected
            raw_parts = expected_text.split(".")
            if len(raw_parts) >= 3:
                upper = (expected[0], expected[1] + 1, 0)
            else:
                upper = (expected[0] + 1, 0, 0)
            if not (lower_ok and current < upper):
                return False
    return True

# Candidate Python versions come from LeRobot classifiers, then are filtered by requires-python.
classifier_versions = re.findall(r'"Programming Language :: Python :: ([0-9]+\.[0-9]+)"', pyproject)
python_candidates = []
for value in classifier_versions + ["3.12", "3.13", "3.14", "3.11", "3.10"]:
    if value not in python_candidates and satisfies(value, python_constraint):
        python_candidates.append(value)

if not python_candidates:
    raise RuntimeError(f"no candidate Python version satisfies LeRobot constraint {python_constraint}")

branch_to_toolkit = {
    "cu132": "13.2",
    "cu130": "13.0",
    "cu129": "12.9",
    "cu128": "12.8",
    "cu126": "12.6",
    "cu124": "12.4",
    "cu121": "12.1",
    "cu118": "11.8",
}

def version_le(left, right):
    return normalize_tuple(left) <= normalize_tuple(right)

def minimum_toolkit_for_capability(capability):
    mapping = {
        "12.0": "12.8",
        "10.0": "12.8",
        "9.0": "11.8",
        "8.9": "11.8",
        "8.6": "11.0",
        "8.0": "11.0",
        "7.5": "10.0",
        "7.0": "9.0",
        "6.1": "8.0",
        "6.0": "8.0",
    }
    return mapping.get(capability)

minimum_toolkit = minimum_toolkit_for_capability(compute_cap)

valid_branches = []
for branch, toolkit in branch_to_toolkit.items():
    if not version_le(toolkit, cuda_max):
        continue
    if minimum_toolkit and not version_le(minimum_toolkit, toolkit):
        continue
    valid_branches.append(branch)

# LeRobot's own configured PyTorch index wins. Other valid branches are fallbacks,
# newest first, but only if actual compatible wheels exist.
valid_branches.sort(key=lambda branch: normalize_tuple(branch_to_toolkit[branch]), reverse=True)
branch_candidates = []
if preferred_branch in valid_branches:
    branch_candidates.append(preferred_branch)
for branch in valid_branches:
    if branch not in branch_candidates:
        branch_candidates.append(branch)

if not branch_candidates:
    raise RuntimeError(
        f"no verified PyTorch CUDA branch is compatible with CUDA_MAX={cuda_max} "
        f"and compute capability {compute_cap}"
    )

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

@lru_cache(maxsize=None)
def fetch_links(index_url, package):
    url = f"{index_url.rstrip('/')}/{package}/"
    parser = LinkParser()
    parser.feed(fetch_text(url))
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
    return normalize_tuple(public)

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
        records.append(
            {
                "filename": filename,
                "url": url,
                "version": version,
                "version_key": key,
                "abi": match.group("abi"),
                "platform": match.group("platform"),
            }
        )
    return records

def public_version(version):
    return version.split("+", 1)[0]

@lru_cache(maxsize=None)
def inspect_torchvision(url):
    request = urllib.request.Request(url, headers={"User-Agent": "lerobot-build-manifest-resolver/2.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        wheel_bytes = response.read()
    with zipfile.ZipFile(BytesIO(wheel_bytes)) as wheel:
        metadata_names = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise RuntimeError("could not uniquely locate torchvision METADATA")
        message = email.message_from_bytes(wheel.read(metadata_names[0]))
    requirements = message.get_all("Requires-Dist", [])
    torch_requirements = [
        item
        for item in requirements
        if re.match(r"^torch(?:\s|\(|==|>=|<=|~=|!=|>|<|$)", item, re.I)
    ]
    for requirement in torch_requirements:
        match = re.match(r"^torch\s*(?:\(\s*)?==\s*([^;\s\)]+)", requirement, re.I)
        if match:
            return public_version(match.group(1)), requirement
    return None, " | ".join(torch_requirements)

attempts = []
for branch in branch_candidates:
    index_url = f"{index_root.rstrip('/')}/{branch}"
    try:
        torch_records = wheel_records("torch", fetch_links(index_url, "torch"))
        tv_records = wheel_records("torchvision", fetch_links(index_url, "torchvision"))
    except Exception as exc:
        attempts.append(f"{branch}: index unavailable ({exc})")
        continue

    for pyver in python_candidates:
        abi = "cp" + pyver.replace(".", "")
        torch_candidates = [
            record
            for record in torch_records
            if record["abi"] == abi
            and record["platform"] == platform_tag
            and satisfies(public_version(record["version"]), torch_constraint)
        ]
        tv_candidates = [
            record
            for record in tv_records
            if record["abi"] == abi
            and record["platform"] == platform_tag
            and satisfies(public_version(record["version"]), torchvision_constraint)
        ]
        torch_candidates.sort(key=lambda record: record["version_key"], reverse=True)
        tv_candidates.sort(key=lambda record: record["version_key"], reverse=True)

        if not torch_candidates or not tv_candidates:
            attempts.append(
                f"{branch}/Python {pyver}: no wheels satisfying "
                f"torch{torch_constraint} and torchvision{torchvision_constraint}"
            )
            continue

        for torch_record in torch_candidates:
            torch_public = public_version(torch_record["version"])
            for tv_record in tv_candidates:
                required_torch, requirement = inspect_torchvision(tv_record["url"])
                if required_torch != torch_public:
                    continue
                print(
                    json.dumps(
                        {
                            "lerobot_version": lerobot_version,
                            "lerobot_commit": lerobot_commit,
                            "python_version": pyver,
                            "python_abi": abi,
                            "torch_cuda_branch": branch,
                            "cuda_toolkit_version": branch_to_toolkit[branch],
                            "pytorch_index_url": index_url,
                            "torch_version": torch_record["version"],
                            "torch_wheel": torch_record["filename"],
                            "torchvision_version": tv_record["version"],
                            "torchvision_wheel": tv_record["filename"],
                        }
                    )
                )
                raise SystemExit(0)

raise SystemExit("No LeRobot-compatible wheel set found. " + "; ".join(attempts))
PYCODE
) || die "could not resolve a LeRobot-compatible build manifest"

readarray -t RESOLVED < <(
  "$HOST_PYTHON" - "$RESOLUTION_JSON" <<'PYCODE'
import json
import sys

value = json.loads(sys.argv[1])
for key in (
    "lerobot_version",
    "lerobot_commit",
    "python_version",
    "python_abi",
    "torch_cuda_branch",
    "cuda_toolkit_version",
    "pytorch_index_url",
    "torch_version",
    "torch_wheel",
    "torchvision_version",
    "torchvision_wheel",
):
    print(value[key])
PYCODE
)

LEROBOT_VERSION="${RESOLVED[0]}"
LEROBOT_COMMIT_SHA="${RESOLVED[1]}"
PYTHON_VERSION="${RESOLVED[2]}"
PYTHON_ABI="${RESOLVED[3]}"
TORCH_CUDA_BRANCH="${RESOLVED[4]}"
CUDA_TOOLKIT_VERSION="${RESOLVED[5]}"
INDEX_URL="${RESOLVED[6]}"
TORCH_VERSION="${RESOLVED[7]}"
TORCH_WHEEL_FILENAME="${RESOLVED[8]}"
TORCHVISION_VERSION="${RESOLVED[9]}"
TORCHVISION_WHEEL_FILENAME="${RESOLVED[10]}"

# Validate the supplied driver against the CUDA Toolkit selected after applying
# LeRobot constraints.
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

PYTHON_COMPACT="${PYTHON_VERSION/./}"
UBI_PYTHON_IMAGE="registry.access.redhat.com/ubi9/python-${PYTHON_COMPACT}:latest"

mkdir -p "$(dirname "$OUTFILE")"
TMP_OUT=$(mktemp "${OUTFILE}.tmp.XXXXXX")
trap 'rm -f "$TMP_OUT"' EXIT

cat > "$TMP_OUT" <<EOF_MANIFEST
# Auto-generated by resolve_build_manifest.sh on $(date -u +%FT%TZ)
# Unified LeRobot-compatible version manifest for a future VLA fine-tuning container.
# The resolver did not pull or build a container image.

GPU_POOL=$(printf '%q' "$POOL")
GPU_MODEL=$(printf '%q' "$GPU_MODEL")
NVIDIA_DRIVER_VERSION=$(printf '%q' "$DRIVER")
GPU_COMPUTE_CAPABILITY=$COMPUTE_CAP
GPU_SM=$SM
TORCH_CUDA_ARCH_LIST=$COMPUTE_CAP

TARGET_OS=linux
TARGET_ARCH=$TARGET_ARCH
WHEEL_PLATFORM=$WHEEL_PLATFORM

LEROBOT_VERSION=$LEROBOT_VERSION
LEROBOT_COMMIT_SHA=$LEROBOT_COMMIT_SHA

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

EOF_MANIFEST

mv "$TMP_OUT" "$OUTFILE"
trap - EXIT

printf '\n==> Generated LeRobot-compatible manifest: %s\n' "$OUTFILE"
printf '    LeRobot version  : %s\n' "$LEROBOT_VERSION"
printf '    LeRobot commit   : %s\n' "$LEROBOT_COMMIT_SHA"
printf '    Python           : %s (%s)\n' "$PYTHON_VERSION" "$PYTHON_ABI"
printf '    UBI image        : %s\n' "$UBI_PYTHON_IMAGE"
printf '    CUDA Toolkit     : %s\n' "$CUDA_TOOLKIT_VERSION"
printf '    PyTorch branch   : %s\n' "$TORCH_CUDA_BRANCH"
printf '    torch            : %s\n' "$TORCH_VERSION"
printf '    torchvision      : %s\n' "$TORCHVISION_VERSION"
printf '    wheel platform   : %s\n' "$WHEEL_PLATFORM"
printf '\n'
cat "$OUTFILE"
