#!/usr/bin/env bash
#
# Resolve a LeRobot-compatible version manifest for a future VLA fine-tuning container.
# The script resolves versions only. It does not pull or build a container image.
#
# Sources of truth:
#   1. LeRobot pyproject.toml for Python / torch / torchvision constraints and preferred CUDA index.
#   2. Official PyTorch wheel indexes for actual wheel availability.
#   3. torchvision wheel METADATA for the exact torch compatibility relation.
#   4. TorchCodec's own README compatibility table (its wheels do not declare
#      a torch version as a pip Requires-Dist, so this is fetched and parsed
#      instead of guessed from wheel metadata; see
#      https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions).
#   5. PyPI JSON API for the latest nvidia-npp-cu<major> release, matching the
#      CUDA major version actually selected (TorchCodec's CUDA wheel dlopen's
#      libnppicc, which PyTorch/TorchCodec wheels do not bundle themselves).
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
                         NOTE: aarch64 is rejected -- TorchCodec has no Linux
                         ARM wheels (see huggingface.co/docs/lerobot/en/installation)
  --lerobot-ref <ref>    LeRobot Git ref whose pyproject.toml defines constraints
                         Default: main
  --http-timeout <sec>   Per-HTTP-request timeout in seconds (raise this on a
                         slow/high-latency network instead of it looking hung)
                         Default: 20
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
HTTP_TIMEOUT="20"
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
    --http-timeout) require_value "$1" "${2:-}"; HTTP_TIMEOUT="$2"; shift 2 ;;
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
[[ "$HTTP_TIMEOUT" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "invalid --http-timeout '$HTTP_TIMEOUT'; expected a number of seconds"

# Per upstream docs, TorchCodec (what this Containerfile builds FFmpeg for)
# is not available on Linux ARM at all -- LeRobot silently falls back to
# pyav there instead. https://huggingface.co/docs/lerobot/en/installation
# Resolving a TorchCodec wheel for aarch64 would just fail later with a
# confusing "no compatible wheel" error, so fail clearly here instead.
if [[ "$TARGET_ARCH" == "aarch64" ]]; then
  die "TARGET_ARCH=aarch64 is not supported by this build: TorchCodec has no" \
      $'\n      Linux ARM wheels (LeRobot falls back to pyav there instead of' \
      $'\n      TorchCodec+FFmpeg). This Containerfile/resolver is built around' \
      $'\n      the TorchCodec+system-FFmpeg path, so it only targets x86_64.' \
      $'\n      See https://huggingface.co/docs/lerobot/en/installation'
fi

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
    "$TARGET_ARCH" \
    "$HTTP_TIMEOUT" <<'PYCODE'
import email
import html.parser
import http.client
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from functools import lru_cache
from io import BytesIO

index_root, lerobot_repo_root, lerobot_ref, cuda_max, compute_cap, platform_tag, target_arch, http_timeout_arg = sys.argv[1:]

USER_AGENT = "lerobot-build-manifest-resolver/2.0"
HTTP_TIMEOUT = float(http_timeout_arg)
_START = time.monotonic()

def log(message):
    """Progress line to stderr (not captured by the caller's $(...) which
    only reads stdout), so it's visible live instead of it looking hung."""
    print(f"[{time.monotonic() - _START:6.1f}s] {message}", file=sys.stderr, flush=True)

def fetch_text(url, timeout=None):
    log(f"GET {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout or HTTP_TIMEOUT) as response:
        return response.read().decode("utf-8", errors="replace")

class _RangeReadError(RuntimeError):
    pass

class HTTPRangeFile:
    """Minimal seekable, read-only file-like object backed by HTTP Range
    requests, reusing one keep-alive HTTPS connection for all reads of a
    given wheel. Lets zipfile.ZipFile read just the central directory and
    one member (METADATA) instead of downloading an entire multi-hundred-MB
    torch/torchvision/torchcodec wheel just to inspect its metadata -- and
    without paying a fresh TCP+TLS handshake for every one of the handful of
    small reads zipfile needs to do that."""

    def __init__(self, url, timeout=None):
        self.url = url
        self.timeout = timeout or HTTP_TIMEOUT
        self._pos = 0
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https":
            raise _RangeReadError(f"only https:// URLs are supported for range reads: {url}")
        self._host = parsed.netloc
        self._path = urllib.parse.urlunsplit(("", "", parsed.path, parsed.query, ""))
        self._conn = http.client.HTTPSConnection(self._host, timeout=self.timeout)
        self.length = self._probe_length()

    def _do_request(self, range_header):
        headers = {"User-Agent": USER_AGENT, "Range": range_header}
        try:
            self._conn.request("GET", self._path, headers=headers)
            response = self._conn.getresponse()
            body = response.read()
        except (http.client.HTTPException, OSError):
            # Keep-alive connection may have been dropped by the server or a
            # proxy in between reads; reconnect once and retry before giving up.
            log(f"  connection to {self._host} dropped; reconnecting")
            self._conn.close()
            self._conn = http.client.HTTPSConnection(self._host, timeout=self.timeout)
            self._conn.request("GET", self._path, headers=headers)
            response = self._conn.getresponse()
            body = response.read()
        if response.status in (301, 302, 303, 307, 308):
            raise _RangeReadError(f"redirect not supported for range reads: {self.url}")
        return response.status, response.getheader("Content-Range"), body

    def _probe_length(self):
        status, content_range, _ = self._do_request("bytes=0-0")
        if status != 206 or not content_range or "/" not in content_range:
            raise _RangeReadError(f"server does not support HTTP range requests for {self.url}")
        return int(content_range.rsplit("/", 1)[-1])

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=0):
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self.length + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        return self._pos

    def read(self, size=-1):
        end = self.length if size is None or size < 0 else min(self.length, self._pos + size)
        if end <= self._pos:
            return b""
        _, _, body = self._do_request(f"bytes={self._pos}-{end - 1}")
        self._pos += len(body)
        return body

    def close(self):
        self._conn.close()

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

commit_url = f"https://api.github.com/repos/huggingface/lerobot/commits/{urllib.parse.quote(release_ref, safe='')}"
log(f"GET {commit_url}")
commit_request = urllib.request.Request(
    commit_url,
    headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/vnd.github+json",
    },
)
with urllib.request.urlopen(commit_request, timeout=HTTP_TIMEOUT) as response:
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
    log(f"listing wheel index: {url}")
    parser = LinkParser()
    parser.feed(fetch_text(url))
    result = []
    for href in parser.links:
        absolute = urllib.parse.urljoin(url, href)
        filename = urllib.parse.unquote(urllib.parse.urlparse(absolute).path.rsplit("/", 1)[-1])
        if filename.endswith(".whl"):
            result.append((filename, absolute))
    log(f"  -> {len(result)} wheel(s) listed for {package}")
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
def fetch_wheel_metadata(url):
    filename = url.rsplit("/", 1)[-1]
    log(f"inspecting metadata (range-read): {filename}")
    try:
        remote = HTTPRangeFile(url)
        try:
            with zipfile.ZipFile(remote) as wheel:
                metadata_names = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
                if len(metadata_names) != 1:
                    raise RuntimeError(f"could not uniquely locate METADATA in {url}")
                return email.message_from_bytes(wheel.read(metadata_names[0]))
        finally:
            remote.close()
    except _RangeReadError as exc:
        # Server doesn't support Range requests: fall back to a full
        # download rather than failing outright.
        log(f"  range reads unsupported ({exc}); falling back to a full download of {filename}")
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=max(HTTP_TIMEOUT, 90)) as response:
            wheel_bytes = response.read()
        with zipfile.ZipFile(BytesIO(wheel_bytes)) as wheel:
            metadata_names = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata_names) != 1:
                raise RuntimeError(f"could not uniquely locate METADATA in {url}")
            return email.message_from_bytes(wheel.read(metadata_names[0]))

def torch_requirement_constraint(message):
    """Turn a wheel's 'Requires-Dist: torch...' entries into a `satisfies()` spec.

    Handles both an exact pin (torch==2.11.0) and open ranges
    (torch>=2.11, torch (>=2.11,<2.12)), which is how recent torchcodec
    releases declare their torch compatibility.
    """
    requirements = message.get_all("Requires-Dist", [])
    torch_requirements = [
        item
        for item in requirements
        if re.match(r"^torch(?:\s|\(|==|>=|<=|~=|!=|>|<|;|$)", item, re.I)
    ]
    for requirement in torch_requirements:
        match = re.match(r"^torch\s*(?:\(\s*)?==\s*([^;\s\)]+)", requirement, re.I)
        if match:
            return f"=={public_version(match.group(1))}", requirement
    clauses = []
    for requirement in torch_requirements:
        body = re.sub(r"^torch\s*", "", requirement, flags=re.I)
        body = re.sub(r"^\((.*)\)$", r"\1", body.strip())
        body = body.split(";", 1)[0]
        for clause in body.split(","):
            clause = clause.strip()
            if re.fullmatch(r"(>=|<=|==|!=|>|<|~=)\s*[0-9]+(?:\.[0-9]+){0,2}", clause):
                clauses.append(clause)
    if clauses:
        return ", ".join(clauses), " | ".join(torch_requirements)
    return None, " | ".join(torch_requirements)

def inspect_torchvision(url):
    constraint, requirement = torch_requirement_constraint(fetch_wheel_metadata(url))
    if constraint and constraint.startswith("=="):
        return constraint[2:], requirement
    return None, requirement

@lru_cache(maxsize=None)
def fetch_torchcodec_compat_table():
    """TorchCodec wheels do not declare their torch version as a pip
    Requires-Dist (confirmed empirically: every torchcodec wheel across
    every branch has zero parseable 'torch' requirements in its METADATA).
    Compatibility is only documented in torchcodec's own README table:
    https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions
    Fetch and parse that instead of guessing from wheel metadata."""
    log("fetching TorchCodec/torch compatibility table (meta-pytorch/torchcodec README)")
    text = fetch_text("https://raw.githubusercontent.com/meta-pytorch/torchcodec/main/README.md")
    table = {}
    for match in re.finditer(r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|", text, re.M):
        tc_minor, torch_spec = match.group(1).strip(), match.group(2).strip()
        if not re.fullmatch(r"[0-9]+\.[0-9]+", tc_minor):
            continue  # skips the header row and the 'main / nightly' row
        table[tc_minor] = torch_spec
    if not table:
        raise RuntimeError("could not parse any rows from the TorchCodec compatibility table")
    log(f"  parsed {len(table)} torchcodec->torch compatibility rows")
    return table

def torch_spec_to_constraint(torch_spec):
    """Turn a README table cell like '>=2.11' or a bare '2.11' (meaning that
    exact minor, any patch) into a `satisfies()`-compatible spec string."""
    match = re.match(r"^(>=|<=|==|~=|!=|>|<)\s*([0-9]+(?:\.[0-9]+){0,2})$", torch_spec)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)", torch_spec)
    if match:
        major, minor = int(match.group(1)), int(match.group(2))
        return f">={major}.{minor}, <{major}.{minor + 1}"
    return None

def resolve_torchcodec(index_url, abi, platform_tag, torch_public):
    """Return (version, filename) for the newest torchcodec wheel on this
    CUDA branch/ABI/platform whose *documented* (README table) torch
    requirement is satisfied by the exact resolved torch version, or
    (None, reason) if none fit."""
    try:
        tc_records = wheel_records("torchcodec", fetch_links(index_url, "torchcodec"))
    except Exception as exc:
        return None, f"torchcodec index unavailable at {index_url} ({exc})"
    candidates = [
        record
        for record in tc_records
        if record["abi"] == abi and record["platform"] == platform_tag
    ]
    candidates.sort(key=lambda record: record["version_key"], reverse=True)
    if not candidates:
        return None, f"no torchcodec wheels for {abi}/{platform_tag} on {index_url}"
    try:
        compat_table = fetch_torchcodec_compat_table()
    except Exception as exc:
        return None, f"could not load TorchCodec compatibility table ({exc})"
    log(f"  torchcodec: {len(candidates)} candidate(s) for {abi}/{platform_tag}, "
        f"checking against torch=={torch_public}")
    checked = []
    for record in candidates:
        public = public_version(record["version"])
        minor_key = ".".join(public.split(".")[:2])
        torch_spec = compat_table.get(minor_key)
        if torch_spec is None:
            checked.append(f"{record['version']}: no compatibility-table entry for torchcodec {minor_key}")
            continue
        constraint = torch_spec_to_constraint(torch_spec)
        if constraint is None:
            checked.append(f"{record['version']}: unrecognized compatibility-table entry ({torch_spec!r})")
            continue
        try:
            ok = satisfies(torch_public, constraint)
        except ValueError as exc:
            log(f"    torchcodec {record['version']}: {exc}")
            checked.append(f"{record['version']}: {exc}")
            continue
        if ok:
            log(f"    torchcodec {record['version']}: OK (table says torch{torch_spec})")
            return record["version"], record["filename"]
        log(f"    torchcodec {record['version']}: requires torch{constraint}, have {torch_public} -- skipping")
        checked.append(f"{record['version']}: requires torch{constraint}, have {torch_public}")
    return None, "; ".join(checked)

attempts = []
for branch in branch_candidates:
    index_url = f"{index_root.rstrip('/')}/{branch}"
    log(f"trying CUDA branch {branch} ({index_url})")
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

        log(f"  Python {pyver} ({abi}): {len(torch_candidates)} torch candidate(s), "
            f"{len(tv_candidates)} torchvision candidate(s)")

        for torch_record in torch_candidates:
            torch_public = public_version(torch_record["version"])
            for tv_record in tv_candidates:
                log(f"  checking torch=={torch_record['version']} vs torchvision=={tv_record['version']}")
                required_torch, requirement = inspect_torchvision(tv_record["url"])
                if required_torch != torch_public:
                    log(f"    torchvision {tv_record['version']} requires torch=={required_torch}, "
                        f"not {torch_public} -- skipping")
                    continue
                log(f"    torchvision {tv_record['version']} matches torch=={torch_public}; "
                    f"resolving torchcodec")
                torchcodec_version, torchcodec_info = resolve_torchcodec(
                    index_url, abi, platform_tag, torch_public
                )
                if torchcodec_version is None:
                    attempts.append(
                        f"{branch}/Python {pyver}/torch {torch_record['version']}: "
                        f"no compatible torchcodec wheel ({torchcodec_info})"
                    )
                    continue
                log(f"RESOLVED: branch={branch} python={pyver} torch={torch_record['version']} "
                    f"torchvision={tv_record['version']} torchcodec={torchcodec_version}")
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
                            "torchcodec_version": torchcodec_version,
                            "torchcodec_wheel": torchcodec_info,
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
    "torchcodec_version",
    "torchcodec_wheel",
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
TORCHCODEC_VERSION="${RESOLVED[11]}"
TORCHCODEC_WHEEL_FILENAME="${RESOLVED[12]}"

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

# TorchCodec's CUDA wheel dlopen's libnppicc, which is not bundled with the
# torch/torchcodec wheels themselves. Resolve the matching NVIDIA NPP wheel
# for the CUDA major version actually selected above (cu128 -> cu12, etc.),
# rather than assuming cu12 unconditionally.
CUDA_MAJOR="${CUDA_TOOLKIT_VERSION%%.*}"
NVIDIA_NPP_PACKAGE="nvidia-npp-cu${CUDA_MAJOR}"

echo "==> Resolving NVIDIA NPP runtime package (${NVIDIA_NPP_PACKAGE})..."
NVIDIA_NPP_VERSION=$(
  "$HOST_PYTHON" - "$NVIDIA_NPP_PACKAGE" "$HTTP_TIMEOUT" <<'PYCODE' || die "could not resolve an NVIDIA NPP wheel; if this GPU/driver selects a CUDA major version without a published nvidia-npp-cu<major> wheel yet, pass --cuda-max for an older branch (e.g. 12.x) until one is published"
import json
import sys
import urllib.request

package, http_timeout_arg = sys.argv[1:]
timeout = float(http_timeout_arg)
url = f"https://pypi.org/pypi/{package}/json"
print(f"[NPP] GET {url}", file=sys.stderr, flush=True)
request = urllib.request.Request(
    url,
    headers={"User-Agent": "lerobot-build-manifest-resolver/2.0"},
)
with urllib.request.urlopen(request, timeout=timeout) as response:
    data = json.load(response)

releases = {
    version: files
    for version, files in data.get("releases", {}).items()
    if files and not any(f.get("yanked") for f in files)
}
if not releases:
    raise SystemExit(f"no published releases found for {package}")

def version_key(value):
    public = value.split("+", 1)[0]
    parts = public.split(".")
    return tuple(int(p) for p in parts if p.isdigit())

latest = sorted(releases, key=version_key)[-1]
print(latest)
PYCODE
)

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

TORCHCODEC_VERSION=$TORCHCODEC_VERSION
TORCHCODEC_WHEEL_FILENAME=$(printf '%q' "$TORCHCODEC_WHEEL_FILENAME")

NVIDIA_NPP_PACKAGE=$NVIDIA_NPP_PACKAGE
NVIDIA_NPP_VERSION=$NVIDIA_NPP_VERSION

# System FFmpeg is compiled from source in the Containerfile's builder stage
# (not installed from a package repo), so its version is a build default
# rather than something resolved here. Override by passing a different
# --build-arg FFMPEG_VERSION at build time if needed.
FFMPEG_VERSION=7.1.1

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
printf '    torchcodec       : %s\n' "$TORCHCODEC_VERSION"
printf '    NVIDIA NPP       : %s==%s\n' "$NVIDIA_NPP_PACKAGE" "$NVIDIA_NPP_VERSION"
printf '    wheel platform   : %s\n' "$WHEEL_PLATFORM"
printf '\n'
cat "$OUTFILE"
