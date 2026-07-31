#!/usr/bin/env bash
#
# pvc-download.sh — download files/folders from a PVC via a temporary pod,
# using real rsync (over "oc exec" as the transport, like -e ssh).
#
# This gives you both:
#   - Resume support (interrupted transfers pick up where they left off)
#   - A single overall progress bar (--info=progress2): total %, speed, ETA
#
# The pod is created once and reused across runs.
#
# Requires "rsync" installed locally (already present on most Linux distros;
# if not: sudo apt install rsync).
#
# Usage:
#   ./pvc-download.sh <source_path_in_pvc> <local_target_dir>
#
# Example:
#   ./pvc-download.sh /mnt/models/lerobot/pi05_base/20260731T130734Z ./finetuned/pi05_base_20260731T130734Z
#
# Extra commands:
#   ./pvc-download.sh --cleanup     delete the temporary helper pod
#   ./pvc-download.sh --status      check the pod status

set -euo pipefail

PVC_NAME="pvc-finetuned-models"
POD_NAME="rsync-helper"
MOUNT_PATH="/mnt/models"
IMAGE="instrumentisto/rsync-ssh"

manifest() {
  cat <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${POD_NAME}
spec:
  containers:
  - name: ${POD_NAME}
    image: ${IMAGE}
    command: ["sh", "-c", "sleep infinity"]
    volumeMounts:
    - name: models
      mountPath: ${MOUNT_PATH}
  volumes:
  - name: models
    persistentVolumeClaim:
      claimName: ${PVC_NAME}
EOF
}

ensure_pod() {
  if oc get pod "${POD_NAME}" >/dev/null 2>&1; then
    echo "[info] Pod ${POD_NAME} already exists."
  else
    echo "[info] Creating pod ${POD_NAME}..."
    manifest | oc apply -f -
  fi

  echo "[info] Waiting for pod to become ready..."
  oc wait --for=condition=Ready "pod/${POD_NAME}" --timeout=120s
}

do_cleanup() {
  echo "[info] Deleting pod ${POD_NAME}..."
  oc delete pod "${POD_NAME}" --ignore-not-found
  echo "[info] Done."
}

do_status() {
  oc get pod "${POD_NAME}" 2>/dev/null || echo "[info] Pod ${POD_NAME} not found."
}

verify_download() {
  local source="$1"
  local target="$2"

  echo "[info] Verifying copy (size and file count)..."

  local remote_size local_size remote_count local_count
  remote_size="$(oc exec "${POD_NAME}" -- sh -c "find '${source%/}' -type f -exec stat -c%s {} \\; | awk '{s+=\$1} END {print s+0}'" 2>/dev/null)" || remote_size=""
  local_size="$(find "${target%/}" -type f -exec stat -c%s {} \; 2>/dev/null | awk '{s+=$1} END {print s+0}')" || local_size=""
  remote_count="$(oc exec "${POD_NAME}" -- find "${source%/}" -type f 2>/dev/null | wc -l)" || remote_count=""
  local_count="$(find "${target%/}" -type f 2>/dev/null | wc -l)" || local_count=""

  if [[ -z "${remote_size}" || -z "${local_size}" || -z "${remote_count}" || -z "${local_count}" ]]; then
    echo "[warn] Could not fully verify (couldn't read one of the values). Check manually if needed."
    return
  fi

  echo "[info] Source:  ${remote_count} files, ${remote_size} bytes"
  echo "[info] Target:  ${local_count} files, ${local_size} bytes"

  if [[ "${remote_size}" == "${local_size}" && "${remote_count}" == "${local_count}" ]]; then
    echo "[info] Verification OK — size and file count match."
  else
    echo "[error] Verification FAILED — size or file count mismatch. Rerun the download command to fetch missing/changed data."
  fi
}

do_download() {
  local source="$1"
  local target="$2"

  # Strip any trailing slash and add our own — matters for rsync semantics
  source="${source%/}/"
  target="${target%/}/"

  mkdir -p "${target}"

  ensure_pod

  echo "[info] Calculating total size of remote directory..."
  local size_human
  size_human="$(oc exec "${POD_NAME}" -- du -sh "${source%/}" 2>/dev/null | awk '{print $1}')" || size_human="unknown"
  echo "[info] Total size to download: ${size_human}"

  # rsync's -e flag runs a "remote shell" command as: <command> <host> <remote-command...>
  # We don't have a real host — we have a pod — so this tiny wrapper drops the
  # fake host argument that rsync inserts, and execs "oc exec" in its place.
  local wrapper
  wrapper="$(mktemp)"
  cat > "${wrapper}" <<EOF
#!/bin/sh
shift  # drop the placeholder host argument rsync inserts
exec oc exec -i "${POD_NAME}" -- "\$@"
EOF
  chmod +x "${wrapper}"
  trap 'rm -f "${wrapper}"' RETURN

  echo "[info] Downloading ${POD_NAME}:${source} -> ${target}"
  echo "[info] If interrupted, just rerun this same command — rsync will resume, not restart."

  rsync -rltz --partial --info=progress2 \
    -e "${wrapper}" \
    "${POD_NAME}:${source}" "${target}"

  verify_download "${source}" "${target}"

  echo "[info] Done: ${target}"
  echo "[info] Pod left running for future downloads."
  echo "[info] To delete it: $0 --cleanup"
}

# ---- main ----

if [[ $# -eq 0 ]]; then
  echo "Usage:"
  echo "  $0 <source_path_in_pvc> <local_target_dir>"
  echo "  $0 --cleanup"
  echo "  $0 --status"
  exit 1
fi

case "$1" in
  --cleanup)
    do_cleanup
    ;;
  --status)
    do_status
    ;;
  *)
    if [[ $# -ne 2 ]]; then
      echo "[error] Exactly 2 arguments required: source and target."
      echo "Example: $0 /mnt/models/lerobot/pi05_base/20260731T130734Z ./finetuned/pi05_base_20260731T130734Z"
      exit 1
    fi
    do_download "$1" "$2"
    ;;
esac
