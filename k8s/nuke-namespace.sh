#!/usr/bin/env bash
#
# nuke-namespace.sh — delete all PVCs, Pods, Jobs in a namespace, then delete
# the namespace itself. Destructive and irreversible — requires typed confirmation.
#
# Usage:
#   ./nuke-namespace.sh <namespace>
#
# What it does, in order:
#   1. Shows a summary of what will be deleted (pods, jobs, pvcs, and their
#      backing PVs with reclaim policy)
#   2. Requires you to type the exact namespace name to confirm
#   3. Deletes jobs, then pods, then PVCs (waiting for each to finish)
#   4. Warns if any backing PV has reclaimPolicy=Retain (won't auto-delete,
#      meaning the EBS volume will keep costing money until removed manually)
#   5. Deletes the namespace/project itself

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <namespace>"
  exit 1
fi

NS="$1"

if ! oc get namespace "${NS}" >/dev/null 2>&1; then
  echo "[error] Namespace '${NS}' not found."
  exit 1
fi

echo "=================================================================="
echo " DESTRUCTIVE ACTION — about to wipe namespace: ${NS}"
echo "=================================================================="
echo

echo "--- Jobs ---"
oc get jobs -n "${NS}" 2>/dev/null || echo "(none)"
echo

echo "--- Pods ---"
oc get pods -n "${NS}" 2>/dev/null || echo "(none)"
echo

echo "--- PVCs (and their backing PVs) ---"
oc get pvc -n "${NS}" -o custom-columns=NAME:.metadata.name,VOLUME:.spec.volumeName,SIZE:.spec.resources.requests.storage 2>/dev/null || echo "(none)"
echo

# Check reclaim policy of any bound PVs — Retain means the EBS volume
# will survive PVC/namespace deletion and keep costing money.
PVS="$(oc get pvc -n "${NS}" -o jsonpath='{.items[*].spec.volumeName}' 2>/dev/null || true)"
if [[ -n "${PVS}" ]]; then
  echo "--- Reclaim policy check ---"
  for pv in ${PVS}; do
    policy="$(oc get pv "${pv}" -o jsonpath='{.spec.persistentVolumeReclaimPolicy}' 2>/dev/null || echo "unknown")"
    echo "  PV ${pv}: reclaimPolicy=${policy}"
    if [[ "${policy}" == "Retain" ]]; then
      echo "  [warn] This PV will NOT be auto-deleted and its EBS volume will keep costing money."
      echo "         You'll need to delete it manually afterwards: oc delete pv ${pv}"
    fi
  done
  echo
fi

echo "=================================================================="
echo "This will permanently delete ALL jobs, pods, PVCs, and the namespace"
echo "'${NS}' itself. This cannot be undone."
echo "=================================================================="
echo
read -rp "Type the namespace name to confirm deletion: " CONFIRM

if [[ "${CONFIRM}" != "${NS}" ]]; then
  echo "[abort] Input did not match '${NS}'. Nothing was deleted."
  exit 1
fi

echo
echo "[info] Deleting jobs..."
oc delete jobs --all -n "${NS}" --ignore-not-found --wait=true || true

echo "[info] Deleting pods..."
oc delete pods --all -n "${NS}" --ignore-not-found --wait=true --grace-period=30 || true

echo "[info] Deleting PVCs..."
oc delete pvc --all -n "${NS}" --ignore-not-found --wait=true || true

echo "[info] Deleting namespace '${NS}'..."
oc delete namespace "${NS}" --wait=true

echo "[info] Done. Namespace '${NS}' and its jobs/pods/PVCs have been deleted."
echo "[info] If any PV warnings were shown above (reclaimPolicy=Retain),"
echo "       check 'oc get pv' now and delete leftover volumes manually to avoid ongoing EBS charges."
