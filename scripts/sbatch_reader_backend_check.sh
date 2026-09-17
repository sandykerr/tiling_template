#!/usr/bin/env bash
#SBATCH --job-name=reader_backend_check
#SBATCH --partition=grace
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=scripts/logs/reader_backend_check_%j.out
#SBATCH --error=scripts/logs/reader_backend_check_%j.err

set -euo pipefail

usage() {
  echo "Usage: sbatch $0 --backend <rasterio|xarray> \
--datasets DATASET [DATASET ...] [backend options]" >&2
}

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

START_TIME="$(date +%s)"
START_READABLE="$(date)"
BACKEND=""
HAS_DATASETS=false
PYTHON_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      if [[ $# -lt 2 ]]; then
        echo "--backend requires a value." >&2
        usage
        exit 2
      fi
      BACKEND="$2"
      shift 2
      ;;
    --datasets)
      HAS_DATASETS=true
      PYTHON_ARGS+=("$1")
      shift
      ;;
    --workers)
      echo "Worker count is controlled by sbatch --cpus-per-task." >&2
      exit 2
      ;;
    *)
      PYTHON_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${HAS_DATASETS}" != true ]]; then
  echo "--datasets is required." >&2
  usage
  exit 2
fi

case "${BACKEND}" in
  rasterio)
    SCRIPT_REL="scripts/rasterio_reader_check.py"
    ;;
  xarray)
    SCRIPT_REL="scripts/xarray_reader_check.py"
    ;;
  *)
    echo "--backend must be either 'rasterio' or 'xarray'." >&2
    usage
    exit 2
    ;;
esac

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "${SUBMIT_DIR}/${SCRIPT_REL}" ]]; then
  REPO_DIR="${SUBMIT_DIR}"
elif [[ -f "${SUBMIT_DIR}/../${SCRIPT_REL}" ]]; then
  REPO_DIR="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  echo "Could not locate ${SCRIPT_REL} from: ${SUBMIT_DIR}" >&2
  echo "Submit from the repository root or the scripts directory." >&2
  exit 1
fi

cd "${REPO_DIR}"
mkdir -p scripts/logs

WORKERS="${SLURM_CPUS_PER_TASK:-1}"
if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SLURM_CPUS_PER_TASK must be a positive integer: ${WORKERS}" >&2
  exit 1
fi

CONTAINER_ROOT="/explore/nobackup/people/ajkerr1/containers"
DEFAULT_CONTAINER_PATH="${CONTAINER_ROOT}/pace-container-arm64"
CONTAINER_PATH="${CONTAINER_PATH:-${DEFAULT_CONTAINER_PATH}}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"
APPTAINER_BIND_PATHS="${APPTAINER_BIND_PATHS:-/panfs/ccds02/nobackup:/explore/nobackup}"
CONTAINER_PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "Job started at: ${START_READABLE}"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "Partition: ${SLURM_JOB_PARTITION:-grace}"
echo "Repository: ${REPO_DIR}"
echo "Container: ${CONTAINER_PATH}"
echo "Backend: ${BACKEND}"
echo "Workers: ${WORKERS}"
echo

"${APPTAINER_BIN}" exec \
  --bind "${APPTAINER_BIND_PATHS}" \
  --bind "${REPO_DIR}" \
  --pwd "${REPO_DIR}" \
  --env "PYTHONPATH=${CONTAINER_PYTHONPATH}" \
  "${CONTAINER_PATH}" \
  python -u "${SCRIPT_REL}" "${PYTHON_ARGS[@]}" --workers "${WORKERS}"

END_TIME="$(date +%s)"
END_READABLE="$(date)"
ELAPSED_SECONDS="$((END_TIME - START_TIME))"

printf -v ELAPSED_HMS "%02d:%02d:%02d" \
  "$((ELAPSED_SECONDS / 3600))" \
  "$(((ELAPSED_SECONDS % 3600) / 60))" \
  "$((ELAPSED_SECONDS % 60))"

echo
echo "${BACKEND} reader checks completed successfully."
echo "Job finished at: ${END_READABLE}"
echo "Elapsed time: ${ELAPSED_HMS} (${ELAPSED_SECONDS} seconds)"
