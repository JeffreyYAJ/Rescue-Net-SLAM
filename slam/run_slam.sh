#!/usr/bin/env bash
# Run ORB-SLAM3 on TUM RGB-D or EuRoC and export poses in TUM format.
#
# Usage:
#   ./slam/run_slam.sh tum  /path/to/rgbd_dataset_freiburg1_xyz  output/poses.txt
#   ./slam/run_slam.sh euroc /path/to/MH_01_easy/mav0           output/poses.txt
#
# Requires ORB-SLAM3 built under third_party/ORB_SLAM3 (see slam/README.md).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ORB_ROOT="${REPO_ROOT}/third_party/ORB_SLAM3"

DATASET_TYPE="${1:-}"
DATASET_PATH="${2:-}"
OUTPUT_POSES="${3:-${REPO_ROOT}/output/poses.txt}"

if [[ -z "${DATASET_TYPE}" || -z "${DATASET_PATH}" ]]; then
  echo "Usage: $0 <tum|euroc> <dataset_path> [output_poses.txt]"
  exit 1
fi

if [[ ! -d "${ORB_ROOT}" ]]; then
  echo "ORB-SLAM3 not found at ${ORB_ROOT}"
  echo "Initialize submodule: git submodule update --init --recursive"
  exit 1
fi

VOCAB="${ORB_ROOT}/Vocabulary/ORBvoc.txt"
if [[ ! -f "${VOCAB}" ]]; then
  echo "ORB vocabulary missing: ${VOCAB}"
  echo "Download ORBvoc.txt per slam/README.md"
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_POSES}")"
WORKDIR="${REPO_ROOT}/output/slam_run"
mkdir -p "${WORKDIR}"

case "${DATASET_TYPE}" in
  tum)
    EXEC="${ORB_ROOT}/Examples/RGB-D/rgbd_tum"
    SETTINGS="${ORB_ROOT}/Examples/RGB-D/TUM1.yaml"
    ASSOC="${DATASET_PATH}/associations.txt"

    if [[ ! -f "${ASSOC}" ]]; then
      echo "associations.txt not found — build it with the Python pipeline first,"
      echo "or place a standard TUM association file at: ${ASSOC}"
      exit 1
    fi

    if [[ ! -x "${EXEC}" ]]; then
      EXEC="${ORB_ROOT}/Examples/RGB-D/rgbd_tum"
    fi

    "${EXEC}" "${VOCAB}" "${SETTINGS}" "${DATASET_PATH}" "${ASSOC}"

    # ORB-SLAM3 saves CameraTrajectory.txt in cwd
    if [[ -f "CameraTrajectory.txt" ]]; then
      cp "CameraTrajectory.txt" "${OUTPUT_POSES}"
      mv "CameraTrajectory.txt" "${WORKDIR}/" || true
    elif [[ -f "KeyFrameTrajectory.txt" ]]; then
      cp "KeyFrameTrajectory.txt" "${OUTPUT_POSES}"
    else
      echo "ORB-SLAM3 did not produce CameraTrajectory.txt"
      exit 1
    fi
    ;;

  euroc)
    EXEC="${ORB_ROOT}/Examples/Monocular/mono_euroc"
    SETTINGS="${ORB_ROOT}/Examples/Monocular/EuRoC.yaml"
    TIMES="${DATASET_PATH}/cam0/data.csv"

    "${EXEC}" "${VOCAB}" "${SETTINGS}" "${DATASET_PATH}" "${TIMES}"

    if [[ -f "CameraTrajectory.txt" ]]; then
      cp "CameraTrajectory.txt" "${OUTPUT_POSES}"
    else
      echo "ORB-SLAM3 did not produce CameraTrajectory.txt"
      exit 1
    fi
    ;;

  *)
    echo "Unknown dataset type: ${DATASET_TYPE}. Use tum or euroc."
    exit 1
    ;;
esac

echo "Poses saved to ${OUTPUT_POSES}"
