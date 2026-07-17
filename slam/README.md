# ORB-SLAM3 integration

Rescue-Net uses [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) for camera pose estimation.
Phase 1 runs SLAM offline on public datasets and exports trajectories in TUM format.

## Setup

### 1. Initialize submodule

From the repository root:

```bash
git submodule add https://github.com/UZ-SLAMLab/ORB_SLAM3.git third_party/ORB_SLAM3
git submodule update --init --recursive
```

If the submodule is already configured:

```bash
git submodule update --init --recursive
```

### 2. Download vocabulary

ORB-SLAM3 requires `ORBvoc.txt` (~145 MB):

```bash
cd third_party/ORB_SLAM3/Vocabulary
tar -xf ORBvoc.txt.tar.gz
```

### 3. Build ORB-SLAM3

Dependencies (Ubuntu/Debian):

```bash
sudo apt install cmake git libeigen3-dev libopencv-dev libglew-dev libboost-all-dev
```

Build Pangolin (if not installed), then:

```bash
cd third_party/ORB_SLAM3
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Executables used by Rescue-Net:

- `Examples/RGB-D/rgbd_tum` — TUM RGB-D datasets
- `Examples/Monocular/mono_euroc` — EuRoC MAV monocular

## Running SLAM

```bash
chmod +x slam/run_slam.sh

# TUM RGB-D
./slam/run_slam.sh tum data/tum/rgbd_dataset_freiburg1_xyz output/poses.txt

# EuRoC
./slam/run_slam.sh euroc data/euroc/MH_01_easy/mav0 output/poses.txt
```

Output: `output/poses.txt` in TUM format:

```
timestamp tx ty tz qx qy qz qw
```

## Fallback (without building ORB-SLAM3)

The Python pipeline accepts an existing `poses.txt`. You can:

1. Run ORB-SLAM3 separately and copy `CameraTrajectory.txt` to `output/poses.txt`
2. Use identity poses for semantic-only testing: the pipeline will still detect and project objects in the camera frame (limited world consistency)

## Config references

- TUM: [`config/slam/tum_rgbd.yaml`](../config/slam/tum_rgbd.yaml)
- EuRoC: [`config/slam/euroc_mono.yaml`](../config/slam/euroc_mono.yaml)
