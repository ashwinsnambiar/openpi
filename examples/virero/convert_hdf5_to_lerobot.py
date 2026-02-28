"""
Convert SIA20F HDF5 teleoperation data to LeRobot format for OpenPi fine-tuning.

This script converts HDF5 files from the Yaskawa SIA20F robot to the LeRobot v2 format
with the correct field naming conventions required by OpenPi π₀ and π₀.₅ models.

Key features:
- Uses correct field names: observation/state, observation/image, observation/wrist_image, actions
- Extracts 8 actuated DOF (7 arm joints + 1 gripper)
- Derives joint-space actions from trajectory: action[t] = joint_pos[t+1]
- Saves videos in mp4 format

Usage:
    python examples/virero/convert_hdf5_to_lerobot.py \
        --hdf5-file /path/to/data.hdf5 \
        --output-dir /path/to/output \
        --task-description "pick and place red cube"
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torchvision
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset
from tqdm import tqdm

# Indices of the 8 actuated joints within the 15-joint state array
ACTUATED_JOINT_INDICES = [2, 3, 4, 5, 6, 7, 8, 9]

ACTUATED_JOINT_NAMES = [
    "sia20f_joint_s",
    "sia20f_joint_l",
    "sia20f_joint_e",
    "sia20f_joint_u",
    "sia20f_joint_r",
    "sia20f_joint_b",
    "sia20f_joint_t",
    "sia20f_gripper_finger_joint",
]


def convert_hdf5_to_lerobot(
    hdf5_file: Path,
    output_dir: Path,
    repo_name: str,
    task_description: str,
    fps: int = 30,
    image_size: tuple[int, int] = (480, 480),
    *,
    push_to_hub: bool = False,
) -> None:
    """
    Convert SIA20F HDF5 dataset to LeRobot format.

    Args:
        hdf5_file: Path to input HDF5 file
        output_dir: Output directory for LeRobot dataset
        repo_name: Repository name (e.g., "username/virero_sia20f")
        task_description: Natural language task description
        fps: Frames per second
        image_size: Target image size (height, width)
        push_to_hub: Whether to push to Hugging Face Hub
    """
    # Override LEROBOT_HOME if output_dir specified
    if output_dir:
        output_path = output_dir / repo_name.split("/")[-1]
    else:
        output_path = LEROBOT_HOME / repo_name

    # Clean up existing dataset
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)

    # Create LeRobot dataset with OpenPi-compatible field names
    print(f"Creating LeRobot dataset: {repo_name}")
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="yaskawa_sia20f",
        fps=fps,
        features={
            # Images: use observation/image and observation/wrist_image (with slashes!)
            "observation/image": {
                "dtype": "image",
                "shape": (image_size[0], image_size[1], 3),
                "names": ["height", "width", "channel"],
            },
            "observation/wrist_image": {
                "dtype": "image",
                "shape": (image_size[0], image_size[1], 3),
                "names": ["height", "width", "channel"],
            },
            # State: use observation/state (with slash!)
            "observation/state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ACTUATED_JOINT_NAMES,
            },
            # Actions: use "actions" (plural!)
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ACTUATED_JOINT_NAMES,
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Load HDF5 data
    print(f"Loading HDF5 file: {hdf5_file}")
    with h5py.File(hdf5_file, "r") as hdf5_handler:
        hdf5_data = hdf5_handler["data"]
        trajectory_ids = list(hdf5_data.keys())
        print(f"Found {len(trajectory_ids)} trajectories")

        # Process each episode
        for episode_index, trajectory_id in enumerate(tqdm(trajectory_ids, desc="Converting episodes")):
            trajectory = hdf5_data[trajectory_id]

            # 1. Read absolute joint positions from states/
            if "states" not in trajectory.keys():
                print(f"Warning: Skipping trajectory {trajectory_id} - missing 'states' group")
                continue

            abs_joint_pos = np.array(
                trajectory["states"]["articulation"]["robot"]["joint_position"]
            )  # shape: (N, 15)

            if abs_joint_pos.ndim != 2 or abs_joint_pos.shape[1] != 15:
                print(f"Warning: Skipping trajectory {trajectory_id} - unexpected joint shape {abs_joint_pos.shape}")
                continue

            # 2. Extract only the 8 actuated joints
            actuated_pos = abs_joint_pos[:, ACTUATED_JOINT_INDICES]  # shape: (N, 8)

            # 3. Derive state and action from trajectory
            # state[t] = joint_pos[t], action[t] = joint_pos[t+1]
            states = actuated_pos[:-1].astype(np.float32)  # shape: (N-1, 8)
            actions = actuated_pos[1:].astype(np.float32)  # shape: (N-1, 8)
            length = states.shape[0]

            # 4. Load images from obs/ group
            obs = trajectory.get("obs", {})
            
            # Try to get camera data
            overhead_images = None
            wrist_images = None
            
            if "overhead_cam" in obs:
                overhead_images = np.array(obs["overhead_cam"]["rgb"])
            if "wrist_cam" in obs:
                wrist_images = np.array(obs["wrist_cam"]["rgb"])

            # Fallback: check for alternative camera names
            if overhead_images is None:
                for key in ["camera", "cam_high", "agentview"]:
                    if key in obs:
                        overhead_images = np.array(obs[key]["rgb"] if "rgb" in obs[key] else obs[key])
                        break

            if wrist_images is None:
                for key in ["wrist", "cam_wrist", "eye_in_hand"]:
                    if key in obs:
                        wrist_images = np.array(obs[key]["rgb"] if "rgb" in obs[key] else obs[key])
                        break

            # Handle missing cameras
            if overhead_images is None:
                print(f"Warning: No overhead camera found for trajectory {trajectory_id}, using zeros")
                overhead_images = np.zeros((length + 1, *image_size, 3), dtype=np.uint8)
            if wrist_images is None:
                print(f"Warning: No wrist camera found for trajectory {trajectory_id}, using zeros")
                wrist_images = np.zeros((length + 1, *image_size, 3), dtype=np.uint8)

            # Ensure images are (N, H, W, 3) and match trajectory length
            overhead_images = overhead_images[:-1]  # Match state/action length
            wrist_images = wrist_images[:-1]

            # 5. Add frames to dataset using OpenPi-compatible field names
            for t in range(length):
                frame_data = {
                    "observation/image": overhead_images[t],  # Use slash notation!
                    "observation/wrist_image": wrist_images[t],  # Use slash notation!
                    "observation/state": states[t],  # Use slash notation!
                    "actions": actions[t],  # Use plural!
                    "task": task_description,  # Language instruction
                }
                dataset.add_frame(frame_data)

            # Save episode
            dataset.save_episode()

    print(f"\n✅ Conversion complete!")
    print(f"   Output: {output_path}")
    print(f"   Episodes: {len(trajectory_ids)}")

    # Optionally push to hub
    if push_to_hub:
        print("\nPushing to Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["virero", "sia20f", "openpi"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert SIA20F HDF5 data to LeRobot format for OpenPi"
    )
    parser.add_argument(
        "--hdf5-file",
        type=Path,
        required=True,
        help="Path to input HDF5 file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: $LEROBOT_HOME)",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="virero/sia20f_teleop",
        help="Repository name (default: virero/sia20f_teleop)",
    )
    parser.add_argument(
        "--task-description",
        type=str,
        required=True,
        help="Natural language task description (e.g., 'pick and place red cube')",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[480, 480],
        help="Image size [height width] (default: 480 480)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push dataset to Hugging Face Hub",
    )

    args = parser.parse_args()

    convert_hdf5_to_lerobot(
        hdf5_file=args.hdf5_file,
        output_dir=args.output_dir,
        repo_name=args.repo_name,
        task_description=args.task_description,
        fps=args.fps,
        image_size=tuple(args.image_size),
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
