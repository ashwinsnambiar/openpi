"""
Convert existing VirERO teleop dataset to OpenPi-compatible LeRobot format.

This script fixes field naming issues in an existing LeRobot dataset to make it
compatible with OpenPi π₀ and π₀.₅ models.

Key fixes:
- Renames "action" → "actions" (plural)
- Renames "observation.state" → "observation/state" (slash notation)
- Renames "observation.images.*" → "observation/image" and "observation/wrist_image"
- Updates metadata files (info.json, episodes.jsonl, etc.)

Usage:
    python examples/virero/convert_existing_dataset.py \
        --input-dir virero_teleop_data/lift_01_10_episodes_annotated/lerobot \
        --output-dir virero_teleop_data/lift_01_10_episodes_openpi
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


def convert_existing_dataset(input_dir: Path, output_dir: Path) -> None:
    """
    Convert existing dataset to OpenPi-compatible format.

    Args:
        input_dir: Path to existing LeRobot dataset
        output_dir: Path for output dataset
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting dataset:")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")

    # Mapping of old field names to new field names
    FIELD_MAPPING = {
        "observation.state": "observation/state",
        "action": "actions",
        "observation.images.overhead_view": "observation/image",
        "observation.images.wrist_view": "observation/wrist_image",
    }

    # 1. Convert parquet files
    data_dir = input_dir / "data"
    output_data_dir = output_dir / "data"

    if data_dir.exists():
        print("\n1. Converting parquet files...")
        for parquet_file in tqdm(list(data_dir.rglob("*.parquet"))):
            # Read parquet file
            df = pd.read_parquet(parquet_file)

            # Rename columns
            df = df.rename(columns=FIELD_MAPPING)

            # Remove extra fields not needed by OpenPi
            columns_to_drop = []
            for col in df.columns:
                if col in ["observation.eef_pose", "observation.img_state_delta"]:
                    columns_to_drop.append(col)
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)

            # Save to output directory
            relative_path = parquet_file.relative_to(data_dir)
            output_file = output_data_dir / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_file)

        print(f"   ✓ Converted {len(list(data_dir.rglob('*.parquet')))} parquet files")

    # 2. Update info.json
    meta_dir = input_dir / "meta"
    output_meta_dir = output_dir / "meta"
    output_meta_dir.mkdir(parents=True, exist_ok=True)

    if (meta_dir / "info.json").exists():
        print("\n2. Updating info.json...")
        with open(meta_dir / "info.json") as f:
            info = json.load(f)

        # Update feature names
        new_features = {}
        for old_name, feature_info in info["features"].items():
            # Map to new name
            new_name = FIELD_MAPPING.get(old_name, old_name)

            # Skip fields that OpenPi doesn't use
            if old_name in ["observation.eef_pose", "observation.img_state_delta"]:
                continue

            # Update dtype for images (optional but recommended)
            if new_name in ["observation/image", "observation/wrist_image"]:
                feature_info["dtype"] = "image"

            new_features[new_name] = feature_info

        info["features"] = new_features

        # Save updated info.json
        with open(output_meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=4)

        print("   ✓ Updated info.json with new field names")

    # 3. Copy other metadata files
    print("\n3. Copying other metadata files...")
    for meta_file in ["episodes.jsonl", "tasks.jsonl", "modality.json"]:
        if (meta_dir / meta_file).exists():
            shutil.copy(meta_dir / meta_file, output_meta_dir / meta_file)
            print(f"   ✓ Copied {meta_file}")

    # 4. Handle videos
    videos_dir = input_dir / "videos"
    output_videos_dir = output_dir / "videos"

    if videos_dir.exists():
        print("\n4. Copying/renaming videos...")
        
        # Walk through video directory and rename camera folders
        for chunk_dir in videos_dir.iterdir():
            if not chunk_dir.is_dir():
                continue

            output_chunk_dir = output_videos_dir / chunk_dir.name
            output_chunk_dir.mkdir(parents=True, exist_ok=True)

            # Rename camera folders
            camera_mapping = {
                "observation.images.overhead_view": "observation.images.image",
                "observation.images.wrist_view": "observation.images.wrist_image",
            }

            for camera_dir in chunk_dir.iterdir():
                if not camera_dir.is_dir():
                    continue

                # Map camera name
                new_camera_name = camera_mapping.get(camera_dir.name, camera_dir.name)
                output_camera_dir = output_chunk_dir / new_camera_name
                
                # Copy videos
                if not output_camera_dir.exists():
                    shutil.copytree(camera_dir, output_camera_dir)

        print(f"   ✓ Copied video files with updated camera names")

    print("\n✅ Conversion complete!")
    print(f"\n📊 Summary:")
    print(f"   Input:  {input_dir}")
    print(f"   Output: {output_dir}")
    print("\n🔍 Verify the output:")
    print("   1. Check parquet columns have 'observation/state' and 'actions' (with slashes)")
    print("   2. Check info.json has 'observation/image' and 'observation/wrist_image'")
    print("   3. Test loading with OpenPi training config")


def verify_dataset(dataset_dir: Path) -> None:
    """Verify that the converted dataset has the correct format."""
    dataset_dir = Path(dataset_dir)

    print("\n" + "=" * 80)
    print("DATASET VERIFICATION")
    print("=" * 80)

    # Check info.json
    info_path = dataset_dir / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)

        print("\n✓ Features in info.json:")
        for feature_name in info["features"].keys():
            status = "✓" if "/" in feature_name or feature_name in ["actions", "timestamp", "episode_index", "frame_index", "index", "next.reward", "next.done", "task_index"] else "⚠"
            print(f"  {status} {feature_name}")

    # Check a sample parquet file
    data_dir = dataset_dir / "data"
    parquet_files = list(data_dir.rglob("*.parquet"))
    if parquet_files:
        df = pd.read_parquet(parquet_files[0])
        print("\n✓ Columns in parquet file:")
        for col in df.columns:
            status = "✓" if "/" in col or col in ["actions", "timestamp", "episode_index", "frame_index", "index", "next.reward", "next.done", "task_index", "prompt", "task"] else "⚠"
            print(f"  {status} {col}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Convert existing VirERO dataset to OpenPi-compatible format"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to existing LeRobot dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path for output dataset directory",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the output dataset after conversion",
    )

    args = parser.parse_args()

    convert_existing_dataset(args.input_dir, args.output_dir)

    if args.verify:
        verify_dataset(args.output_dir)


if __name__ == "__main__":
    main()
