# VirERO SIA20F Examples

This directory contains scripts for converting VirERO SIA20F robot data to OpenPi-compatible LeRobot format.

## Scripts

### 1. `convert_hdf5_to_lerobot.py`

Converts HDF5 teleoperation data directly to OpenPi-compatible LeRobot format.

**Usage:**
```bash
python examples/virero/convert_hdf5_to_lerobot.py \
    --hdf5-file /path/to/your/data.hdf5 \
    --output-dir ./converted_data \
    --repo-name virero/sia20f_pick_place \
    --task-description "pick and place the red cube on the blue platform" \
    --fps 30
```

**Arguments:**
- `--hdf5-file`: Path to input HDF5 file
- `--output-dir`: Output directory (default: `$LEROBOT_HOME`)
- `--repo-name`: Dataset repository name (default: `virero/sia20f_teleop`)
- `--task-description`: Natural language task description (required)
- `--fps`: Frames per second (default: 30)
- `--image-size`: Image size [height width] (default: 480 480)
- `--push-to-hub`: Push to Hugging Face Hub

**Output Format:**
- Field names: `observation/state`, `observation/image`, `observation/wrist_image`, `actions` (plural!)
- Uses forward slashes in field names (required by OpenPi)
- 8 DOF actions: 7 arm joints + 1 gripper
- Actions derived as next-state targets: `action[t] = joint_pos[t+1]`

---

### 2. `convert_existing_dataset.py`

Fixes field naming issues in an existing LeRobot dataset to make it OpenPi-compatible.

**Usage:**
```bash
python examples/virero/convert_existing_dataset.py \
    --input-dir virero_teleop_data/lift_01_10_episodes_annotated/lerobot \
    --output-dir virero_teleop_data/lift_01_10_episodes_openpi \
    --verify
```

**Arguments:**
- `--input-dir`: Path to existing LeRobot dataset
- `--output-dir`: Path for output dataset
- `--verify`: Verify the output dataset after conversion

**What it fixes:**
- `action` → `actions` (plural)
- `observation.state` → `observation/state` (slash notation)
- `observation.images.overhead_view` → `observation/image`
- `observation.images.wrist_view` → `observation/wrist_image`
- Removes unused fields: `observation.eef_pose`, `observation.img_state_delta`
- Updates all metadata files

---

## Data Format Requirements

OpenPi models expect LeRobot datasets with these **exact** field names:

### Required Fields (in parquet files)
- `observation/state`: Robot joint positions (8 DOF for SIA20F)
- `observation/image`: Main/base/third-person camera view
- `observation/wrist_image`: Wrist-mounted camera view
- `actions`: Robot actions (plural, not singular!)
- `prompt` or `task`: Language instruction

### Standard Fields (auto-generated)
- `episode_index`: Episode number
- `frame_index`: Frame within episode  
- `index`: Global frame index
- `timestamp`: Time in seconds
- `next.reward`: Reward (0 except 1 at episode end)
- `next.done`: Done flag (False except True at episode end)

### Critical Notes
- **Use forward slashes** (`/`) not dots (`.`) in field names
- **Use plural** `actions` not singular `action`
- The image dtype should be `"image"` not `"video"` (though both may work)

---

## SIA20F Robot Details

### Actuated Joints (8 DOF)
From the 15 total joints, only 8 are actuated:
```
Index | Joint Name
------|--------------------------
  2   | sia20f_joint_s    (arm)
  3   | sia20f_joint_l    (arm)
  4   | sia20f_joint_e    (arm)
  5   | sia20f_joint_u    (arm)
  6   | sia20f_joint_r    (arm)
  7   | sia20f_joint_b    (arm)
  8   | sia20f_joint_t    (arm)
  9   | sia20f_gripper_finger_joint
```

Excluded joints:
- Indices 0-1: Linear actuators (constant)
- Indices 10-14: Mimic gripper joints

---

## Verification

After conversion, verify your dataset:

```python
import pandas as pd
import json

# Check parquet columns
df = pd.read_parquet("path/to/episode.parquet")
print("Columns:", df.columns.tolist())
# Should include: observation/state, observation/image, observation/wrist_image, actions

# Check info.json
with open("path/to/meta/info.json") as f:
    info = json.load(f)
print("Features:", list(info["features"].keys()))
# Should include: observation/image, observation/wrist_image, observation/state, actions
```

---

## Next Steps: Fine-tuning OpenPi

After converting your dataset, you need to:

1. **Create a policy config** (`src/openpi/policies/sia20f_policy.py`):
   - Define `SIA20FInputs` class (similar to `LiberoInputs`)
   - Define `SIA20FOutputs` class (similar to `LiberoOutputs`)
   - Map your 8-DOF robot data to model inputs/outputs

2. **Create a training config** (in `src/openpi/training/config.py`):
   - Add `LeRobotSIA20FDataConfig` class
   - Define repack transforms, data transforms, model transforms
   - Configure normalization and action space

3. **Compute normalization statistics**:
   ```bash
   uv run scripts/compute_norm_stats.py --config-name sia20f_config
   ```

4. **Run fine-tuning**:
   ```bash
   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py sia20f_config --exp-name=my_experiment
   ```

See the [main README](../../README.md) for detailed fine-tuning instructions.

---

## Troubleshooting

**Problem:** `KeyError: 'observation.state'` during training

**Solution:** Field names must use forward slashes (`/`), not dots (`.`). Use `observation/state`.

---

**Problem:** `KeyError: 'action'` during training

**Solution:** Use plural `actions`, not singular `action`.

---

**Problem:** Images not loading

**Solution:** Ensure images are named `observation/image` and `observation/wrist_image` (with slashes).

---

**Problem:** Action dimensions mismatch

**Solution:** Verify you're extracting exactly 8 DOF (indices 2-9 from 15-joint array).

---

## References

- [OpenPi README](../../README.md)
- [Libero Example](../libero/convert_libero_data_to_lerobot.py)
- [DROID Example](../droid/convert_droid_data_to_lerobot.py)
- [LeRobot Documentation](https://github.com/huggingface/lerobot)
