# LeRobot → XR-1 verified SO-101 conversion

This repository contains the conversion and validation scripts actually used for the RW-RL dual-SO101 fold-clothes dataset. It is a **reconversion of an existing XR-1 dataset**, not a generic exporter that builds all XR-1 fields from scratch.

The converter reads the original LeRobot Parquet joint state/action arrays and an existing XR-1 JSON episode. It preserves that XR-1 episode's schema, instruction, and video references, verifies that the saved joint/gripper values match the source, then replaces only the four state/action EE pose arrays using the calibrated SO-101 URDF. This is the validated data path.

## Verified conventions

- Source: dual SO-101 LeRobot dataset, 12 channels ordered left arm joints 1–5, left gripper, right arm joints 1–5, right gripper.
- Arm joint angles are degrees; gripper values pass through unchanged.
- FK chain: `base_link` → `gripper_frame_link`.
- URDF joint limits are not used to clamp, reject, or filter data.
- Only `proprios.{left,right}_ee_{pos,rotm}` and `actions.{left,right}_ee_{pos,rotm}` are replaced. Joint arrays, gripper arrays, instruction/schema, and video references are retained from the original XR-1 JSON.
- All source episodes are converted; the converter does not filter success/failure labels.

The calibrated URDF is intentionally not bundled. Supply the exact `so101_new_calib.urdf` used for validation (SHA-256: `3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c`). A different URDF or zero convention changes the EE labels.

## Install

Python 3.10+ is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Convert

The source dataset, its prior XR-1 conversion, and the calibrated URDF must all be available locally. Start with a separate one-episode output directory:

```bash
python scripts/convert_rwrl_so101_official_urdf.py \
  --dataset-dir /data/RW-RL-Dataset/so101_fold_clothes_left_stack_right \
  --old-json-dir /data/xr1_rwrl_json/so101_fold_clothes_left_stack_right \
  --urdf /models/so101_new_calib.urdf \
  --output-dir /tmp/xr1-fold-smoke \
  --max-episodes 1
```

After checking the smoke output, run the full dataset into a new empty directory (omit `--max-episodes`):

```bash
python scripts/convert_rwrl_so101_official_urdf.py \
  --dataset-dir /data/RW-RL-Dataset/so101_fold_clothes_left_stack_right \
  --old-json-dir /data/xr1_rwrl_json/so101_fold_clothes_left_stack_right \
  --urdf /models/so101_new_calib.urdf \
  --output-dir /data/xr1_rwrl_json_official_urdf
```

The output includes one `episode_XXXXX.json` per source episode and `conversion_manifest.json` with source/URDF hashes, frame counts, output hashes, and the EE-frame/limit policy. `--resume` is only for continuing an interrupted run made with the same inputs and output directory.

## Validate

The validator checks episode counts and frame totals against LeRobot metadata, compares every output JSON with its original except the four intended EE pose arrays, checks pose shapes and rotation matrices, and confirms that referenced videos exist. With no `--sample-indices`, it deeply checks all episodes:

```bash
python scripts/validate_official_urdf_conversion.py \
  --dataset /data/RW-RL-Dataset/so101_fold_clothes_left_stack_right \
  --old-json-dir /data/xr1_rwrl_json/so101_fold_clothes_left_stack_right \
  --new-json-dir /data/xr1_rwrl_json_official_urdf/so101_fold_clothes_left_stack_right
```

`official_urdf_reconvert_one_episode.py` is the independent one-episode audit/report tool. It recomputes FK directly from LeRobot Parquet and uses the old XR-1 JSON only for schema and video references; it does not use old EE labels as targets.

## Validation record

See [VALIDATION.md](VALIDATION.md) for the server-side real-data regression, file hashes, and the exact scope of what was and was not rerun. Synthetic CI checks are not presented as real-dataset validation.

## Scope

Offline conversion only. This repository does not include the dataset, videos, generated annotations, URDF, meshes, robot-control code, IK, or policy inference. Keep generated data and manifests private unless you have permission to publish the source dataset and video references.
