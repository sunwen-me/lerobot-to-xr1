#!/usr/bin/env python3
"""Full read-only validation for an official-URDF reconversion directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


POSE_KEYS = {
    ("proprios", "left_ee_pos"),
    ("proprios", "left_ee_rotm"),
    ("proprios", "right_ee_pos"),
    ("proprios", "right_ee_rotm"),
    ("actions", "left_ee_pos"),
    ("actions", "left_ee_rotm"),
    ("actions", "right_ee_pos"),
    ("actions", "right_ee_rotm"),
}


def assert_same_except(old: dict, new: dict, label: str) -> None:
    if set(old) != set(new):
        raise ValueError(f"{label}: top-level keys differ")
    for key in old:
        if key in ("proprios", "actions"):
            if set(old[key]) != set(new[key]):
                raise ValueError(f"{label}: {key} keys differ")
            for subkey in old[key]:
                if (key, subkey) not in POSE_KEYS and old[key][subkey] != new[key][subkey]:
                    raise ValueError(f"{label}: unexpected change at {key}.{subkey}")
        elif old[key] != new[key]:
            raise ValueError(f"{label}: unexpected change at {key}")


def check_pose(values, count: int, shape: tuple[int, ...], label: str):
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{label}: expected {shape}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{label}: NaN/Inf found")
    if shape[1] != 9:
        return
    mats = arr.reshape(count, 3, 3)
    ident = np.eye(3)
    orth_error = np.max(np.abs(np.einsum("nji,njk->nik", mats, mats) - ident))
    det = np.linalg.det(mats)
    det_error = float(np.max(np.abs(det - 1.0)))
    if orth_error > 2e-6 or det_error > 2e-6:
        raise ValueError(f"{label}: invalid rotation matrix: orth={orth_error}, det_error={det_error}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--old-json-dir", type=Path, required=True)
    p.add_argument("--new-json-dir", type=Path, required=True)
    p.add_argument("--sample-indices", type=int, nargs="*", default=None,
                   help="Optional episode indices for deep old/new JSON validation.")
    args = p.parse_args()

    info = json.loads((args.dataset / "meta/info.json").read_text())
    metas = []
    for file in sorted((args.dataset / "meta/episodes").glob("chunk-*/file-*.parquet")):
        metas.extend(pq.read_table(file).to_pylist())
    metas.sort(key=lambda row: int(row["episode_index"]))
    expected_count = int(info["total_episodes"])
    if len(metas) != expected_count:
        raise ValueError(f"source metadata episodes mismatch: {len(metas)} != {expected_count}")
    old_files = sorted(args.old_json_dir.glob("episode_*.json"))
    new_files = sorted(args.new_json_dir.glob("episode_*.json"))
    if len(old_files) != expected_count or len(new_files) != expected_count:
        raise ValueError(f"JSON file counts: old={len(old_files)} new={len(new_files)} expected={expected_count}")

    total_frames = sum(int(meta["length"]) for meta in metas)
    selected = set(args.sample_indices) if args.sample_indices is not None else set(range(expected_count))
    if not selected or min(selected) < 0 or max(selected) >= expected_count:
        raise ValueError(f"sample indices must be within [0, {expected_count - 1}]")
    video_checks = 0
    max_rotation_orth_error = 0.0
    max_rotation_det_error = 0.0
    for index, meta in enumerate(metas):
        if int(meta["episode_index"]) != index:
            raise ValueError(f"source episode index gap at {index}")
        filename = f"episode_{index:05d}.json"
        if index not in selected:
            continue
        old = json.loads((args.old_json_dir / filename).read_text())
        new = json.loads((args.new_json_dir / filename).read_text())
        assert_same_except(old, new, filename)
        count = int(meta["length"])
        if old.get("num_frames") != count or new.get("num_frames") != count:
            raise ValueError(f"{filename}: frame count mismatch against source metadata")

        for bucket, side in (
            ("proprios", "left"), ("proprios", "right"),
            ("actions", "left"), ("actions", "right"),
        ):
            check_pose(new[bucket][f"{side}_ee_pos"], count, (count, 3), f"{filename}.{bucket}.{side}_ee_pos")
            mats = np.asarray(new[bucket][f"{side}_ee_rotm"], dtype=np.float64).reshape(count, 3, 3)
            check_pose(new[bucket][f"{side}_ee_rotm"], count, (count, 9), f"{filename}.{bucket}.{side}_ee_rotm")
            orth_error = float(np.max(np.abs(np.einsum("nji,njk->nik", mats, mats) - np.eye(3))))
            det_error = float(np.max(np.abs(np.linalg.det(mats) - 1.0)))
            max_rotation_orth_error = max(max_rotation_orth_error, orth_error)
            max_rotation_det_error = max(max_rotation_det_error, det_error)

        for view in ("ego", "wrist_left", "wrist_right"):
            observations = new["observations"].get(view)
            if not isinstance(observations, list) or not observations:
                raise ValueError(f"{filename}: missing observation video reference {view}")
            for observation in observations:
                video = Path(observation["path"])
                if not video.is_file():
                    raise FileNotFoundError(f"{filename}.{view}: {video}")
                video_checks += 1

    if total_frames != int(info["total_frames"]):
        raise ValueError(f"total frame mismatch: {total_frames} != {info['total_frames']}")
    print(json.dumps({
        "status": "PASS",
        "episodes": expected_count,
        "frames": total_frames,
        "deep_json_comparison_episode_indices": sorted(selected),
        "videos_referenced_and_present": video_checks,
        "rotation_max_orthogonality_error": max_rotation_orth_error,
        "rotation_max_det_error": max_rotation_det_error,
        "only_pose_arrays_changed_from_legacy": True,
    }, indent=2))


if __name__ == "__main__":
    main()
