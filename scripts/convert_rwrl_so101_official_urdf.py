#!/usr/bin/env python3
"""Rebuild XR1 JSON EE labels with the official SO101 calibrated URDF.

The source LeRobot joint arrays and old XR1 JSON are treated as immutable
references. Only the four EE pose arrays (left/right state/action) are replaced.
The converter intentionally does not read, enforce, or filter on URDF limits.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
RECONVERTER = HERE / "official_urdf_reconvert_one_episode.py"


def load_reconverter(path: Path):
    spec = importlib.util.spec_from_file_location("official_urdf_reconverter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import FK implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def close_exact(actual, expected, label: str) -> None:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        max_error = (
            float(np.max(np.abs(actual - expected)))
            if actual.shape == expected.shape and actual.size
            else None
        )
        raise ValueError(f"old XR1 {label} does not exactly match source LeRobot values; max_abs_error={max_error}")


def get_rows(dataset: Path) -> list[dict]:
    rows = []
    for path in sorted((dataset / "meta/episodes").glob("chunk-*/file-*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return sorted(rows, key=lambda row: int(row["episode_index"]))


def read_episode(dataset: Path, info: dict, episode: dict):
    path = dataset / info["data_path"].format(
        chunk_index=int(episode["data/chunk_index"]),
        file_index=int(episode["data/file_index"]),
    )
    table = pq.read_table(
        path, columns=["episode_index", "frame_index", "observation.state", "action"]
    )
    mask = np.asarray(table["episode_index"].to_numpy()) == int(episode["episode_index"])
    frame_index = np.asarray(table["frame_index"].to_numpy())[mask]
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)[mask]
    action = np.asarray(table["action"].to_pylist(), dtype=np.float64)[mask]
    expected = int(episode["length"])
    if state.shape != (expected, 12) or action.shape != (expected, 12):
        raise ValueError(
            f"episode {episode['episode_index']}: expected ({expected}, 12), got {state.shape}, {action.shape}"
        )
    if not np.array_equal(frame_index, np.arange(expected)):
        raise ValueError(f"episode {episode['episode_index']}: non-contiguous frame indices")
    return path, state, action


def validate_legacy_joints(trajectory: dict, state: np.ndarray, action: np.ndarray) -> None:
    for key, values in (
        ("left_arm_joint", state[:, :5]),
        ("left_gripper_pos", state[:, 5:6]),
        ("right_arm_joint", state[:, 6:11]),
        ("right_gripper_pos", state[:, 11:12]),
    ):
        close_exact(trajectory["proprios"][key], values, f"proprios.{key}")
    for key, values in (
        ("left_gripper_pos", action[:, 5:6]),
        ("right_gripper_pos", action[:, 11:12]),
    ):
        close_exact(trajectory["actions"][key], values, f"actions.{key}")


def convert_one(
    index: int, episode: dict, dataset: Path, info: dict, old_json_dir: Path,
    output_dir: Path, fk, resume: bool,
) -> dict:
    old_path = old_json_dir / f"episode_{index:05d}.json"
    output_path = output_dir / old_path.name
    if not old_path.is_file():
        raise FileNotFoundError(old_path)
    trajectory = json.loads(old_path.read_text(encoding="utf-8"))
    data_path, state, action = read_episode(dataset, info, episode)
    if int(trajectory.get("num_frames", -1)) != len(state):
        raise ValueError(f"episode {index}: old XR1 JSON frame count differs from source")
    validate_legacy_joints(trajectory, state, action)

    if output_path.exists():
        if not resume:
            raise FileExistsError(output_path)
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("num_frames") != len(state) or existing.get("time") != f"episode_{index:05d}":
            raise ValueError(f"resume output is incomplete or mismatched: {output_path}")
        digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
        return {"episode_index": index, "num_frames": len(state), "output_sha256": digest, "status": "resumed-existing"}

    state_poses = {
        "left": fk.batch(state[:, :5]),
        "right": fk.batch(state[:, 6:11]),
    }
    action_poses = {
        "left": fk.batch(action[:, :5]),
        "right": fk.batch(action[:, 6:11]),
    }
    for bucket, poses in (("proprios", state_poses), ("actions", action_poses)):
        for side, (positions, rotations) in poses.items():
            trajectory[bucket][f"{side}_ee_pos"] = positions.tolist()
            trajectory[bucket][f"{side}_ee_rotm"] = rotations.reshape(len(state), 9).tolist()
    payload = json.dumps(trajectory, separators=(",", ":"), allow_nan=False).encode("utf-8")
    temp_path = output_path.with_name(output_path.name + ".tmp")
    with temp_path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, output_path)
    return {
        "episode_index": index,
        "num_frames": len(state),
        "source_parquet": str(data_path),
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "status": "converted",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--old-json-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    for required in (args.dataset_dir / "meta/info.json", args.urdf, args.old_json_dir):
        if not required.exists():
            raise FileNotFoundError(required)
    info_path = args.dataset_dir / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    module = load_reconverter(RECONVERTER)
    expected_features = module.ARM_JOINT_FEATURES
    for key in ("observation.state", "action"):
        if info["features"][key]["names"] != expected_features:
            raise ValueError(f"unexpected {key} joint order in source metadata")
    episodes = get_rows(args.dataset_dir)
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError(f"metadata has {len(episodes)} episodes, info.json says {info['total_episodes']}")
    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"output directory is non-empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fk = module.URDFFK(args.urdf)
    results = []
    total = len(episodes)
    for done, episode in enumerate(episodes, 1):
        index = int(episode["episode_index"])
        if index != done - 1:
            raise ValueError(f"episode index sequence has a gap: expected {done - 1}, got {index}")
        result = convert_one(
            index, episode, args.dataset_dir, info, args.old_json_dir,
            args.output_dir, fk, args.resume,
        )
        results.append(result)
        print(
            f"[{done}/{total}] episode={index:05d} frames={result['num_frames']} "
            f"status={result['status']} sha256={result['output_sha256']}",
            flush=True,
        )

    manifest = {
        "source_dataset": str(args.dataset_dir),
        "source_info_sha256": hashlib.sha256(info_path.read_bytes()).hexdigest(),
        "source_total_episodes": int(info["total_episodes"]),
        "converted_episodes": len(results),
        "converted_frames": sum(item["num_frames"] for item in results),
        "official_urdf": str(args.urdf),
        "official_urdf_sha256": hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
        "ee_frame": "gripper_frame_link",
        "joint_limits_used_for_filtering_or_rejection": False,
        "legacy_source_json_dir": str(args.old_json_dir),
        "outputs": results,
    }
    manifest_path = args.output_dir / "conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("converted_episodes", "converted_frames", "ee_frame", "joint_limits_used_for_filtering_or_rejection")}, indent=2))


if __name__ == "__main__":
    main()
