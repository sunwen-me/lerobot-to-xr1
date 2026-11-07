#!/usr/bin/env python3
"""Recompute one dual-SO101 LeRobot episode with the official SO-101 URDF.

This deliberately does not modify the legacy converter or its output. It writes
one separate XR-1 JSON episode and a compact audit report.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow.parquet as pq

JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
ARM_JOINT_FEATURES = [
    "left_shoulder_pan.pos",
    "left_shoulder_lift.pos",
    "left_elbow_flex.pos",
    "left_wrist_flex.pos",
    "left_wrist_roll.pos",
    "left_gripper.pos",
    "right_shoulder_pan.pos",
    "right_shoulder_lift.pos",
    "right_elbow_flex.pos",
    "right_wrist_flex.pos",
    "right_wrist_roll.pos",
    "right_gripper.pos",
]


def vector3(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    values = np.fromstring(text, sep=" ", dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"expected 3 URDF values, got {text!r}")
    return values


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def axis_rotation(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues rotations; returns one 3x3 rotation per angle."""
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    one_minus_c = 1.0 - c
    result = np.empty((len(angle), 3, 3), dtype=np.float64)
    result[:, 0, 0] = c + x * x * one_minus_c
    result[:, 0, 1] = x * y * one_minus_c - z * s
    result[:, 0, 2] = x * z * one_minus_c + y * s
    result[:, 1, 0] = y * x * one_minus_c + z * s
    result[:, 1, 1] = c + y * y * one_minus_c
    result[:, 1, 2] = y * z * one_minus_c - x * s
    result[:, 2, 0] = z * x * one_minus_c - y * s
    result[:, 2, 1] = z * y * one_minus_c + x * s
    result[:, 2, 2] = c + z * z * one_minus_c
    return result


class URDFFK:
    def __init__(self, urdf_path: Path):
        root = ET.parse(urdf_path).getroot()
        if root.tag != "robot":
            raise ValueError(f"not a URDF robot file: {urdf_path}")
        self.joint_by_child = {
            joint.find("child").attrib["link"]: joint
            for joint in root.findall("joint")
        }
        chain = []
        child = "gripper_frame_link"
        while child != "base_link":
            if child not in self.joint_by_child:
                raise ValueError(f"no URDF path from base_link to {child}")
            joint = self.joint_by_child[child]
            chain.append(joint)
            child = joint.find("parent").attrib["link"]
        self.chain = list(reversed(chain))
        revolute_names = [
            joint.attrib["name"]
            for joint in self.chain
            if joint.attrib["type"] in ("revolute", "continuous")
        ]
        if revolute_names != list(JOINT_ORDER):
            raise ValueError(
                f"unexpected base-to-gripper_frame joint order: {revolute_names}"
            )

    def batch(self, joints_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        joints_deg = np.asarray(joints_deg, dtype=np.float64)
        if joints_deg.ndim != 2 or joints_deg.shape[1] != 5:
            raise ValueError(f"expected N x 5 joint angles, got {joints_deg.shape}")
        count = len(joints_deg)
        transform = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
        for joint in self.chain:
            origin = joint.find("origin")
            origin_transform = np.eye(4, dtype=np.float64)
            if origin is not None:
                origin_transform[:3, :3] = rpy_matrix(
                    vector3(origin.attrib.get("rpy"))
                )
                origin_transform[:3, 3] = vector3(origin.attrib.get("xyz"))
            transform = transform @ origin_transform

            if joint.attrib["type"] not in ("revolute", "continuous"):
                continue
            joint_index = JOINT_ORDER.index(joint.attrib["name"])
            axis = vector3(joint.find("axis").attrib.get("xyz"), (0, 0, 1))
            rotations = axis_rotation(axis, np.deg2rad(joints_deg[:, joint_index]))
            motion = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
            motion[:, :3, :3] = rotations
            transform = transform @ motion

        positions = transform[:, :3, 3]
        rotations = transform[:, :3, :3]
        if not np.isfinite(transform).all():
            raise ValueError("FK produced non-finite transforms")
        return positions, rotations


def rotation_error_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = left.T @ right
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def episode_data(dataset_dir: Path, episode_index: int):
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    state_names = info["features"]["observation.state"]["names"]
    action_names = info["features"]["action"]["names"]
    if state_names != ARM_JOINT_FEATURES or action_names != ARM_JOINT_FEATURES:
        raise ValueError("source state/action channels do not match the expected dual-SO101 order")

    meta_files = sorted((dataset_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    matches = []
    for path in meta_files:
        table = pq.read_table(path)
        matches.extend(
            row for row in table.to_pylist()
            if int(row["episode_index"]) == episode_index
        )
    if len(matches) != 1:
        raise ValueError(f"expected one metadata row for episode {episode_index}, got {len(matches)}")
    episode = matches[0]
    relative = info["data_path"].format(
        chunk_index=int(episode["data/chunk_index"]),
        file_index=int(episode["data/file_index"]),
    )
    data_path = dataset_dir / relative
    table = pq.read_table(
        data_path,
        columns=["episode_index", "frame_index", "observation.state", "action"],
    )
    mask = np.asarray(table["episode_index"].to_numpy()) == episode_index
    frame_index = np.asarray(table["frame_index"].to_numpy())[mask]
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)[mask]
    action = np.asarray(table["action"].to_pylist(), dtype=np.float64)[mask]
    expected = int(episode["length"])
    if state.shape != (expected, 12) or action.shape != (expected, 12):
        raise ValueError(
            f"episode {episode_index}: expected ({expected}, 12), got {state.shape}, {action.shape}"
        )
    if not np.array_equal(frame_index, np.arange(expected)):
        raise ValueError("episode frame indices are not contiguous from zero")
    return info, episode, data_path, state, action


def build_trajectory(
    episode_index: int,
    episode: dict,
    state: np.ndarray,
    action: np.ndarray,
    state_poses: dict,
    action_poses: dict,
    schema_reference: dict,
) -> dict:
    count = len(state)
    zero1 = [[0.0] for _ in range(count)]
    zero3 = [[0.0, 0.0, 0.0] for _ in range(count)]
    result = {
        "trajectory_type": schema_reference.get("trajectory_type", "ongoing"),
        "time": f"episode_{episode_index:05d}",
        "num_frames": count,
        "instruction": copy.deepcopy(schema_reference["instruction"]),
        # Reuse only the legacy file's schema/video references, never its FK labels.
        "observations": copy.deepcopy(schema_reference["observations"]),
        "proprios": {
            "left_ee_pos": state_poses["left"][0].tolist(),
            "left_ee_rotm": state_poses["left"][1].reshape(count, 9).tolist(),
            "left_arm_joint": state[:, :5].tolist(),
            "left_gripper_pos": state[:, 5:6].tolist(),
            "right_ee_pos": state_poses["right"][0].tolist(),
            "right_ee_rotm": state_poses["right"][1].reshape(count, 9).tolist(),
            "right_arm_joint": state[:, 6:11].tolist(),
            "right_gripper_pos": state[:, 11:12].tolist(),
            "waist_pos": zero1,
        },
        "actions": {
            "left_ee_pos": action_poses["left"][0].tolist(),
            "left_ee_rotm": action_poses["left"][1].reshape(count, 9).tolist(),
            "left_gripper_pos": action[:, 5:6].tolist(),
            "right_ee_pos": action_poses["right"][0].tolist(),
            "right_ee_rotm": action_poses["right"][1].reshape(count, 9).tolist(),
            "right_gripper_pos": action[:, 11:12].tolist(),
            "waist_pos": zero1,
            "base_vel": zero3,
        },
    }
    if int(episode["length"]) != result["num_frames"]:
        raise AssertionError("output length does not match episode metadata")
    return result


def make_report(
    urdf_path: Path,
    data_path: Path,
    episode_index: int,
    state: np.ndarray,
    action: np.ndarray,
    state_poses: dict,
    action_poses: dict,
    schema_reference: dict,
    chunk_starts: list[int],
) -> dict:
    report = {
        "episode_index": episode_index,
        "num_frames": len(state),
        "source_data": str(data_path),
        "source_joint_order": ARM_JOINT_FEATURES,
        "joint_units": "degrees; gripper values passed through unchanged",
        "urdf": str(urdf_path),
        "urdf_sha256": hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
        "base_frame": "base_link",
        "ee_frame": "gripper_frame_link",
        "arm_joint_chain": list(JOINT_ORDER),
        "output_json": None,
        "old_frame_comparison_caveat": (
            "Legacy and official conversion use different FK/frame contracts. Raw pose differences are not physical pose errors unless their base and tool frames are first registered."
        ),
        "old_vs_official": {},
        "chunks": {},
    }
    for side, gripper_index in (("left", 5), ("right", 11)):
        old_state_pos = np.asarray(
            schema_reference["proprios"][f"{side}_ee_pos"], dtype=np.float64
        )
        old_state_rot = np.asarray(
            schema_reference["proprios"][f"{side}_ee_rotm"], dtype=np.float64
        ).reshape(-1, 3, 3)
        old_target_pos = np.asarray(
            schema_reference["actions"][f"{side}_ee_pos"], dtype=np.float64
        )
        old_target_rot = np.asarray(
            schema_reference["actions"][f"{side}_ee_rotm"], dtype=np.float64
        ).reshape(-1, 3, 3)
        new_state_pos, new_state_rot = state_poses[side]
        new_target_pos, new_target_rot = action_poses[side]
        if len(old_state_pos) != len(state):
            raise ValueError(f"legacy schema reference has mismatched {side} trajectory length")
        report["old_vs_official"][side] = {
            "absolute_target_position_mean_error_mm": float(
                np.mean(np.linalg.norm(old_target_pos - new_target_pos, axis=1)) * 1000.0
            ),
            "absolute_target_position_max_error_mm": float(
                np.max(np.linalg.norm(old_target_pos - new_target_pos, axis=1)) * 1000.0
            ),
            "absolute_target_rotation_mean_geodesic_error_deg": float(
                np.mean([
                    rotation_error_deg(old_target_rot[i], new_target_rot[i])
                    for i in range(len(state))
                ])
            ),
            "gripper_source_passthrough_max_abs_error": float(
                np.max(
                    np.abs(
                        np.asarray(
                            schema_reference["actions"][f"{side}_gripper_pos"],
                            dtype=np.float64,
                        ).reshape(-1)
                        - action[:, gripper_index]
                    )
                )
            ),
        }
        for start in chunk_starts:
            if start < 0 or start + 30 > len(state):
                raise ValueError(f"30-frame chunk start {start} is outside this episode")
            new_local_pos = np.einsum(
                "ij,nj->ni",
                new_state_rot[start].T,
                new_target_pos[start : start + 30] - new_state_pos[start],
            )
            old_local_pos = np.einsum(
                "ij,nj->ni",
                old_state_rot[start].T,
                old_target_pos[start : start + 30] - old_state_pos[start],
            )
            old_first, new_first = old_local_pos[0], new_local_pos[0]
            norm_product = np.linalg.norm(old_first) * np.linalg.norm(new_first)
            direction_cosine = (
                float(np.dot(old_first, new_first) / norm_product)
                if norm_product > 1e-12
                else None
            )
            new_target_chunk = new_target_pos[start : start + 30]
            gripper = action[start : start + 30, gripper_index]
            report["chunks"].setdefault(str(start), {})[side] = {
                "new_target_position_first_mm": (new_target_chunk[0] * 1000.0).tolist(),
                "new_target_position_last_mm": (new_target_chunk[-1] * 1000.0).tolist(),
                "new_target_path_length_mm": float(
                    np.linalg.norm(np.diff(new_target_chunk, axis=0), axis=1).sum() * 1000.0
                ),
                "new_target_endpoint_displacement_mm": float(
                    np.linalg.norm(new_target_chunk[-1] - new_target_chunk[0]) * 1000.0
                ),
                "old_vs_new_first_local_delta_direction_cosine": direction_cosine,
                "old_vs_new_local_delta_xyz_mae_mm": (
                    np.mean(np.abs(old_local_pos - new_local_pos), axis=0) * 1000.0
                ).tolist(),
                "old_vs_new_target_position_mean_distance_mm": float(
                    np.mean(
                        np.linalg.norm(
                            old_target_pos[start : start + 30]
                            - new_target_pos[start : start + 30],
                            axis=1,
                        )
                    )
                    * 1000.0
                ),
                "old_vs_new_target_rotation_mean_geodesic_deg": float(
                    np.mean([
                        rotation_error_deg(old_target_rot[i], new_target_rot[i])
                        for i in range(start, start + 30)
                    ])
                ),
                "gripper_first_last_min_max": [
                    float(gripper[0]),
                    float(gripper[-1]),
                    float(gripper.min()),
                    float(gripper.max()),
                ],
                "gripper_passthrough_max_abs_error": report["old_vs_official"][side][
                    "gripper_source_passthrough_max_abs_error"
                ],
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--schema-reference", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-starts", type=int, nargs="+", default=[0, 3000, 3090, 8910])
    args = parser.parse_args()

    for path in (args.dataset_dir / "meta" / "info.json", args.urdf, args.schema_reference):
        if not path.is_file():
            raise FileNotFoundError(path)
    report_path = args.output.with_suffix(".validation.json")
    if args.output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite an existing conversion or report")

    _info, episode, data_path, state, action = episode_data(
        args.dataset_dir, args.episode_index
    )
    fk = URDFFK(args.urdf)
    state_poses = {
        "left": fk.batch(state[:, :5]),
        "right": fk.batch(state[:, 6:11]),
    }
    action_poses = {
        "left": fk.batch(action[:, :5]),
        "right": fk.batch(action[:, 6:11]),
    }
    schema_reference = json.loads(args.schema_reference.read_text())
    if schema_reference.get("num_frames") != len(state):
        raise ValueError("schema reference episode length differs from raw source")
    trajectory = build_trajectory(
        args.episode_index, episode, state, action, state_poses, action_poses, schema_reference
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(trajectory, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    report = make_report(
        args.urdf, data_path, args.episode_index, state, action,
        state_poses, action_poses, schema_reference, args.chunk_starts,
    )
    report["output_json"] = str(args.output)
    report["chunk_starts"] = args.chunk_starts
    report["source_video_paths_reused_from_schema_reference"] = True
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "report": str(report_path), "frames": len(state)}, indent=2))


if __name__ == "__main__":
    main()
