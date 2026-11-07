# Real-data validation record

Run date: 2026-11-07. Published commit: `56381a2b2ee0fc49e41c52a3f07211b955f51ad9`.

Target: the RW-RL dual-SO101 `so101_fold_clothes_left_stack_right` dataset on the remote conversion server.

## Inputs and implementation identity

- LeRobot source: 449 episodes; metadata reports 5,085,540 frames.
- Calibrated URDF SHA-256: `3a65d2d35e68a8d2f0c2cc176d19b884506543c93ba72980145b80abe276022c`.
- `scripts/convert_rwrl_so101_official_urdf.py`: `f93cbb5ba71de307b8ec93e535bebc69ae97d7bdd834897c814986a28ca88bb9`.
- `scripts/official_urdf_reconvert_one_episode.py`: `49f6943a86f606a6e4999422ae2f9e9cfd8d0fcca2959f33c6d874c0c7349bb4`.
- `scripts/validate_official_urdf_conversion.py`: `5ec5c0e405ff08ac8d9513d50a35925e571564adc0e178424f8bd1ea116a2604`.

The server cloned the public GitHub repository at the commit above and ran those checked-in scripts. The only portability edit from the server copy was making machine-specific input paths explicit required CLI arguments; the conversion/FK logic was not changed.

## Fresh episode regression

Using the release converter on source `episode_00000` produced 14,655 frames. The new output SHA-256 was `aad6603959a4a2a423175287574c7a49d048c0a3810539a6aa9291500068252e`; a byte-for-byte `cmp` against the episode from the prior validated full conversion passed.

## Fresh full conversion

The release converter was then run from scratch over the full source dataset into a new directory, without overwriting either existing conversion:

`/mnt/c/xr1_rwrl_json_release_56381a2/so101_fold_clothes_left_stack_right`

Result: 449/449 episodes converted, 5,085,540 frames, all episode indices contiguous, EE frame `gripper_frame_link`, and no URDF joint-limit filtering. New manifest SHA-256: `4dc0b814bd45ca4de82f69b25263c04a7dd19ae1353440e4159602aae68c146a`.

The new manifest's per-episode output SHA-256 values were compared with the prior full-conversion manifest: all 449 hashes matched (0 mismatches), and the URDF hashes matched.

## Full validation of the fresh output

The checked-in validator returned `PASS` over the fresh output:

- 449 episodes and 5,085,540 frames match source metadata.
- Deep old/new JSON comparison covered indices 0–448.
- All 1,347 referenced video files exist.
- Only the four intended left/right EE position/rotation arrays differ from the legacy JSON; joint arrays, grippers, schema, instructions, and video references are preserved.
- Maximum rotation orthogonality error: `1.6653345369377348e-15`.
- Maximum determinant error from 1: `1.5543122344752192e-15`.

This is a full real-data rerun and validation, not a synthetic CI test or a sample-only claim. The generated dataset was written to the server path above, validated, and then removed on 2026-11-08 because every episode output hash matched the already-retained full conversion. It can be regenerated using the commands in `README.md`. Dataset contents are not included in this GitHub repository.
