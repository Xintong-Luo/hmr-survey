"""Extract paired pelvis centres from FreeMoCap validation dataset v1.0."""

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def read_pelvis(path):
    table = pq.read_table(
        path,
        columns=["frame", "x", "y", "z"],
        filters=[("keypoint", "=", "hips_center"), ("trajectory", "=", "3d_xyz")],
    )
    frames = np.array(table["frame"])
    xyz_mm = np.column_stack([np.array(table[axis]) for axis in ("x", "y", "z")])
    order = np.argsort(frames)
    ids = frames[order]
    if not len(ids) or np.any(np.diff(ids) != 1):
        raise ValueError(f"{path}: missing or duplicate pelvis frame IDs")
    return ids, xyz_mm[order] / 1000.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Directory containing extracted sub-001, sub-002 and sub-003")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("pelvis_tracks.npz"))
    args = parser.parse_args()
    tracks = {}
    for subject in ("sub-001", "sub-002", "sub-003"):
        for trial in ("01", "02"):
            base = args.data / subject / f"task-treadmill_trial-{trial}"
            filename = "aligned_3d_data/freemocap_data_by_frame.parquet"
            ref_ids, reference = read_pelvis(base / "qualisys" / filename)
            est_ids, estimate = read_pelvis(base / "mediapipe" / filename)
            if not np.array_equal(ref_ids, est_ids):
                raise ValueError(f"{base}: reference and estimate do not share the same frame IDs")
            tracks[f"{subject}_trial-{trial}"] = np.column_stack((ref_ids, reference, estimate))
    np.savez_compressed(args.output, **tracks)


if __name__ == "__main__":
    main()
