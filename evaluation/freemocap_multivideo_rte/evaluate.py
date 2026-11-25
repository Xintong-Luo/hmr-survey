"""Score independently recorded FreeMoCap trials, then aggregate their RTE values."""

import argparse
import csv
from pathlib import Path

import numpy as np


def score_trial(trial, track):
    frames, reference, estimate = track[:, 0], track[:, 1:4], track[:, 4:7]
    if len(frames) < 3 or not np.all(np.diff(frames) == 1):
        raise ValueError(f"{trial}: expected consecutive frames in temporal order")
    if not np.isfinite(reference).all():
        raise ValueError(f"{trial}: reference is incomplete; path length is undefined")
    valid = np.isfinite(estimate).all(axis=1)
    if valid.sum() < 3:
        raise ValueError(f"{trial}: too few paired frames")
    target = reference[valid] - reference[valid].mean(axis=0)
    source = estimate[valid] - estimate[valid].mean(axis=0)
    u, _, vt = np.linalg.svd(source.T @ target)
    signs = np.ones(3)
    signs[-1] = np.sign(np.linalg.det(u @ vt))
    aligned = source @ ((u * signs) @ vt)
    errors = np.linalg.norm(aligned - target, axis=1)
    # The denominator uses the entire continuous reference trajectory.
    path_length = np.linalg.norm(np.diff(reference, axis=0), axis=1).sum()
    if path_length <= 0:
        raise ValueError(f"{trial}: RTE is undefined for a stationary reference")
    return {
        "trial": trial,
        "total_frames": len(frames),
        "scored_frames": int(valid.sum()),
        "reference_path_m": float(path_length),
        "mean_error_mm": float(errors.mean() * 1000),
        "rte_percent": float(errors.mean() / path_length * 100),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("pelvis_tracks.npz"))
    parser.add_argument("--csv", type=Path, help="Write the per-trial results to this path")
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as tracks:
        rows = [score_trial(name, tracks[name]) for name in sorted(tracks.files)]
    print("Trial               Scored/total    Path (m)    Mean error (mm)    RTE (%)")
    for row in rows:
        print(f"{row['trial']:<19} {row['scored_frames']:>5}/{row['total_frames']:<5}"
              f" {row['reference_path_m']:>10.3f} {row['mean_error_mm']:>18.3f}"
              f" {row['rte_percent']:>11.5f}")
    percentages = [r["rte_percent"] for r in rows]
    counts = [r["scored_frames"] for r in rows]
    print(f"Equal weight per trial: {np.mean(percentages):.7f}%")
    print(f"Weighted by scored frames: {np.average(percentages, weights=counts):.7f}%")
    if args.csv:
        with args.csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
