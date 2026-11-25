"""Evaluate the paired pelvis trajectories in EMDB's public GLAMR example."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from urllib.request import urlopen

import numpy as np


SOURCE_REV = "9a4eab677181a3789bda7ba5c36ab8cff797380c"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/eth-ait/emdb/{SOURCE_REV}/"
    "assets/GLAMR/P9_80_seed1.pkl"
)
SOURCE_SHA256 = "9cb0e38ccf2b37b28458ce9fd5ad6974f3d81cd41d6fa28690779f730898d9e3"


def read_sample(local_file):
    if local_file is None:
        with urlopen(SOURCE_URL, timeout=60) as response:
            data = response.read()
    else:
        data = local_file.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError("File checksum differs from the pinned EMDB example.")
    sample = pickle.loads(data)
    target = sample["gt"][0]
    prediction = sample["person_data"][0]
    reference = np.asarray(target["root_trans"], dtype=np.float64)
    estimate = np.asarray(prediction["root_trans_world"], dtype=np.float64)
    frame_ids = np.asarray(prediction["frames"])
    n_frames = sample["seq_len"]
    if reference.shape != (n_frames, 3) or estimate.shape != reference.shape:
        raise ValueError("Expected two complete 3-D trajectories.")
    if not np.array_equal(frame_ids, np.arange(n_frames)):
        raise ValueError("Prediction frame IDs do not cover the reference in order.")
    if not np.all(prediction["exist_frames"]) or not np.all(target["visible"]):
        raise ValueError("The example expects a reference and prediction for every frame.")
    if not (np.isfinite(reference).all() and np.isfinite(estimate).all()):
        raise ValueError("Non-finite coordinate in input.")
    return reference, estimate


def score(reference, estimate):
    """Align the entire estimate by one proper rigid transform at unit scale."""
    ref_center = reference.mean(axis=0)
    est_center = estimate.mean(axis=0)
    ref_centered = reference - ref_center
    est_centered = estimate - est_center
    left, _, right_t = np.linalg.svd(est_centered.T @ ref_centered)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_t))
    rotation = left @ correction @ right_t
    aligned = est_centered @ rotation + ref_center
    errors_m = np.linalg.norm(aligned - reference, axis=1)
    length_m = np.linalg.norm(np.diff(reference, axis=0), axis=1).sum()
    if length_m <= 0:
        raise ValueError("RTE is undefined for a zero-length reference trajectory.")
    mean_error_m = float(errors_m.mean())
    result = {
        "method": "GLAMR (released prediction, seed 1)",
        "sequence": "P9/80_outdoor_walk_big_circle",
        "person": 0,
        "frame_count": len(reference),
        "first_frame": 0,
        "last_frame": len(reference) - 1,
        "reference": "gt[0].root_trans",
        "prediction": "person_data[0].root_trans_world",
        "alignment": "one rigid transform fitted to all frames; scale fixed at 1",
        "reference_path_length_m": float(length_m),
        "mean_aligned_root_distance_m": mean_error_m,
        "rte_percent": 100.0 * mean_error_m / float(length_m),
        "sample_sha256": SOURCE_SHA256,
    }
    return result, aligned, errors_m


def plot_result(path, reference, aligned, errors_m, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (trajectory, distance) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for points, name, color in [(reference, "Reference", "#00776D"),
                                (aligned, "GLAMR (aligned)", "#B15528")]:
        points = points - reference[0]
        trajectory.plot(points[:, 0], points[:, 2], label=name, color=color, linewidth=1.8)
        trajectory.scatter(points[0, 0], points[0, 2], color=color, s=22, zorder=3)
    trajectory.set_aspect("equal")
    trajectory.set(xlabel="Reference world X (m)", ylabel="Reference world Z (m)",
                   title="Root trajectories in the X–Z plane")
    trajectory.legend(frameon=False, loc="best", fontsize=9)
    distance.plot(np.arange(len(errors_m)), errors_m, color="#374E6B", linewidth=1.4)
    mean_m = result["mean_aligned_root_distance_m"]
    distance.axhline(mean_m, linestyle="--", linewidth=1, color="#B15528",
                     label=f"Mean {mean_m:.2f} m")
    distance.set(xlabel="Frame", ylabel="3-D root distance (m)",
                 title="Frame-wise error after alignment", ylim=(0, None))
    distance.legend(frameon=False)
    for ax in (trajectory, distance):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.18)
    fig.suptitle(f"EMDB P9 / 80 · GLAMR · {len(reference):,} frames · RTE {result['rte_percent']:.2f}%")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Use a local copy of the pinned official .pkl")
    parser.add_argument("--output", type=Path, help="Save the printed JSON to this file")
    parser.add_argument("--plot", type=Path, help="Save a diagnostic figure (requires matplotlib)")
    args = parser.parse_args()
    reference, estimate = read_sample(args.input)
    result, aligned, errors_m = score(reference, estimate)
    report = json.dumps(result, indent=2)
    print(report)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
    if args.plot:
        plot_result(args.plot, reference, aligned, errors_m, result)


if __name__ == "__main__":
    main()
