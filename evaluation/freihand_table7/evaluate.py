"""Evaluate the mesh metric in Table 7 using all 3,960 FreiHAND v2 test images."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.linalg import orthogonal_procrustes


COUNT = 3960
VERTICES = 778


def read_reference(archive):
    with ZipFile(archive) as z:
        with z.open("evaluation_verts.json") as f:
            return json.load(f)


def read_submission(path):
    with path.open(encoding="utf-8") as f:
        submission = json.load(f)
    if len(submission) != 2 or any(len(part) != COUNT for part in submission):
        raise ValueError("Expected a complete FreiHAND [joints, vertices] submission.")
    return submission[1]


def vertices_from_parameters(path, mano_file):
    # The original HaMeR wrapper augments only joints. The table compares vertices.
    import torch
    from smplx import MANOLayer

    if not mano_file.is_file() or mano_file.name != "MANO_RIGHT.pkl":
        raise ValueError("Expected the official right-hand MANO_RIGHT.pkl file.")
    layer = MANOLayer(model_path=str(mano_file), is_rhand=True).eval()
    with np.load(path, allow_pickle=False) as data:
        if data["imgname"].tolist() != [f"{i:08d}.jpg" for i in range(COUNT)]:
            raise ValueError("Image order differs from the FreiHAND evaluation split.")
        parameters = {}
        for name, shape in (("global_orient", (1, 3, 3)), ("hand_pose", (15, 3, 3)), ("betas", (10,))):
            value = data[name]
            if value.shape != (COUNT, *shape) or not np.isfinite(value).all():
                raise ValueError(f"Invalid HaMeR output: {name}")
            parameters[name] = torch.from_numpy(value).float()
    with torch.inference_mode():
        for start in range(0, COUNT, 64):
            batch = {name: value[start:start + 64] for name, value in parameters.items()}
            yield from layer(**batch, pose2rot=False).vertices.numpy()


def mesh_error_mm(reference, estimate):
    reference, estimate = (np.asarray(x, dtype=np.float64) for x in (reference, estimate))
    for mesh in (reference, estimate):
        if mesh.shape != (VERTICES, 3) or not np.isfinite(mesh).all():
            raise ValueError("Expected a finite 778-by-3 mesh in metres.")
    true_center = reference.mean(axis=0)
    true = reference - true_center
    pred = estimate - estimate.mean(axis=0)
    true_norm = np.linalg.norm(true) + 1e-8
    true /= true_norm
    pred /= np.linalg.norm(pred) + 1e-8
    rotation, scale = orthogonal_procrustes(true, pred)
    aligned = (pred @ rotation.T) * scale * true_norm + true_center
    return float(np.linalg.norm(aligned - reference, axis=1).mean() * 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Official FreiHAND_pub_v2_eval.zip")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--submission", type=Path, help="Model output in the FreiHAND JSON format")
    source.add_argument("--parameters", type=Path, help="HaMeR image IDs and pose/shape matrices (.npz)")
    parser.add_argument("--mano", type=Path, help="Path to your MANO_RIGHT.pkl for decoding parameters")
    parser.add_argument("--output", type=Path, help="Save the metric report as JSON")
    args = parser.parse_args()
    if args.parameters and not args.mano:
        parser.error("--parameters requires --mano")
    reference = read_reference(args.reference)
    if len(reference) != COUNT:
        raise ValueError("The reference must cover exactly 3,960 test images.")
    if args.submission:
        predicted = read_submission(args.submission)
    else:
        predicted = vertices_from_parameters(args.parameters, args.mano)
    errors = [mesh_error_mm(gt, pred) for gt, pred in zip(reference, predicted, strict=True)]
    result = {"images": len(errors), "vertices_per_image": VERTICES, "pa_mpvpe_mm": float(np.mean(errors))}
    report = json.dumps(result, indent=2, allow_nan=False)
    print(report)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
