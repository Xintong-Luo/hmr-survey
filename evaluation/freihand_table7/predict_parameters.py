"""Run the two public HaMeR inference submodules on the ordered FreiHAND split."""

import argparse
import importlib
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import torch


def load_submodules(repo):
    # Import the original source, allowing the encoder and head to be used independently
    # from the optional parametric mesh layer, GUI renderer, and training framework.
    for relative in ["hamer", "hamer.datasets", "hamer.models", "hamer.utils"]:
        path = repo.joinpath(*relative.split("."))
        mod = types.ModuleType(relative)
        mod.__path__ = [str(path)]
        sys.modules[relative] = mod
    dataset_type = importlib.import_module("hamer.datasets.image_dataset").ImageDataset
    get_config = importlib.import_module("hamer.configs").get_config
    build_backbone = importlib.import_module("hamer.models.backbones").create_backbone
    build_head = importlib.import_module("hamer.models.heads.mano_head").build_mano_head
    return get_config, dataset_type, build_backbone, build_head


class Images:
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        return row["img"], row["idx"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--rgb", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for the full-set run")
    torch.set_num_threads(2)
    get_config, dataset_type, create_backbone, create_head = load_submodules(args.repo)
    cfg = get_config(str(args.config))
    print("Reading official weights", flush=True)
    # Trusted official file is externally hash-verified. The memory mapped loader
    # does not access mesh-layer data. Only the two network prefixes are selected.
    weights = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=True
    )["state_dict"]
    backbone_weights = {
        k.removeprefix("backbone."): v
        for k, v in weights.items()
        if k.startswith("backbone.")
    }
    head_weights = {
        k.removeprefix("mano_head."): v
        for k, v in weights.items()
        if k.startswith("mano_head.")
    }
    del weights
    cfg.defrost()
    if cfg.MODEL.BACKBONE.TYPE != "vit" or cfg.MODEL.IMAGE_SIZE != 256:
        raise ValueError("Unexpected checkpoint configuration")
    if "BBOX_SHAPE" not in cfg.MODEL:
        cfg.MODEL.BBOX_SHAPE = [192, 256]
    if "PRETRAINED_WEIGHTS" in cfg.MODEL.BACKBONE:
        cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
    with tempfile.TemporaryDirectory() as temp:
        # These predictor initialization vectors are also stored verbatim in the public head.
        cfg.MANO.MEAN_PARAMS = str(Path(temp) / "init.npz")
        np.savez(
            cfg.MANO.MEAN_PARAMS,
            pose=head_weights["init_hand_pose"].numpy()[0],
            shape=head_weights["init_betas"].numpy()[0],
            cam=head_weights["init_cam"].numpy()[0],
        )
        cfg.freeze()
        backbone = create_backbone(cfg)
        head = create_head(cfg)
    backbone.load_state_dict(backbone_weights, strict=True, assign=True)
    head.load_state_dict(head_weights, strict=True, assign=True)
    del backbone_weights, head_weights
    backbone.cuda()
    backbone.eval()
    head.cuda()
    head.eval()
    dataset = dataset_type(
        cfg, str(args.meta), str(args.rgb), train=False, rescale_factor=-1
    )
    ids = np.array(dataset.imgname).astype(str)
    if len(ids) != 3960 or ids.tolist() != [f"{i:08d}.jpg" for i in range(3960)]:
        raise ValueError("Unexpected FreiHAND sample IDs or coverage")
    loader = torch.utils.data.DataLoader(
        Images(dataset),
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True,
    )
    print(
        "Ready",
        torch.__version__,
        torch.cuda.get_device_name(),
        len(dataset),
        flush=True,
    )
    arrays = {}
    counter = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for images, indexes in loader:
            if not np.array_equal(
                indexes.numpy(), np.arange(counter, counter + len(images))
            ):
                raise RuntimeError("Sample order changed")
            features = backbone(images.cuda(non_blocking=True)[:, :, :, 32:-32])
            parameters, cameras, _ = head(features)
            for k, v in {**parameters, "camera": cameras}.items():
                val = v.cpu().numpy()
                if not np.isfinite(val).all():
                    raise ValueError(f"{k} nonfinite")
                if k not in arrays:
                    arrays[k] = np.empty(
                        (len(dataset),) + val.shape[1:], dtype=val.dtype
                    )
                arrays[k][counter : counter + len(images)] = val
            counter += len(images)
            if counter == len(images) or counter % 256 == 0 or counter == len(dataset):
                print(
                    "Processed",
                    counter,
                    "seconds",
                    round(time.perf_counter() - start, 1),
                    flush=True,
                )
    assert counter == len(ids)
    np.savez_compressed(args.output, imgname=ids, **arrays)
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
