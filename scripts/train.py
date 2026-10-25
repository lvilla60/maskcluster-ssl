"""Train one MaskCluster_SSL variant using training and validation only."""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from maskcluster_ssl.data.audit import within_root
from maskcluster_ssl.data.hdf5 import TrainingH5Dataset
from maskcluster_ssl.data.provenance import training_signatures
from maskcluster_ssl.data.sampling import EpochSampler
from maskcluster_ssl.data.transforms import MultiCropTransform
from maskcluster_ssl.losses import LossWeights, MaskClusterLoss
from maskcluster_ssl.models import MaskClusterSSL, VARIANTS
from maskcluster_ssl.training.runtime import seed_everything, select_device, write_run_metadata
from maskcluster_ssl.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--variant", choices=sorted(VARIANTS))
    parser.add_argument("--resume")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = within_root(args.config, ROOT)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.variant:
        config["model"]["variant"] = args.variant
    training_path = within_root(config["data"]["training"], ROOT)
    validation_path = within_root(config["data"]["validation"], ROOT)
    config["data"]["training"] = str(training_path)
    config["data"]["validation"] = str(validation_path)
    data_signatures = training_signatures(training_path, validation_path)
    seed_everything(int(config["seed"]))
    device = select_device(config["runtime"]["device"])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{config['model']['variant']}_{uuid.uuid4().hex[:8]}"
    run_dir = within_root(Path(config["output_dir"]) / run_id, ROOT)
    write_run_metadata(run_dir, config, ROOT, data_signatures=data_signatures)

    model_config = config["model"]
    transform = MultiCropTransform(
        global_size=model_config["image_size"],
        local_size=model_config["local_size"],
        local_crops=model_config["local_crops"],
        patch_size=model_config["patch_size"],
        mask_ratio=model_config["mask_ratio"],
        seed=config["seed"],
        **model_config.get("photometric", {}),
    )
    dataset = TrainingH5Dataset(training_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        sampler=EpochSampler(dataset, config["seed"]),
        drop_last=True,
        num_workers=config["runtime"]["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=config["runtime"]["num_workers"] > 0,
    )
    model = MaskClusterSSL(
        variant=model_config["variant"],
        image_size=model_config["image_size"],
        patch_size=model_config["patch_size"],
        embed_dim=model_config["embed_dim"],
        depth=model_config["depth"],
        num_heads=model_config["num_heads"],
        mlp_ratio=model_config["mlp_ratio"],
        prototypes=model_config["prototypes"],
        reconstruction_layers=model_config["reconstruction_layers"],
        attention_mode=model_config.get("attention_mode", "visible"),
    )
    weights = LossWeights(**config["loss"]["weights"])
    loss_fn = MaskClusterLoss(
        student_temperature=config["loss"]["student_temperature"],
        teacher_temperature=config["loss"]["teacher_temperature"],
        me_max_temperature=config["loss"]["me_max_temperature"],
        patch_size=model_config["patch_size"],
        weights=weights,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    trainer = Trainer(model, loss_fn, optimizer, loader, config, run_dir, device,
                      data_signatures=data_signatures)
    if args.resume:
        trainer.resume(within_root(args.resume, ROOT))
    summary = trainer.fit()
    print(yaml.safe_dump(summary, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
