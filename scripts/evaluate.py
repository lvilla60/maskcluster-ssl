"""Evaluate one checkpoint encoder on validation or reserved final testing."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maskcluster_ssl.data.hdf5 import ReIDH5Dataset
from maskcluster_ssl.data.provenance import evaluation_provenance
from maskcluster_ssl.data.transforms import EvalTransform, IMAGENET_MEAN, IMAGENET_STD
from maskcluster_ssl.evaluation.extract import extract_embeddings
from maskcluster_ssl.evaluation.reid import evaluate_reid, summarize_conditions
from maskcluster_ssl.evaluation.report import export_cases, ranking_records, write_metrics_csv, write_rankings
from maskcluster_ssl.models import MaskClusterSSL
from maskcluster_ssl.training.runtime import select_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="Explicit validation or final testing HDF5 path")
    parser.add_argument("--split", required=True, choices=("validation", "testing"))
    parser.add_argument("--encoder", required=True, choices=("teacher", "student"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ranking-limit", type=int, default=100)
    args = parser.parse_args()
    if args.split not in Path(args.data).name.lower():
        parser.error("--split must match the input HDF5 filename")
    data_provenance = evaluation_provenance(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    device = select_device(args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    m = config["model"]
    model = MaskClusterSSL(variant=m["variant"], image_size=m["image_size"], patch_size=m["patch_size"],
                           embed_dim=m["embed_dim"], depth=m["depth"], num_heads=m["num_heads"],
                           mlp_ratio=m["mlp_ratio"], prototypes=m["prototypes"],
                           reconstruction_layers=m["reconstruction_layers"],
                           attention_mode=m.get("attention_mode", "standard"))
    model.load_state_dict(state["model"])
    encoder = getattr(model.to(device), args.encoder).eval()
    transform = EvalTransform(m["image_size"])
    protocol = {
        "version": 1,
        "split": args.split,
        "source_file": data_provenance["file"],
        "gallery_sha256": data_provenance["gallery_sha256"],
        "groups": data_provenance["groups"],
        "preprocessing": {"input_size": list(transform.size), "interpolation": "bicubic",
                          "align_corners": False, "antialias": True,
                          "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "retrieval": {"embedding": "L2-normalized CLS", "similarity": "cosine",
                      "junk_person_id": -1, "exclude_same_identity_camera": True,
                      "ap": "mean precision at relevant ranks", "tie_break": "gallery order"},
    }
    gallery = ReIDH5Dataset(args.data, "gallery", transform=transform)
    queries = [ReIDH5Dataset(args.data, "query", low_resolution=low, transform=transform) for low in (False, True)]
    def loader(dataset):
        return DataLoader(dataset, batch_size=config["evaluation"]["batch_size"], shuffle=False,
                          num_workers=0, pin_memory=device.type == "cuda")
    try:
        gallery_vectors, gallery_meta = extract_embeddings(encoder, loader(gallery), device)
        results = []
        cases = {}
        for name, dataset in zip(("original", "low_resolution"), queries):
            vectors, metadata = extract_embeddings(encoder, loader(dataset), device)
            results.append(evaluate_reid(vectors, gallery_vectors, metadata["person_id"],
                                         gallery_meta["person_id"], metadata["camera_id"],
                                         gallery_meta["camera_id"],
                                         query_block_size=config["evaluation"]["query_block_size"]))
            records = list(ranking_records(vectors, gallery_vectors, metadata, gallery_meta,
                                           limit=args.ranking_limit))
            write_rankings(output / f"rankings_{name}.jsonl", records)
            cases[name] = export_cases(args.data, name, records, output / "cases")
        metrics = summarize_conditions(*results)
        report = {"split": args.split, "checkpoint": str(Path(args.checkpoint).resolve()),
                  "run_id": Path(args.checkpoint).resolve().parent.name,
                  "variant": m["variant"], "encoder": args.encoder,
                  "config": config, "data": data_provenance, "protocol": protocol,
                  "metrics": metrics, "cases": cases,
                  "ranking_limit": args.ranking_limit}
        (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_metrics_csv(output / "metrics.csv", report["run_id"], m["variant"], args.encoder, metrics)
        print(json.dumps({"metrics": metrics, "cases": cases}, indent=2))
    finally:
        gallery.close()
        for dataset in queries:
            dataset.close()


if __name__ == "__main__":
    main()
