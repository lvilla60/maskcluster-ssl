"""Build a checkpoint-identified comparison table from completed evaluations."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="metrics.json files from evaluate.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-partial", action="store_true", help="Only for fixture or draft tables")
    args = parser.parse_args()
    rows = []
    seen = set()
    common_protocol = None
    for path in args.reports:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        protocol = report.get("protocol")
        if not isinstance(protocol, dict) or protocol.get("version") != 1:
            raise ValueError(f"Missing or unsupported evaluation protocol: {path}")
        if (report.get("split") != protocol.get("split")
                or report.get("data", {}).get("file") != protocol.get("source_file")
                or report.get("data", {}).get("gallery_sha256") != protocol.get("gallery_sha256")):
            raise ValueError(f"Inconsistent dataset provenance: {path}")
        if common_protocol is None:
            common_protocol = protocol
        elif protocol != common_protocol:
            raise ValueError(f"Incompatible evaluation protocol or dataset: {path}")
        if report["config"]["model"]["variant"] != report["variant"]:
            raise ValueError(f"Checkpoint variant and report variant differ: {path}")
        key = (report["variant"], report["encoder"])
        if key in seen:
            raise ValueError(f"Duplicate variant/encoder: {key}")
        seen.add(key)
        metrics = report["metrics"]
        rows.append({"variant": key[0], "encoder": key[1], "run_id": report["run_id"],
                     "checkpoint": report["checkpoint"], "split": report["split"],
                     "dataset_sha256": report["data"]["file"]["sha256"],
                     "gallery_sha256": protocol["gallery_sha256"],
                     "config": json.dumps(report["config"], sort_keys=True),
                     "original_mAP": metrics["original"]["mAP"],
                     "low_resolution_mAP": metrics["low_resolution"]["mAP"],
                     "mAP_drop_percentage_points": metrics["mAP_drop_percentage_points"],
                     "original_rank1": metrics["original"]["rank1"],
                     "low_resolution_rank1": metrics["low_resolution"]["rank1"]})
    expected = {(variant, encoder) for variant in ("baseline", "full", "no_mim")
                for encoder in ("teacher", "student")}
    if not args.allow_partial and seen != expected:
        raise ValueError(f"Comparison requires all six variant/encoder reports; missing: {sorted(expected - seen)}")
    if len({row["split"] for row in rows}) != 1:
        raise ValueError("Comparison reports must use the same evaluation split")
    with Path(args.output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["variant"], row["encoder"])))


if __name__ == "__main__":
    main()
