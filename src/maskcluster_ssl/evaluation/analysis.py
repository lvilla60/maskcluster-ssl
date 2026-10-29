"""Strict comparison contracts and plots derived from recorded values."""

import copy
import csv
import json
from pathlib import Path


def comparison_contract(report):
    config = copy.deepcopy(report["config"])
    config.pop("output_dir", None)
    config["model"].pop("variant", None)
    signatures = report.get("training_data_signatures")
    if signatures:
        config["data"] = signatures
    for key in ("batch_size", "query_block_size"):
        config.get("evaluation", {}).pop(key, None)
    return {"config": config, "selection_rule": report.get("selection_rule"),
            "source_sha256": (report.get("run_metadata") or {}).get("source_sha256"),
            "steps_per_epoch": (report.get("run_summary") or {}).get("steps_per_epoch"),
            "purpose": report.get("purpose", "unclassified")}


def write_table(path, rows):
    if not rows:
        raise ValueError("No recorded rows to export")
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def require_completed_budget(report):
    summary = report.get("run_summary") or {}
    planned = report["config"]["training"]["epochs"]
    if (summary.get("paused") is not False or summary.get("status") != "completed"
            or summary.get("planned_epochs") != planned
            or summary.get("completed_epochs") != planned
            or summary.get("global_step") != planned * summary.get("steps_per_epoch", -1)):
        raise ValueError("Comparison requires completed common training budgets")


def plot_series(path, rows, x, keys, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(10, 5), layout="constrained")
    for key in keys:
        points = [(row[x], row[key]) for row in rows if row.get(key) is not None]
        if points:
            axis.plot(*zip(*points), marker="o", label=key)
    axis.set(xlabel=x, title=title)
    axis.grid(alpha=0.25)
    if axis.lines:
        axis.legend(fontsize=7)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def export_history(run_dir, output):
    run_dir, output = Path(run_dir), Path(output)
    records = [json.loads(line) for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = []
    for record in records:
        if record.get("event") != "epoch":
            continue
        row = {key: record.get(key) for key in ("run_id", "variant", "seed", "epoch", "steps", "global_step", "train_seconds", "validation_seconds", "samples", "train_images_per_second", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")}
        row.update(record["train_summary"])
        for encoder, results in (record.get("validation") or {}).items():
            for condition in ("original", "low_resolution"):
                for metric, value in results.get(condition, {}).items():
                    row[f"{encoder}/{condition}/{metric}"] = value
            row[f"{encoder}/mAP_drop_percentage_points"] = results.get("mAP_drop_percentage_points")
        rows.append(row)
    if not rows:
        raise ValueError("No completed epoch records; smoke steps are not epochs")
    epochs = [r["epoch"] for r in rows]
    if len(set(epochs)) != len(epochs):
        raise ValueError("Duplicate epoch records; resolve resume lineage first")
    output.mkdir(parents=True, exist_ok=False)
    write_table(output / "epochs.csv", rows)
    keys = sorted(set().union(*(r.keys() for r in rows)))
    plot_series(output / "losses.png", rows, "epoch", [k for k in keys if k.startswith("loss/")], "Recorded epoch losses")
    plot_series(output / "validation.png", rows, "epoch", [k for k in keys if k.endswith(("/mAP", "/rank1", "/rank5", "/rank10"))], "Recorded validation metrics (0-1)")
    plot_series(output / "resolution_drop.png", rows, "epoch", [k for k in keys if k.endswith("percentage_points")], "Validation mAP drop (percentage points)")
    (output / "source.json").write_text(json.dumps({"run": str(run_dir.resolve()), "history": str((run_dir / "history.jsonl").resolve()), "synthetic": "synthetic" in str(run_dir).lower()}, indent=2), encoding="utf-8")
