"""Export bounded diagnostic records without treating steps as epochs."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from maskcluster_ssl.evaluation.analysis import write_table, plot_series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    if not report.get("diagnostic_only"):
        raise ValueError("Expected an explicitly diagnostic smoke report")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for variant, result in report["variants"].items():
        for encoder in ("teacher", "student"):
            metrics = result["validation_subset"][encoder]
            row = {"run_id": report["run_id"], "purpose": "smoke", "variant": variant,
                   "encoder": encoder, "model": f"{variant}/{encoder}",
                   "checkpoint": result["checkpoint"], "steps": result["optimization_steps"],
                   "mAP_drop_percentage_points": metrics["mAP_drop_percentage_points"]}
            for condition in ("original", "low_resolution"):
                row.update({f"{condition}_{key}": value for key, value in metrics[condition].items()})
            rows.append(row)
    write_table(output / "diagnostic_metrics.csv", rows)
    write_table(output / "diagnostic_losses.csv", [{"variant": v, **r["losses"]} for v, r in report["variants"].items()])
    plot_series(output / "diagnostic_metrics.png", rows, "model", ["original_mAP", "low_resolution_mAP"], "SMOKE ONLY: four validation queries, one step per variant")
    plot_series(output / "diagnostic_drop.png", rows, "model", ["mAP_drop_percentage_points"], "SMOKE ONLY: mAP drop in percentage points")
    (output / "source.json").write_text(json.dumps({"report": str(Path(args.report).resolve()), "diagnostic_only": True}, indent=2))


if __name__ == "__main__":
    main()
