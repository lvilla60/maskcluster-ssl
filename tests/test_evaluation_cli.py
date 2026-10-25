import json
import subprocess
import sys
import uuid
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from maskcluster_ssl.models import MaskClusterSSL


def test_evaluation_cli_validation_fixture():
    root = Path(__file__).resolve().parents[1]
    folder = root / ".tmp" / f"eval_{uuid.uuid4().hex}"
    folder.mkdir(parents=True)
    path = folder / "validation.h5"
    images = np.full((3, 3, 32, 16), 100, dtype=np.uint8)
    images[1] = 150
    with h5py.File(path, "w") as handle:
        for name, count in (("query", 2), ("gallery", 3)):
            group = handle.create_group(name)
            group.create_dataset("images", data=images[:count])
            if name == "query":
                group.create_dataset("images_low_resolution", data=images[:count, :, ::2, ::2])
            group.create_dataset("person_ids", data=[1, 2][:count] if name == "query" else [1, 2, -1])
            group.create_dataset("camera_ids", data=[1, 1][:count] if name == "query" else [2, 2, 2])
            group.create_dataset("image_ids", data=[f"{name}-{i}".encode() for i in range(count)])
    config = yaml.safe_load((root / "configs" / "smoke_cpu.yaml").read_text(encoding="utf-8"))
    config["model"].update(variant="full", image_size=[32, 16], local_size=[16, 8])
    model = MaskClusterSSL(variant="full", image_size=(32, 16), patch_size=8,
                           embed_dim=32, depth=2, num_heads=4, mlp_ratio=2,
                           prototypes=16, reconstruction_layers=(1, 2))
    checkpoint = folder / "fixture.pt"
    torch.save({"model": model.state_dict(), "config": config}, checkpoint)
    output = folder / "report"
    command = [sys.executable, "-B", str(root / "scripts" / "evaluate.py"),
               "--checkpoint", str(checkpoint), "--data", str(path), "--split", "validation",
               "--encoder", "student", "--output", str(output)]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert Path(report["data"]["path"]) == path.resolve()
    assert report["data"]["groups"]["gallery"]["images"]["count"] == 3
    assert report["protocol"]["gallery_sha256"] == report["data"]["gallery_sha256"]
    assert report["protocol"]["preprocessing"]["input_size"] == [32, 16]
    assert report["metrics"]["original"]["valid_queries"] == 2
    assert report["cases"]["original"]["success"]["available"] + report["cases"]["original"]["failure"]["available"] == 2
    assert len((output / "rankings_original.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert list((output / "cases").glob("*.png"))
    table = folder / "comparison.csv"
    compare = subprocess.run([sys.executable, "-B", str(root / "scripts" / "compare_results.py"),
                              str(output / "metrics.json"), "--output", str(table), "--allow-partial"],
                             cwd=root, capture_output=True, text=True, timeout=30)
    assert compare.returncode == 0, compare.stderr
    assert "checkpoint" in table.read_text(encoding="utf-8")

    second = json.loads(json.dumps(report))
    second["variant"] = "baseline"
    second["config"]["model"]["variant"] = "baseline"
    second_path = folder / "second.json"
    second_path.write_text(json.dumps(second), encoding="utf-8")
    comparison = [sys.executable, "-B", str(root / "scripts" / "compare_results.py"),
                  str(output / "metrics.json"), str(second_path),
                  "--output", str(folder / "two.csv"), "--allow-partial"]
    assert subprocess.run(comparison, cwd=root, capture_output=True, text=True).returncode == 0

    for field, replacement in (("gallery_sha256", "different gallery"),
                               ("preprocessing", {"input_size": [64, 32]}),
                               ("source_file", {"sha256": "different file", "size_bytes": 1})):
        incompatible = json.loads(json.dumps(second))
        incompatible["protocol"][field] = replacement
        if field == "gallery_sha256":
            incompatible["data"]["gallery_sha256"] = replacement
        if field == "source_file":
            incompatible["data"]["file"] = replacement
        second_path.write_text(json.dumps(incompatible), encoding="utf-8")
        result = subprocess.run(comparison, cwd=root, capture_output=True, text=True)
        assert result.returncode != 0 and "Incompatible evaluation protocol" in result.stderr
