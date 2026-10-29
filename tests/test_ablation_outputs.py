"""Synthetic regression checks; these are not experimental results."""
import copy
import json
import uuid
from pathlib import Path

import pytest
import torch

from maskcluster_ssl.evaluation.analysis import comparison_contract, export_history
from maskcluster_ssl.losses import LossWeights, MaskClusterLoss
from maskcluster_ssl.models import MaskClusterSSL


def test_no_mim_matches_full_zero_reconstruction_gradients():
    models = []
    for variant in ("full", "no_mim"):
        torch.manual_seed(73)
        models.append(MaskClusterSSL(variant=variant, image_size=(32, 16), patch_size=8,
                      embed_dim=32, depth=2, num_heads=4, mlp_ratio=2,
                      prototypes=16, reconstruction_layers=(1, 2)))
    full, no_mim = models
    assert no_mim.reconstruction is None
    assert all(torch.equal(value, full.state_dict()[name]) for name, value in no_mim.state_dict().items())
    views = tuple(torch.rand(2, 3, 32, 16) for _ in range(2))
    masks = tuple(torch.ones(2, 4, 2, dtype=torch.bool) for _ in range(2))
    for network in models:
        outputs = network(views, views, (), masks)
        loss, diag = MaskClusterLoss(patch_size=8, weights=LossWeights(reconstruction=0))(outputs, views, masks)
        loss.backward()
        assert diag["loss/reconstruction"] == 0
    for name, parameter in no_mim.student.named_parameters():
        assert torch.allclose(parameter.grad, dict(full.student.named_parameters())[name].grad, atol=1e-7)


def test_contract_rejects_scientific_changes():
    report = {"config": {"seed": 1, "model": {"variant": "full"},
                          "training": {"epochs": 20, "batch_size": 8},
                          "evaluation": {"every_epochs": 1}}}
    other = copy.deepcopy(report)
    other["config"]["model"]["variant"] = "no_mim"
    assert comparison_contract(report) == comparison_contract(other)
    for key in ("epochs", "batch_size"):
        changed = copy.deepcopy(other)
        changed["config"]["training"][key] += 1
        assert comparison_contract(report) != comparison_contract(changed)


def test_synthetic_epoch_exports():
    folder = Path(__file__).resolve().parents[1] / "outputs" / "audit" / "synthetic" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    records = []
    for epoch in range(2):
        records.append({"event": "epoch", "epoch": epoch, "variant": "full", "seed": 1,
                        "train_summary": {"loss/total": 2-epoch, "loss/reconstruction": 0.5},
                        "validation": {"teacher": {"original": {"mAP": 0.2},
                            "low_resolution": {"mAP": 0.3}, "mAP_drop_percentage_points": -10}}})
    (folder / "history.jsonl").write_text("\n".join(map(json.dumps, records)))
    export_history(folder, folder / "export")
    assert "-10" in (folder / "export/epochs.csv").read_text()
    from PIL import Image
    for path in (folder / "export").glob("*.png"):
        with Image.open(path) as image:
            image.verify()
    assert json.loads((folder / "export/source.json").read_text())["synthetic"]
    (folder / "history.jsonl").write_text(json.dumps({"epoch": 0, "step": 0}))
    with pytest.raises(ValueError, match="smoke steps are not epochs"):
        export_history(folder, folder / "rejected")
    (folder / "history.jsonl").write_text("\n".join(map(json.dumps, records)))
