"""Run a tiny synthetic forward/backward and checkpoint round trip."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from maskcluster_ssl.losses import MaskClusterLoss
from maskcluster_ssl.models import MaskClusterSSL
from maskcluster_ssl.training.checkpoint import load_checkpoint, save_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(11)
    device = torch.device(args.device)
    model = MaskClusterSSL(
        variant="full", image_size=(32, 16), patch_size=8, embed_dim=32,
        depth=2, num_heads=4, mlp_ratio=2, prototypes=16,
        reconstruction_layers=(1, 2),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4
    )
    teacher = tuple(torch.randn(2, 3, 32, 16, device=device) for _ in range(2))
    student = tuple(torch.randn(2, 3, 32, 16, device=device) for _ in range(2))
    local = (torch.randn(2, 3, 16, 8, device=device),)
    targets = tuple(torch.rand(2, 3, 32, 16, device=device) for _ in range(2))
    masks = tuple(torch.rand(2, 4, 2, device=device) > 0.5 for _ in range(2))
    outputs = model(teacher, student, local)
    loss, diagnostics = MaskClusterLoss(patch_size=8)(outputs, targets, masks)
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("Smoke loss is non-finite")
    if any(parameter.grad is not None for parameter in model.teacher.parameters()):
        raise RuntimeError("EMA teacher received gradients")
    optimizer.step()
    before = next(model.teacher.parameters()).detach().clone()
    model.update_teacher(0.9)
    after = next(model.teacher.parameters()).detach()
    if torch.equal(before, after):
        raise RuntimeError("EMA update did not change teacher parameters")

    checkpoint = ROOT / ".tmp" / f"smoke_checkpoint_{uuid.uuid4().hex}.pt"
    save_checkpoint(
        checkpoint, model=model, optimizer=optimizer, scaler=None, epoch=0,
        global_step=1, config={"smoke": True}, best_scores={"teacher": 0.1, "student": 0.2},
    )
    reloaded = MaskClusterSSL(
        variant="full", image_size=(32, 16), patch_size=8, embed_dim=32,
        depth=2, num_heads=4, mlp_ratio=2, prototypes=16,
        reconstruction_layers=(1, 2),
    ).to(device)
    reloaded_optimizer = torch.optim.AdamW(
        [parameter for parameter in reloaded.parameters() if parameter.requires_grad], lr=1e-4
    )
    state = load_checkpoint(checkpoint, model=reloaded, optimizer=reloaded_optimizer, map_location=device)
    checkpoint.unlink()
    print(json.dumps({
        "loss": float(loss.detach().cpu()),
        "diagnostics": {key: float(value.cpu()) for key, value in diagnostics.items()},
        "resume_epoch": state["epoch"],
        "resume_step": state["global_step"],
        "device": str(device),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
