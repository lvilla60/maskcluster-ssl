import uuid
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from maskcluster_ssl.data.transforms import MultiCropTransform, corrupt_masked_patches, normalize_images
from maskcluster_ssl.losses import MaskClusterLoss
from maskcluster_ssl.losses.maskcluster import global_me_max
from maskcluster_ssl.models import MaskClusterSSL, VisionTransformer
from maskcluster_ssl.training.checkpoint import load_checkpoint, save_checkpoint


def tiny_model(variant="full"):
    return MaskClusterSSL(
        variant=variant,
        image_size=(32, 16),
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
        prototypes=16,
        reconstruction_layers=(1, 2),
    )


def test_rectangular_positions_and_local_crop_grid():
    backbone = VisionTransformer(
        image_size=(32, 16), patch_size=8, embed_dim=32, depth=2, num_heads=4
    )
    global_output = backbone.forward_features(torch.randn(2, 3, 32, 16))
    local_output = backbone.forward_features(torch.randn(2, 3, 16, 8))
    assert global_output.grid == (4, 2)
    assert global_output.patches.shape == (2, 8, 32)
    assert local_output.grid == (2, 1)
    assert local_output.patches.shape == (2, 2, 32)


def test_view_mask_correspondence_and_gmml_changes_only_masked_patches():
    transform = MultiCropTransform(
        global_size=(32, 16), local_size=(16, 8), local_crops=1,
        patch_size=8, mask_ratio=0.5,
    )
    item = transform(torch.rand(3, 32, 16))
    assert item["global_views"][0].shape == (3, 32, 16)
    assert item["local_views"][0].shape == (3, 16, 8)
    assert item["patch_masks"][0].shape == (4, 2)
    assert int(item["patch_masks"][0].sum()) == 4
    images = torch.stack((item["global_targets"][0], item["global_targets"][0]))
    masks = torch.stack((item["patch_masks"][0], item["patch_masks"][0]))
    corrupted = corrupt_masked_patches(images, masks, 8)
    pixel_mask = masks.repeat_interleave(8, 1).repeat_interleave(8, 2)[:, None]
    assert torch.equal(corrupted.masked_select(~pixel_mask), images.masked_select(~pixel_mask))
    assert not torch.equal(corrupted.masked_select(pixel_mask), images.masked_select(pixel_mask))
    assert torch.isfinite(normalize_images(corrupted)).all()


def test_finite_losses_gradients_teacher_ema_and_branch_absence():
    torch.manual_seed(3)
    model = tiny_model("full")
    teacher = tuple(torch.randn(2, 3, 32, 16) for _ in range(2))
    student = tuple(torch.randn(2, 3, 32, 16) for _ in range(2))
    locals_ = (torch.randn(2, 3, 16, 8),)
    targets = tuple(torch.rand(2, 3, 32, 16) for _ in range(2))
    masks = tuple(torch.rand(2, 4, 2) > 0.5 for _ in range(2))
    outputs = model(teacher, student, locals_)
    loss, diagnostics = MaskClusterLoss(patch_size=8)(outputs, targets, masks)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in diagnostics.values())
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.student.parameters())
    assert any(parameter.grad is not None for parameter in model.reconstruction.parameters())
    assert all(parameter.grad is None for parameter in model.teacher.parameters())
    old = [parameter.detach().clone() for parameter in model.teacher.parameters()]
    with torch.no_grad():
        next(model.student.parameters()).add_(0.1)
    model.update_teacher(0.5)
    assert not torch.equal(old[0], next(model.teacher.parameters()))
    assert all(not parameter.requires_grad for parameter in model.teacher.parameters())

    baseline = tiny_model("baseline")
    no_mim = tiny_model("no_mim")
    assert baseline.student.patch_head is None and baseline.reconstruction is None
    assert no_mim.student.patch_head is not None and no_mim.reconstruction is None


def test_checkpoint_round_trip():
    model = tiny_model("no_mim")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4
    )
    root = Path(__file__).resolve().parents[1] / ".tmp"
    root.mkdir(exist_ok=True)
    path = root / f"checkpoint_test_{uuid.uuid4().hex}.pt"
    reference = {name: value.detach().clone() for name, value in model.state_dict().items()}
    save_checkpoint(
        path, model=model, optimizer=optimizer, scaler=None, epoch=2, global_step=17,
        config={"variant": "no_mim"}, best_scores={"teacher": 0.2, "student": 0.1},
    )
    with torch.no_grad():
        next(model.student.parameters()).zero_()
    state = load_checkpoint(path, model=model, optimizer=optimizer)
    path.unlink()
    assert state["epoch"] == 2 and state["global_step"] == 17
    for name, value in model.state_dict().items():
        assert torch.equal(value, reference[name])


def test_global_me_max_matches_paper_a5_local_weighting():
    temperature = 1.0
    global_logits = [
        torch.tensor([[4.0, 0.0]]),
        torch.tensor([[0.0, 4.0]]),
    ]
    local_logits = [torch.tensor([[5.0, -2.0]])]
    loss, entropy = global_me_max(global_logits, local_logits, temperature)
    probabilities = [torch.softmax(value, dim=-1) for value in global_logits + local_logits * 2]
    expected_mean = torch.cat(probabilities).mean(dim=0)
    expected_entropy = -(expected_mean * expected_mean.log()).sum()
    assert entropy == pytest.approx(float(expected_entropy))
    assert loss == pytest.approx(float(-expected_entropy))
