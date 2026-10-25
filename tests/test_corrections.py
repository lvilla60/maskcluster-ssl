import copy
import random
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch

from maskcluster_ssl.data.sampling import EpochSampler
from maskcluster_ssl.data.transforms import MultiCropTransform, corrupt_masked_patches, sample_generator
from maskcluster_ssl.evaluation.report import ranking_records
from maskcluster_ssl.losses.maskcluster import masked_l1
from maskcluster_ssl.models import MaskClusterSSL, VisionTransformer
from maskcluster_ssl.training.checkpoint import load_checkpoint, save_checkpoint, validate_resume_config


def model(variant="full"):
    return MaskClusterSSL(variant=variant, image_size=(32, 16), patch_size=8,
                          embed_dim=32, depth=2, num_heads=4, mlp_ratio=2,
                          prototypes=16, reconstruction_layers=(1, 2))


def test_variant_inputs_and_shared_initialization():
    image = torch.linspace(0, 1, 3 * 32 * 16).reshape(3, 32, 16)
    references = {}
    states = {}
    for variant in ("baseline", "no_mim", "full"):
        torch.manual_seed(17)
        states[variant] = model(variant).state_dict()
        transform = MultiCropTransform(global_size=(32, 16), local_size=(16, 8),
                                       local_crops=1, patch_size=8, seed=29)
        sampler = EpochSampler(range(10), 29)
        sampler.epoch = 3
        order = list(sampler)
        views = transform(image, index=order[0][1], epoch=3)
        references[variant] = (order, views)
    for variant in ("no_mim", "full"):
        assert references[variant][0] == references["baseline"][0]
        for key in ("global_views", "global_targets", "local_views", "patch_masks"):
            assert all(torch.equal(a, b) for a, b in zip(references[variant][1][key], references["baseline"][1][key]))
        for key in states["baseline"]:
            if key in states[variant]:
                assert torch.equal(states[variant][key], states["baseline"][key])
    transform = MultiCropTransform(global_size=(32, 16), local_size=(16, 8), local_crops=1, patch_size=8, seed=29)
    other = transform(image, index=references["full"][0][0][1], epoch=4)
    assert any(not torch.equal(a, b) for a, b in zip(other["global_targets"], references["full"][1]["global_targets"]))
    ref = references["full"][1]
    source = torch.stack((ref["global_targets"][0], ref["global_targets"][0]))
    mask = torch.stack((ref["patch_masks"][0], ref["patch_masks"][0]))
    outputs = [corrupt_masked_patches(source, mask, 8,
               generator=sample_generator(29, 3, 0, "corruption")) for _ in range(3)]
    assert torch.equal(outputs[0], outputs[1]) and torch.equal(outputs[1], outputs[2])


def test_visible_attention_excludes_masked_keys_but_updates_masked_queries():
    torch.manual_seed(2)
    vit = VisionTransformer(image_size=(32, 16), patch_size=8, embed_dim=32, depth=2,
                            num_heads=4, attention_mode="visible")
    x = torch.randn(1, 3, 32, 16)
    mask = torch.zeros(1, 4, 2, dtype=torch.bool)
    mask[:, 0, 0] = True
    tokens, _ = vit.patch_embed(x)
    base = torch.cat((vit.cls_token, tokens), 1) + vit._position((4, 2))
    key_mask = torch.cat((torch.zeros(1, 1, dtype=torch.bool), mask.flatten(1)), 1)
    perturbed = base.clone()
    perturbed[:, 1, 0] += 5
    first = vit.blocks[0](base, key_mask)
    second = vit.blocks[0](perturbed, key_mask)
    assert torch.allclose(first[:, 0], second[:, 0], atol=1e-6)
    assert torch.allclose(first[:, 2:], second[:, 2:], atol=1e-6)
    visible_changed = base.clone()
    visible_changed[:, 2, 0] += 5
    third = vit.blocks[0](visible_changed, key_mask)
    assert not torch.allclose(first[:, 1], third[:, 1])
    standard = vit.blocks[0](base)
    standard_changed = vit.blocks[0](perturbed)
    # A masked key may affect CLS under the explicitly configured standard path.
    assert not torch.allclose(standard[:, 0], standard_changed[:, 0])


def test_fp16_masked_l1_reduces_in_fp32():
    prediction = torch.full((32, 3, 128, 64), 0.5, dtype=torch.float16, requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.ones((32, 8, 4), dtype=torch.bool)
    value = masked_l1(prediction, target, mask, 16)
    assert value.dtype == torch.float32
    assert value.item() == pytest.approx(0.5)
    value.backward()
    assert prediction.grad.abs().sum() > 0


def test_resume_config_rejects_trajectory_changes():
    config = {"training": {"epochs": 20}, "loss": {"student_temperature": 0.1},
              "runtime": {"device": "cpu", "amp": False}, "output_dir": "a"}
    changed = copy.deepcopy(config)
    changed["runtime"]["device"] = "cuda"
    changed["output_dir"] = "b"
    validate_resume_config(config, changed)
    for section, key, value in (("training", "epochs", 200), ("loss", "student_temperature", 0.9), ("runtime", "amp", True)):
        changed = copy.deepcopy(config)
        changed[section][key] = value
        with pytest.raises(ValueError, match=f"{section}.{key}"):
            validate_resume_config(config, changed)


def test_cpu_continuation_restores_rng_optimizer_and_generator():
    torch.manual_seed(5)
    random.seed(5)
    np.random.seed(5)
    original = model("baseline")
    optimizer = torch.optim.AdamW([p for p in original.parameters() if p.requires_grad], lr=1e-3)
    generator = sample_generator(5, 0, 0, "test")
    def step(network, opt, gen):
        opt.zero_grad()
        value = network.student.encode(torch.randn(2, 3, 32, 16))
        loss = value.square().mean() + 0 * torch.rand((), generator=gen)
        loss.backward()
        opt.step()
        return (random.random(), np.random.random(), torch.rand((), generator=gen).item())
    step(original, optimizer, generator)
    root = Path(__file__).resolve().parents[1] / ".tmp"
    root.mkdir(exist_ok=True)
    path = root / f"resume_{uuid.uuid4().hex}.pt"
    config = {"training": {"epochs": 2}}
    save_checkpoint(path, model=original, optimizer=optimizer, scaler=None, epoch=0,
                    global_step=1, config=config, best_scores={}, generators={"test": generator})
    expected_rng = step(original, optimizer, generator)
    expected = {key: value.clone() for key, value in original.state_dict().items()}
    resumed = model("baseline")
    resumed_opt = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=1e-3)
    resumed_gen = torch.Generator()
    load_checkpoint(path, model=resumed, optimizer=resumed_opt, config=config,
                    generators={"test": resumed_gen})
    path.unlink()
    assert step(resumed, resumed_opt, resumed_gen) == expected_rng
    assert all(torch.equal(value, resumed.state_dict()[key]) for key, value in expected.items())
    for p, state in resumed_opt.state.items():
        assert state["exp_avg"].device == p.device
    for left, right in zip(optimizer.state.values(), resumed_opt.state.values()):
        assert torch.equal(left["exp_avg"], right["exp_avg"])


def test_known_rankings_and_exclusions():
    query = np.array([[1., 0.], [0., 1.]])
    gallery = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    qm = {"person_id": np.array([1, 2]), "camera_id": np.array([1, 1]),
          "image_id": np.array(["q1", "q2"]), "index": np.array([0, 1])}
    gm = {"person_id": np.array([1, 1, 2, -1]), "camera_id": np.array([1, 2, 2, 2]),
          "image_id": np.array(["excluded", "hit1", "hit2", "junk"]), "index": np.arange(4)}
    records = list(ranking_records(query, gallery, qm, gm))
    assert [record["rank1_correct"] for record in records] == [True, True]
    assert records[0]["ranking"][0]["image_id"] == "hit1"
    assert all(hit["image_id"] not in {"excluded", "junk"} for hit in records[0]["ranking"])
