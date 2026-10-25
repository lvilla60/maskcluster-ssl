import csv
import io
import uuid
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image

from maskcluster_ssl.data.audit import audit, within_root


@pytest.fixture
def tmp_path():
    """Use project-local directories without pytest's restrictive Windows temp ACL."""
    root = Path(__file__).resolve().parents[1] / ".tmp" / ("audit_test_" + uuid.uuid4().hex)
    root.mkdir(parents=True)
    return root


def fixture_data(root, *, missing=False, mismatch=False, encoded=False):
    data = root / "Input" / "raw"
    data.mkdir(parents=True)
    (data / "testing.h5").write_bytes(b"must never be opened")
    a = np.zeros((3, 3, 4, 4), dtype="float32")
    a[0, 0, 0, 0] = np.nan
    a[1, 0, 0, 0] = np.inf
    a[2] = 1
    with h5py.File(data / "training.h5", "w") as f:
        f.attrs["layout"] = "NCHW"
        f.create_dataset("images", data=a)
        f.create_dataset("camera_ids", data=[1, 2, 1])
        if not missing:
            f.create_dataset("image_ids", data=[b"a", b"b", b"a"])
    with h5py.File(data / "validation.h5", "w") as f:
        f.attrs["layout"] = "NCHW"
        q = f.create_group("query")
        g = f.create_group("gallery")
        q.create_dataset("images", data=np.ones((2, 3, 4, 4), dtype="uint8"))
        q.create_dataset("images_low_resolution", data=np.ones((2, 3, 2, 2), dtype="uint8"))
        q.create_dataset("person_ids", data=[5, 6])
        q.create_dataset("camera_ids", data=[1, 1])
        q.create_dataset("image_ids", data=[b"q1", b"q2"])
        if encoded:
            raw = io.BytesIO()
            Image.fromarray(np.ones((4, 4, 3), dtype="uint8")).save(raw, "PNG")
            ds = g.create_dataset("images", (2,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
            for i in range(2):
                ds[i] = np.frombuffer(raw.getvalue(), dtype="uint8")
        else:
            g.create_dataset("images", data=np.ones((2, 3, 4, 4), dtype="uint8"))
        g.create_dataset("person_ids", data=[5, 6])
        g.create_dataset("camera_ids", data=[2, 1])
        g.create_dataset("image_ids", data=[b"g1"] if mismatch else [b"g1", b"g2"])
    return data


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_full_checks_and_testing_untouched(tmp_path, monkeypatch):
    fixture_data(tmp_path)
    original = h5py.File

    def guarded(path, *args, **kwargs):
        assert Path(path).name != "testing.h5"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", guarded)
    run, summary, critical = audit("Input/raw", "outputs/audit", tmp_path, batch_size=2, progress=lambda _: None)
    assert critical  # Tiny fixture deliberately differs from fixed production expectations.
    issues = read_csv(run / "issues.csv")
    assert not [x for x in issues if x["check"] == "schema"]
    assert len([x for x in issues if x["check"] == "nonfinite"]) == 2
    assert summary["valid_queries"] == 1
    queries = read_csv(run / "validation_queries.csv")
    assert [q["positive_other_camera"] for q in queries] == ["1", "0"]
    assert [q["same_identity_same_camera"] for q in queries] == ["0", "1"]
    dups = read_csv(run / "duplicates.csv")
    assert any(x["kind"] == "exact_image" and x["scope"] == "within_gallery_images" for x in dups)
    assert any(x["kind"] == "image_id" for x in dups)
    assert (run / "figures" / "query_pairs.png").is_file()


@pytest.mark.parametrize("missing,mismatch,check", [(True, False, "schema"), (False, True, "metadata_length")])
def test_schema_issues(tmp_path, missing, mismatch, check):
    fixture_data(tmp_path, missing=missing, mismatch=mismatch)
    run, _, critical = audit("Input/raw", "outputs/audit", tmp_path, progress=lambda _: None)
    assert critical
    assert any(x["check"] == check for x in read_csv(run / "issues.csv"))


def test_encoded_images(tmp_path):
    fixture_data(tmp_path, encoded=True)
    run, summary, _ = audit("Input/raw", "outputs/audit", tmp_path, progress=lambda _: None)
    assert summary["formats"]["validation/gallery/images"].startswith("encoded/PNG/")
    assert any(x["scope"] == "within_gallery_images" for x in read_csv(run / "duplicates.csv"))


def test_path_escape(tmp_path):
    fixture_data(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        audit("Input/raw", "../outside", tmp_path, progress=lambda _: None)
    with pytest.raises(ValueError, match="escapes"):
        within_root(tmp_path.parent, tmp_path)
    link = tmp_path / "out_link"
    try:
        link.symlink_to(tmp_path.parent, target_is_directory=True)
    except OSError:
        pass  # Windows may prohibit symlink creation without developer mode.
    else:
        with pytest.raises(ValueError, match="escapes"):
            within_root(link / "new_report", tmp_path)


def test_negative_person_id_is_recorded_as_convention(tmp_path):
    fixture_data(tmp_path)
    with h5py.File(tmp_path / "Input/raw/validation.h5", "r+") as f:
        f["gallery/person_ids"][0] = -1
    run, summary, _ = audit("Input/raw", "outputs/audit", tmp_path, progress=lambda _: None)
    assert summary["valid_queries"] == 0
    assert any(x["check"] == "negative_id" and x["dataset"] == "gallery/person_ids"
               for x in read_csv(run / "issues.csv"))
