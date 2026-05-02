"""End-to-end test of the FastAPI annotator server.

Uses fastapi.testclient → no socket required. Validates:
- listing
- single-record fetch
- POST update creates an edit + activity entry
- stats reflect the change
"""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from egoownership.schema import (
    BBox,
    ClipCandidate,
    FrameDetections,
    ObjectDetection,
    OwnershipLabel,
    SceneRecord,
    Taxonomy,
)
from egoownership.server import create_app


def _record() -> SceneRecord:
    clip = ClipCandidate(
        dataset="ego4d_fho",
        clip_id="demo:give_pen",
        video_id="video_A",
        taxonomy=Taxonomy.CONTEXTUAL,
        t_minus_2_sec=20.0,
        t_minus_1_sec=20.5,
        t_sec=21.0,
        verb="give",
        nouns=["pen"],
    )
    bbox = BBox(x_min=0.45, y_min=0.45, x_max=0.55, y_max=0.55)
    obj = ObjectDetection(label="pen", bbox=bbox, score=0.9, instance_id="pen_1",
                          ownership=OwnershipLabel.SHARED)
    frame = FrameDetections(tag="t", timestamp_sec=21.0, width=640, height=480, objects=[obj])
    return SceneRecord(clip=clip, frames=[frame], scene_label=OwnershipLabel.SHARED,
                       auto_label_confidence=0.7)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    scenes = tmp_path / "scenes.jsonl"
    scenes.write_text(_record().model_dump_json() + "\n", encoding="utf-8")
    frames_root = tmp_path / "frames"
    frames_root.mkdir()
    app = create_app(scenes_path=scenes, frames_root=frames_root)
    return TestClient(app)


def test_list_endpoint(client: TestClient):
    res = client.get("/api/scenes")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    assert data[0]["clip_id"] == "demo:give_pen"


def test_get_endpoint(client: TestClient):
    res = client.get("/api/scenes/demo:give_pen")
    assert res.status_code == 200
    rec = res.json()
    assert rec["clip"]["clip_id"] == "demo:give_pen"
    assert rec["scene_label"] == "SHARED"


def test_post_update_records_edit_and_activity(client: TestClient):
    res = client.post(
        "/api/scenes/demo:give_pen",
        json={
            "annotator": "alice",
            "scene_label": "PERSON_k",
            "review_status": "verified",
            "notes": "human reviewed; pen handed off",
        },
    )
    assert res.status_code == 200
    rec = res.json()
    assert rec["scene_label"] == "PERSON_k"
    assert rec["review_status"] == "verified"
    assert any(e["field"] == "scene_label" and e["new_value"] == "PERSON_k" for e in rec["edits"])

    activity = client.get("/api/activity").json()
    assert any(a["clip_id"] == "demo:give_pen" and a["annotator"] == "alice" for a in activity)


def test_object_override_applies_label(client: TestClient):
    res = client.post(
        "/api/scenes/demo:give_pen",
        json={"annotator": "bob", "object_overrides": {"pen_1": "MINE"}},
    )
    assert res.status_code == 200
    rec = res.json()
    pen = rec["frames"][0]["objects"][0]
    assert pen["ownership"] == "MINE"


def test_stats_reflects_status_changes(client: TestClient):
    pre = client.get("/api/stats").json()
    assert pre["by_status"].get("draft", 0) == 1

    client.post(
        "/api/scenes/demo:give_pen",
        json={"annotator": "carol", "review_status": "verified"},
    )
    post = client.get("/api/stats").json()
    assert post["by_status"].get("verified", 0) == 1
    assert post["by_status"].get("draft", 0) == 0


def test_list_filter_by_status(client: TestClient):
    client.post(
        "/api/scenes/demo:give_pen",
        json={"annotator": "alice", "review_status": "in_review"},
    )
    drafts = client.get("/api/scenes?status=draft").json()
    in_review = client.get("/api/scenes?status=in_review").json()
    assert len(drafts) == 0
    assert len(in_review) == 1
