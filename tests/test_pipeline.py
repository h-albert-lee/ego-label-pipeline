"""End-to-end smoke test for the filter → label pipeline on fixture data.

Skips the frame-extraction and detection stages (they need real video +
models); those are exercised separately in manual runs.
"""

from pathlib import Path

from egoownership import pipeline
from egoownership.detection.ownership import assign_ownership, build_scene_record
from egoownership.schema import (
    BBox,
    ClipCandidate,
    FrameDetections,
    ObjectDetection,
    Taxonomy,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_stage_filter_produces_jsonl(tmp_path: Path):
    out = tmp_path / "candidates.jsonl"
    n = pipeline.stage_filter(
        dataset="ego4d-fho",
        annotations_path=FIXTURES / "fho_mini.json",
        taxonomy=Taxonomy.CONTEXTUAL,
        out_path=out,
    )
    assert n == 3
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        assert '"taxonomy":"C"' in line.replace(" ", "")


def test_full_loop_with_synthetic_detections(tmp_path: Path):
    """Write fake detections then run stage_label to confirm the labels land."""
    import json

    clip = ClipCandidate(
        dataset="synthetic",
        clip_id="syn_001",
        video_id="vid",
        taxonomy=Taxonomy.CONTEXTUAL,
        t_minus_2_sec=0.0,
        t_minus_1_sec=0.5,
        t_sec=1.0,
        verb="give",
        nouns=["pen"],
    )
    frames = [
        FrameDetections(
            tag="t-2", timestamp_sec=0.0, width=640, height=480,
            objects=[ObjectDetection(label="pen", bbox=BBox(x_min=0.48, y_min=0.80, x_max=0.62, y_max=0.95), score=0.9)],
        ),
        FrameDetections(
            tag="t-1", timestamp_sec=0.5, width=640, height=480,
            objects=[ObjectDetection(label="pen", bbox=BBox(x_min=0.44, y_min=0.44, x_max=0.56, y_max=0.56), score=0.9)],
        ),
        FrameDetections(
            tag="t", timestamp_sec=1.0, width=640, height=480,
            objects=[ObjectDetection(label="pen", bbox=BBox(x_min=0.09, y_min=0.19, x_max=0.21, y_max=0.31), score=0.9)],
        ),
    ]

    det_path = tmp_path / "detections.jsonl"
    with det_path.open("w") as f:
        f.write(json.dumps({
            "clip": clip.model_dump(mode="json"),
            "frames": [fd.model_dump(mode="json") for fd in frames],
        }) + "\n")

    out = tmp_path / "scenes.jsonl"
    n = pipeline.stage_label(det_path, out)
    assert n == 1
    scene = json.loads(out.read_text().strip())
    # MINE → SHARED → PERSON_k transition under our zones.
    assert scene["scene_label"] == "PERSON_k"
