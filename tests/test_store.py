"""Unit tests for SceneStore's read/write cache (server/store.py).

Focus: the mtime-validated cache must (a) actually avoid re-parsing when the
file hasn't changed, and (b) still pick up an external modification to the
file rather than silently ignoring it — the latter is the exact class of bug
that caused a real data-loss incident earlier in this project (a second
process rewriting scene_records.jsonl underneath a running server).
"""

import time
from pathlib import Path

from egoownership.schema import BBox, ClipCandidate, FrameDetections, ObjectDetection, OwnershipLabel, SceneRecord, Taxonomy
from egoownership.server.store import SceneStore


def _record(clip_id: str = "demo:clip1") -> SceneRecord:
    clip = ClipCandidate(
        dataset="ego4d_fho", clip_id=clip_id, video_id="video_A", taxonomy=Taxonomy.CONTEXTUAL,
        t_minus_2_sec=20.0, t_minus_1_sec=20.5, t_sec=21.0, verb="give", nouns=["pen"],
    )
    bbox = BBox(x_min=0.45, y_min=0.45, x_max=0.55, y_max=0.55)
    obj = ObjectDetection(label="pen", bbox=bbox, score=0.9, instance_id="pen_1", ownership=OwnershipLabel.SHARED)
    frame = FrameDetections(tag="t", timestamp_sec=21.0, width=640, height=480, objects=[obj])
    return SceneRecord(clip=clip, frames=[frame], scene_label=OwnershipLabel.SHARED)


def _make_store(tmp_path: Path, records: list[SceneRecord]) -> SceneStore:
    scenes_path = tmp_path / "scenes.jsonl"
    scenes_path.write_text("\n".join(r.model_dump_json() for r in records) + "\n", encoding="utf-8")
    return SceneStore(scenes_path)


def test_repeated_reads_reuse_cache_when_file_unchanged(tmp_path: Path):
    store = _make_store(tmp_path, [_record("a"), _record("b")])
    first = store._load()
    second = store._load()
    # Same list object -- proof the file wasn't re-read/re-parsed.
    assert first is second


def test_read_reparses_when_file_modified_externally(tmp_path: Path):
    store = _make_store(tmp_path, [_record("a")])
    first = store._load()
    assert len(first) == 1

    # Simulate a second process rewriting the file underneath this store --
    # exactly what caused the real data-loss incident this cache must not
    # reintroduce a variant of.
    time.sleep(0.01)  # ensure a distinct mtime on filesystems with coarse resolution
    store.path.write_text(
        "\n".join(r.model_dump_json() for r in [_record("a"), _record("b"), _record("c")]) + "\n",
        encoding="utf-8",
    )

    second = store._load()
    assert len(second) == 3
    assert second is not first


def test_update_refreshes_cache_so_next_read_does_not_reparse(tmp_path: Path):
    store = _make_store(tmp_path, [_record("a")])
    store._load()  # warm the cache

    store.update("a", annotator="alice", notes="hello")

    cache_after_write = store._cache
    reread = store._load()
    # update()'s _atomic_rewrite should have refreshed the cache in place --
    # the very next _load() must not trigger a fresh disk read/parse.
    assert reread is cache_after_write
    assert reread[0].notes == "hello"


def test_update_result_matches_what_is_actually_on_disk(tmp_path: Path):
    store = _make_store(tmp_path, [_record("a")])
    store.update("a", annotator="alice", review_status="verified")

    # Force a fresh parse from disk (new store instance, no cache) to confirm
    # the write was actually persisted, not just reflected in memory.
    fresh_store = SceneStore(store.path, activity_path=store.activity_path)
    on_disk = fresh_store._load()
    assert on_disk[0].review_status == "verified"
