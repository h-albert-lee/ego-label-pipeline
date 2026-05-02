from egoownership.detection.tracking import assign_instance_ids, collect_instance_track
from egoownership.schema import BBox, FrameDetections, ObjectDetection


def _o(label, x, y, sz=0.15, score=0.9):
    h = sz / 2
    return ObjectDetection(
        label=label,
        bbox=BBox(x_min=max(0, x - h), y_min=max(0, y - h), x_max=min(1, x + h), y_max=min(1, y + h)),
        score=score,
    )


def _f(tag, t, objs):
    return FrameDetections(tag=tag, timestamp_sec=t, objects=objs)


def test_iou_tracking_keeps_id_when_object_barely_moves():
    f0 = _f("t-2", 0.0, [_o("cup", 0.50, 0.85)])
    f1 = _f("t-1", 0.1, [_o("cup", 0.51, 0.84)])
    f2 = _f("t", 0.2, [_o("cup", 0.52, 0.83)])
    out = assign_instance_ids([f0, f1, f2])
    ids = [fd.objects[0].instance_id for fd in out]
    assert ids[0] == ids[1] == ids[2]
    assert ids[0].startswith("cup_")


def test_singleton_fallback_links_same_class_across_low_iou_motion():
    """When exactly one cup is in both frames, identity is preserved even
    if IoU is 0 — common for a hand-over that moves the object across the
    frame between t-2 and t."""
    f0 = _f("t-2", 0.0, [_o("cup", 0.10, 0.10)])
    f1 = _f("t-1", 0.1, [_o("cup", 0.90, 0.90)])
    out = assign_instance_ids([f0, f1])
    assert out[0].objects[0].instance_id == out[1].objects[0].instance_id


def test_low_iou_with_duplicates_gets_distinct_ids():
    """Taxonomy D: two cups; one moves drastically. Singleton fallback must
    NOT collapse them, so we verify each cup keeps a stable id."""
    f0 = _f("t-2", 0.0, [_o("cup", 0.30, 0.50), _o("cup", 0.70, 0.50)])
    f1 = _f("t-1", 0.1, [_o("cup", 0.32, 0.50), _o("cup", 0.68, 0.50)])
    out = assign_instance_ids([f0, f1])
    ids_f0 = sorted(o.instance_id for o in out[0].objects)
    ids_f1 = sorted(o.instance_id for o in out[1].objects)
    assert ids_f0 == ids_f1
    assert ids_f0[0] != ids_f0[1]


def test_two_instances_of_same_class_get_distinct_ids():
    """Taxonomy D scenario: two cups symmetrically on a table."""
    f0 = _f("t-2", 0.0, [_o("cup", 0.30, 0.50), _o("cup", 0.70, 0.50)])
    f1 = _f("t-1", 0.1, [_o("cup", 0.30, 0.50), _o("cup", 0.70, 0.50)])
    out = assign_instance_ids([f0, f1])
    ids_f0 = sorted(o.instance_id for o in out[0].objects)
    ids_f1 = sorted(o.instance_id for o in out[1].objects)
    # Two distinct ids per frame, and identity is preserved.
    assert ids_f0[0] != ids_f0[1]
    assert ids_f0 == ids_f1


def test_collect_instance_track_returns_one_per_frame():
    f0 = _f("t-2", 0.0, [_o("pen", 0.50, 0.85)])
    f1 = _f("t-1", 0.1, [])
    f2 = _f("t", 0.2, [_o("pen", 0.50, 0.85)])
    out = assign_instance_ids([f0, f1, f2])
    iid = out[0].objects[0].instance_id
    track = collect_instance_track(out, iid)
    assert track[0] is not None and track[2] is not None
    assert track[1] is None
