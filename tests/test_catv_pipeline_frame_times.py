from egoownership.catv_pipeline import _jsonl_sparse_frame_times


def test_anchors_on_reference_frame_sec_when_present():
    # The bug this fixes: picking t-2/t-1/t independently from
    # described_frame_timestamps_sec's first/middle/last silently disagreed
    # with reference_frame_sec (what object.bbox/the caption/the reference
    # frame image actually correspond to) for ~99.5% of ego4d rows, anchoring
    # the served "t" frame on a moment the object was never detected at.
    record = {
        "reference_frame_sec": 15.0,
        "described_frame_timestamps_sec": [0, 1, 3, 5, 7, 9, 11, 13, 15, 16, 18, 20, 22, 24, 26, 28],
    }
    assert _jsonl_sparse_frame_times(record) == {"t-2": 13.0, "t-1": 14.0, "t": 15.0}


def test_clamps_to_zero_near_start_of_clip():
    record = {"reference_frame_sec": 0.5}
    assert _jsonl_sparse_frame_times(record) == {"t-2": 0.0, "t-1": 0.0, "t": 0.5}


def test_falls_back_to_described_timestamps_without_reference_frame_sec():
    record = {"described_frame_timestamps_sec": [0, 1, 3, 5, 7, 9, 11, 13, 15, 16, 18, 20, 22, 24, 26, 28]}
    assert _jsonl_sparse_frame_times(record) == {"t-2": 0.0, "t-1": 15.0, "t": 28.0}
