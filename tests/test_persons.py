from egoownership.detection.persons import (
    _MAX_FACE_AREA_RATIO,
    _contains_face_center,
    _ego_hand_score,
    _pick_ego_hand_index,
)
from egoownership.schema import BBox


def test_contains_face_center_true_when_face_inside_box():
    person_box = BBox(x_min=0.2, y_min=0.1, x_max=0.5, y_max=0.6)
    face = BBox(x_min=0.30, y_min=0.15, x_max=0.40, y_max=0.25)
    assert _contains_face_center(person_box, [face])


def test_contains_face_center_false_for_bare_hand_box():
    # A hand/arm reaching in from the frame edge — no face anywhere near it.
    hand_box = BBox(x_min=0.0, y_min=0.6, x_max=0.35, y_max=1.0)
    face = BBox(x_min=0.6, y_min=0.05, x_max=0.7, y_max=0.15)
    assert not _contains_face_center(hand_box, [face])


def test_contains_face_center_false_when_no_faces_detected():
    person_box = BBox(x_min=0.2, y_min=0.1, x_max=0.5, y_max=0.6)
    assert not _contains_face_center(person_box, [])


def test_ego_hand_score_negative_when_not_touching_border():
    # Fully contained box — could be a real person with an occluded face.
    box = BBox(x_min=0.3, y_min=0.2, x_max=0.5, y_max=0.5)
    assert _ego_hand_score(box) < 0


def test_ego_hand_score_positive_and_ranked_by_lowness_when_touching_border():
    low_hand = BBox(x_min=0.0, y_min=0.6, x_max=0.35, y_max=1.0)
    high_arm_at_edge = BBox(x_min=0.0, y_min=0.1, x_max=0.2, y_max=0.4)
    assert _ego_hand_score(low_hand) > _ego_hand_score(high_arm_at_edge) >= 0


def test_pick_ego_hand_index_excludes_only_the_ego_like_faceless_box():
    boxes = [
        BBox(x_min=0.0, y_min=0.6, x_max=0.35, y_max=1.0),  # 0: faceless, touches bottom edge -> ego hand
        BBox(x_min=0.5, y_min=0.1, x_max=0.7, y_max=0.5),  # 1: faceless, fully inside -> real person, occluded face
        BBox(x_min=0.1, y_min=0.1, x_max=0.3, y_max=0.5),  # 2: has a face -> real person
    ]
    kept = [0, 1, 2]
    has_face = [False, False, True]
    assert _pick_ego_hand_index(kept, boxes, has_face) == 0


def test_pick_ego_hand_index_none_when_no_faceless_boxes():
    boxes = [BBox(x_min=0.1, y_min=0.1, x_max=0.3, y_max=0.5)]
    assert _pick_ego_hand_index([0], boxes, [True]) is None


def test_pick_ego_hand_index_none_when_faceless_box_not_border_touching():
    # A faceless but fully-contained box shouldn't be dropped — it's more
    # likely a real person whose face is occluded than a hand.
    boxes = [BBox(x_min=0.3, y_min=0.2, x_max=0.5, y_max=0.5)]
    assert _pick_ego_hand_index([0], boxes, [False]) is None


def test_giant_spurious_face_box_is_excluded_before_face_center_check():
    # Grounding DINO's low-confidence "a human face." prompt occasionally
    # fires on the whole low-light/textureless frame instead of a real face;
    # such a giant box's center then falls inside almost any large person
    # candidate (e.g. the wearer's own body, lying down with a laptop),
    # falsely marking it as "has a face" and blocking ego-hand exclusion.
    ego_body_box = BBox(x_min=0.234, y_min=0.338, x_max=0.855, y_max=0.99)
    giant_spurious_face = BBox(x_min=0.005, y_min=0.003, x_max=0.995, y_max=0.994)
    real_background_face = BBox(x_min=0.874, y_min=0.201, x_max=0.923, y_max=0.261)

    assert giant_spurious_face.area > _MAX_FACE_AREA_RATIO
    assert real_background_face.area <= _MAX_FACE_AREA_RATIO

    plausible_faces = [
        b for b in (giant_spurious_face, real_background_face) if b.area <= _MAX_FACE_AREA_RATIO
    ]
    assert not _contains_face_center(ego_body_box, plausible_faces)
