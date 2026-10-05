from egoownership.config import load_config
from egoownership.detection.zones import person_relative_zones, static_zones
from egoownership.schema import BBox, PersonDetection


def test_static_zones_use_yaml():
    cfg = load_config()
    z = static_zones(cfg.zones)
    assert z.derivation == "static-yaml"
    assert z.mine_y_min == cfg.zones.mine_near_y_min
    assert z.shared_x_min == cfg.zones.shared_x_min


def test_person_relative_zones_lower_mine_floor_when_persons_high():
    cfg = load_config()
    persons = [
        PersonDetection(bbox=BBox(x_min=0.10, y_min=0.10, x_max=0.30, y_max=0.45), person_id="person_1"),
        PersonDetection(bbox=BBox(x_min=0.65, y_min=0.10, x_max=0.85, y_max=0.45), person_id="person_2"),
    ]
    z = person_relative_zones(persons, cfg.zones)
    # Lowest person bottom is 0.45 → mine floor sits a hair below that.
    assert 0.45 < z.mine_y_min <= 0.85
    assert z.derivation == "person-relative"
    # Each person gets an influence rectangle.
    assert "person_1" in z.person_zones
    assert "person_2" in z.person_zones
    # Influence boxes extend to top of frame.
    assert z.person_zones["person_1"].y_min == 0.0


def test_person_relative_falls_back_when_no_persons():
    cfg = load_config()
    z = person_relative_zones([], cfg.zones)
    assert z.derivation == "static-yaml"


def test_shared_band_widens_with_two_persons():
    cfg = load_config()
    persons = [
        PersonDetection(bbox=BBox(x_min=0.05, y_min=0.10, x_max=0.20, y_max=0.40), person_id="person_1"),
        PersonDetection(bbox=BBox(x_min=0.80, y_min=0.10, x_max=0.95, y_max=0.40), person_id="person_2"),
    ]
    z = person_relative_zones(persons, cfg.zones)
    # Shared band stretches between the two centers.
    assert z.shared_x_min < 0.20
    assert z.shared_x_max > 0.80


def _person(person_id, x_min, x_max, y_max):
    from egoownership.schema import BBox, PersonDetection

    return PersonDetection(
        bbox=BBox(x_min=x_min, y_min=0.0, x_max=x_max, y_max=y_max), person_id=person_id
    )


def test_a_person_seated_at_the_side_drags_the_default_floor_to_the_bottom():
    # Two diners across the table (short bboxes) and one at its side (tall bbox).
    cfg = load_config()
    persons = [_person("p1", 0.12, 0.37, 0.23), _person("p2", 0.38, 0.56, 0.20), _person("p3", 0.59, 1.0, 0.80)]
    assert person_relative_zones(persons, cfg.zones).mine_y_min == 0.85


def test_median_floor_ignores_the_side_seated_outlier():
    cfg = load_config()
    persons = [_person("p1", 0.12, 0.37, 0.23), _person("p2", 0.38, 0.56, 0.20), _person("p3", 0.59, 1.0, 0.80)]
    z = person_relative_zones(persons, cfg.zones, wearer_floor="median")
    assert z.mine_y_min == 0.45  # clamped; median bottom 0.23 + 0.05 sits below the minimum


def test_median_and_lowest_agree_when_everyone_sits_opposite():
    cfg = load_config()
    persons = [_person("p1", 0.2, 0.4, 0.40), _person("p2", 0.6, 0.8, 0.40)]
    assert (
        person_relative_zones(persons, cfg.zones, wearer_floor="median").mine_y_min
        == person_relative_zones(persons, cfg.zones, wearer_floor="lowest").mine_y_min
    )


def test_only_the_place_setting_source_switches_to_the_median_floor():
    from egoownership.evidence_labeling import _wearer_floor

    assert _wearer_floor({"dataset": "a2pd"}) == "median"
    assert _wearer_floor({"dataset": "ego4d"}) == "lowest"
    assert _wearer_floor({}) == "lowest"
