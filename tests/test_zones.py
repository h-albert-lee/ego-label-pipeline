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
