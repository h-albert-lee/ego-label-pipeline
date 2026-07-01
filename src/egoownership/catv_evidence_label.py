"""Merge EgoLife CAT-V object captions with lightweight visual evidence.

This module is intentionally narrower than the original ``detect`` pipeline:
the target object box already comes from SAM-3/SAM-2/CAT-V, so we only add the
useful missing pieces for ownership labels: visible persons, simple zones,
relations, and a conservative taxonomy/GT proposal.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from egoownership.config import load_config, normalize_token
from egoownership.detection.relations import build_scene_graph
from egoownership.detection.zones import person_relative_zones, static_zones
from egoownership.schema import (
    BBox,
    FrameDetections,
    ObjectDetection,
    OwnershipLabel,
    PersonDetection,
    Taxonomy,
)


PersonDetector = Callable[[Path], list[PersonDetection]]
DecisionFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]

_SHARED_OBJECTS = {
    "tissue",
    "napkin",
    "sauce",
    "oil",
    "salt",
    "pepper",
    "sugar",
    "water",
    "food",
    "dish",
    "plate",
    "bowl",
    "chopstick",
    "spoon",
    "fork",
    "knife",
    "trash",
    "bin",
    "pot",
    "tray",
}
_PERSONAL_OBJECTS = {
    "phone",
    "smartphone",
    "laptop",
    "computer",
    "tablet",
    "notebook",
    "pen",
    "wallet",
    "key",
    "card",
    "bag",
    "watch",
    "charger",
    "cable",
    "usb",
}
_OWNERSHIP_HISTORY_VERBS = {
    "give",
    "pass",
    "hand",
    "receive",
    "serve",
    "offer",
    "borrow",
    "return",
    "lend",
    "loan",
}
_OWNERSHIP_HISTORY_PATTERNS = (
    "originally",
    "previously",
    "prior possession",
    "borrowed",
    "lent",
    "loaned",
    "returned",
    "given to",
    "passed to",
    "handed to",
    "served to",
    "placed a new",
    "puts a new",
    "temporary use",
    "from the center",
    "moved from the center",
    "pushed to the center",
    "left in the center",
)
_CONFLICT_PATTERNS = (
    "handle facing",
    "handle points",
    "handle toward",
    "screen facing",
    "screen open toward",
    "open toward",
    "facing the camera wearer",
    "facing the wearer",
    "facing me",
    "name badge",
    "id card",
    "employee id",
    "keychain",
    "car key",
    "mirror",
    "reflection",
    "reflected",
)
_EGO_PATTERNS = (
    "camera wearer",
    "ego hand",
    "wearer hand",
    "camera-wearer",
    "my hand",
    "i hold",
    "i pick",
    "i place",
    "i put",
    "i move",
    "i use",
)
_OTHER_PATTERNS = (
    "other person",
    "another person",
    "visible person",
    "opposite person",
    "person across",
    "someone else",
    "their hand",
    "his hand",
    "her hand",
)
_AMBIGUOUS_PATTERNS = (
    "ambiguous",
    "unclear",
    "cannot be distinguished",
    "cannot tell",
    "not visually inferable",
    "actor identity is ambiguous",
)
_TEMPORAL_FRAME_TAGS = ("t-2", "t-1", "t")
_TEMPORAL_FRAME_PATH_KEYS = {
    "t-2": "frame_t_minus_2_path",
    "t-1": "frame_t_minus_1_path",
    "t": "frame_t_path",
}
_SHARED_ZONES = frozenset({"shared_zone", "center_table"})
_OWNERSHIP_CLUE_ZONES = frozenset({"ego_zone", "other_person_zone"})


def build_evidence_label(
    row: dict[str, Any],
    *,
    person_detector: PersonDetector | None = None,
    decision_fn: DecisionFn | None = None,
) -> dict[str, Any]:
    cfg = load_config()
    target_label = _target_label(row)
    temporal = _build_temporal_evidence(row, target_label, person_detector=person_detector, cfg=cfg)

    frame_path = Path(str(row.get("first_frame_path") or row.get("frame_t_path") or ""))
    obj = row.get("object") or {}
    target_bbox = _bbox_from_dict(obj.get("bbox") or {})
    detector_error = temporal.get("detector_error")
    persons: list[PersonDetection] = []
    zones = static_zones(cfg.zones)
    relation_summary: list[dict[str, Any]] = []
    zone = "background_or_ambiguous_zone"

    current_snapshot = (temporal.get("frame_snapshots") or {}).get("t")
    if current_snapshot is not None:
        zone = current_snapshot["target_zone"]
        persons = _persons_from_snapshot(current_snapshot)
        relation_summary = list(current_snapshot.get("relations") or [])
    elif frame_path.exists() and target_bbox.x_max > target_bbox.x_min:
        try:
            persons = person_detector(frame_path) if person_detector is not None else []
        except Exception as exc:  # noqa: BLE001
            detector_error = f"{type(exc).__name__}: {str(exc)[:300]}"
        zones = person_relative_zones(persons, cfg.zones) if persons else static_zones(cfg.zones)
        target = ObjectDetection(
            label=target_label,
            bbox=target_bbox,
            score=obj.get("score"),
            instance_id="target",
        )
        frame = FrameDetections(
            tag="t",
            frame_path=str(frame_path),
            timestamp_sec=float(row.get("first_frame_sec") or row.get("catv_start_sec") or row.get("start_sec") or 0.0),
            objects=[target],
            persons=persons,
            zones=zones,
            narration=row.get("object_caption") or row.get("dense_caption_en"),
        )
        frame = build_scene_graph([frame])[0]
        zone = _target_zone(target_bbox, zones)
        relation_summary = [
            rel.model_dump(mode="json")
            for rel in frame.relations
            if rel.subject_id == "target" or rel.object_id == "target"
        ]

    nearest_person = _nearest_person(target_bbox, persons)
    evidence = {
        "target_object": target_label,
        "object_type": _object_type(target_label),
        "target_zone": zone,
        "nearest_other_person": nearest_person,
        "person_count": len(persons),
        "visible_other_person": bool(persons),
        "relations": relation_summary,
        "caption_cues": _caption_cues(row),
        "detector_error": detector_error,
        "temporal": temporal,
    }
    decision = (decision_fn or _decide_taxonomy_gt)(row, evidence)
    return {
        **row,
        "evidence": evidence,
        "auto_taxonomy": decision["taxonomy"],
        "auto_ground_truth": decision["ground_truth"],
        "auto_key_evidence": decision["key_evidence"],
        "auto_rationale": decision["rationale"],
        "needs_review": decision["ground_truth"] == OwnershipLabel.AMBIGUOUS.value,
    }


def _decide_taxonomy_gt(row: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    label = evidence["target_object"]
    object_type = evidence["object_type"]
    zone = evidence["target_zone"]
    cues = evidence["caption_cues"]
    relations = evidence.get("relations") or []
    verb = normalize_token(str(row.get("verb") or ""))

    key: dict[str, Any] = {
        "object_type": object_type,
        "target_zone": zone,
        "verb": verb,
        "caption_cues": cues,
        "person_count": evidence.get("person_count"),
    }

    if cues["ambiguous"] and not (cues["ego_actor"] or cues["other_actor"]):
        return _decision(Taxonomy.AMBIGUOUS, OwnershipLabel.AMBIGUOUS, 0.72, key, "Actor/ownership evidence is explicitly ambiguous.")

    contextual = (
        verb in _OWNERSHIP_HISTORY_VERBS
        or bool((evidence.get("temporal") or {}).get("contextual_requires_history"))
    )
    conflict = cues["conflict_cue"]
    if object_type == "shared" and zone == "ego_zone":
        conflict = True
    if object_type == "personal" and zone in {"shared_zone", "other_person_zone"}:
        conflict = True

    explicit_shared_gt = cues["shared_use"] or object_type == "shared"
    if explicit_shared_gt:
        gt = OwnershipLabel.SHARED
        confidence = 0.82
        rationale = "Caption or object type indicates shared/common use."
    elif cues["other_actor"] and not cues["ego_actor"]:
        gt = OwnershipLabel.PERSON_K
        confidence = 0.78
        rationale = "Other visible person is the main actor interacting with the target."
    elif cues["ego_actor"] and not cues["other_actor"]:
        gt = OwnershipLabel.MINE
        confidence = 0.80
        rationale = "Camera wearer/ego hand is the main actor interacting with the target."
    elif any(rel.get("predicate") == "held_by" and str(rel.get("object_id", "")).startswith("person_") for rel in relations):
        gt = OwnershipLabel.PERSON_K
        confidence = 0.72
        rationale = "Target overlaps an other-person hand zone."
    elif zone == "other_person_zone":
        gt = OwnershipLabel.PERSON_K
        confidence = 0.66
        rationale = "Target is spatially closest to another visible person."
    elif object_type == "shared" and zone in {"shared_zone", "center_table"}:
        gt = OwnershipLabel.SHARED
        confidence = 0.76
        rationale = "Shared-type object lies in a shared/central table zone."
    elif zone == "ego_zone":
        gt = OwnershipLabel.MINE
        confidence = 0.66
        rationale = "Target is in the camera-wearer/near zone."
    elif zone in {"shared_zone", "center_table"}:
        gt = OwnershipLabel.SHARED if object_type == "shared" else OwnershipLabel.AMBIGUOUS
        confidence = 0.62 if gt is OwnershipLabel.SHARED else 0.50
        rationale = "Target lies in the shared band, but actor evidence is weak."
    else:
        gt = OwnershipLabel.AMBIGUOUS
        confidence = 0.45
        rationale = "Insufficient actor, object-type, and zone evidence."

    if gt is OwnershipLabel.AMBIGUOUS:
        tax = Taxonomy.AMBIGUOUS
    elif contextual:
        tax = Taxonomy.CONTEXTUAL
        temporal = evidence.get("temporal") or {}
        if temporal.get("contextual_requires_history") and verb not in _OWNERSHIP_HISTORY_VERBS:
            past_tags = [tag for tag in ("t-2", "t-1") if tag in (temporal.get("frame_snapshots") or {})]
            rationale = (
                f"{rationale} Ownership clue visible in past frames ({', '.join(past_tags)}) "
                f"but not at t (zone={temporal.get('current_frame_zone', zone)})."
            )
    elif conflict:
        tax = Taxonomy.CONFLICT
    else:
        tax = Taxonomy.BASELINE
    return _decision(tax, gt, confidence, key, rationale)


def _decision(
    taxonomy: Taxonomy,
    gt: OwnershipLabel,
    confidence: float,
    key: dict[str, Any],
    rationale: str,
) -> dict[str, Any]:
    return {
        "taxonomy": taxonomy.value,
        "ground_truth": gt.value,
        "confidence": confidence,
        "key_evidence": key,
        "rationale": rationale,
    }


def build_evidence_packet(row: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    """Bundle every signal available for an ownership decision into one packet.

    Used both by the rule-based decider (indirectly, via ``evidence``) and by
    ``LLMTaxonomyDecider``, which gets exactly this packet serialized into its
    prompt so the model reasons over the same evidence the rules see, not raw
    pixels it doesn't have access to. Deliberately excludes the free-text
    ``object_caption``/``dense_caption_en``/transcript fields so the decision
    is grounded only in structured/geometric evidence, not caption wording.
    """
    return {
        "target_object": evidence["target_object"],
        "object_type": evidence["object_type"],
        "target_zone": evidence["target_zone"],
        "person_count": evidence["person_count"],
        "visible_other_person": evidence["visible_other_person"],
        "nearest_other_person_distance": (evidence.get("nearest_other_person") or {}).get("distance"),
        "relations": [
            {"predicate": rel.get("predicate"), "object_id": rel.get("object_id"), "subject_id": rel.get("subject_id")}
            for rel in (evidence.get("relations") or [])
        ],
        "verb": str(row.get("verb") or ""),
        "nouns": list(row.get("nouns") or []),
        "detector_error": evidence.get("detector_error"),
        "temporal": {
            "frames_analyzed": evidence.get("temporal", {}).get("frames_analyzed"),
            "frame_zones": evidence.get("temporal", {}).get("frame_zones"),
            "past_frame_ownership_clue": evidence.get("temporal", {}).get("past_frame_ownership_clue"),
            "current_frame_ownership_clue": evidence.get("temporal", {}).get("current_frame_ownership_clue"),
            "contextual_requires_history": evidence.get("temporal", {}).get("contextual_requires_history"),
            "ownership_trajectory": evidence.get("temporal", {}).get("ownership_trajectory"),
            "object_moved": evidence.get("temporal", {}).get("object_moved"),
            "held_by_changed": evidence.get("temporal", {}).get("held_by_changed"),
        },
    }


_LLM_DECISION_SYSTEM_PROMPT = (
    "You decide ownership labels for objects in egocentric video for a research benchmark. "
    "You will be given a JSON evidence packet about one highlighted object (HO): its type, spatial "
    "zone relative to people, detected persons, scene-graph relations, and the verb/nouns describing the "
    "interaction. No caption or transcript text is included — decide using only this structured evidence. "
    "Decide a taxonomy and ground_truth label using ONLY the evidence given — do not invent facts not "
    "present in the packet. If the evidence is genuinely insufficient or conflicting, prefer AMBIGUOUS "
    "over guessing.\n\n"
    "taxonomy must be one of: A (baseline: clear-cut ownership), B (conflict: object type and zone or "
    "actor evidence disagree), C (contextual: ownership depends on history like give/lend/return/borrow), "
    "D (ambiguous: insufficient or contradictory evidence).\n"
    "ground_truth must be one of: MINE (camera wearer owns/uses it), PERSON_k (another visible person "
    "owns/uses it), SHARED (communal/shared-use object), AMBIGUOUS.\n\n"
    "Respond with ONLY a single JSON object, no prose, no markdown fences, in this exact shape:\n"
    '{"taxonomy": "A|B|C|D", "ground_truth": "MINE|PERSON_k|SHARED|AMBIGUOUS", '
    '"confidence": 0.0-1.0, "rationale": "one or two sentences"}'
)


@dataclass
class LLMTaxonomyDecider:
    """Decide taxonomy/ground-truth from a full evidence packet via a local text LLM.

    Falls back to the deterministic rule-based ``_decide_taxonomy_gt`` if the
    model is unavailable or its output can't be parsed into the expected
    schema, so a bad/odd generation never silently corrupts a row.
    """

    model_id: str = "Qwen/Qwen3-4B"
    device: str | None = None
    max_new_tokens: int = 200
    _pipeline: Any = field(default=None, init=False, repr=False)

    def __call__(self, row: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
        packet = build_evidence_packet(row, evidence)
        try:
            raw = self._generate(packet)
            parsed = _parse_llm_decision(raw)
        except Exception as exc:  # noqa: BLE001
            fallback = _decide_taxonomy_gt(row, evidence)
            fallback["rationale"] = f"[llm_decision_failed: {type(exc).__name__}] {fallback['rationale']}"
            return fallback
        if parsed is None:
            fallback = _decide_taxonomy_gt(row, evidence)
            fallback["rationale"] = f"[llm_decision_unparseable] {fallback['rationale']}"
            return fallback
        return {
            "taxonomy": parsed["taxonomy"],
            "ground_truth": parsed["ground_truth"],
            "confidence": parsed["confidence"],
            "key_evidence": packet,
            "rationale": parsed["rationale"],
        }

    def _generate(self, packet: dict[str, Any]) -> str:
        pipe = self._load_pipeline()
        messages = [
            {"role": "system", "content": _LLM_DECISION_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
        ]
        # Some chat models (e.g. Qwen3) default to emitting a long <think>...</think>
        # reasoning block before the answer; disabling it via the chat template keeps
        # the response short enough to fit max_new_tokens and land on the JSON.
        prompt = pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        # Pass a GenerationConfig object rather than loose max_new_tokens/do_sample
        # kwargs: mixing the two is what triggers transformers' "max_new_tokens and
        # max_length both set" / "generation_config passed together with..." warnings.
        from transformers import GenerationConfig

        gen_config = GenerationConfig(max_new_tokens=self.max_new_tokens, do_sample=False)
        output = pipe(prompt, generation_config=gen_config, return_full_text=False)
        return str(output[0]["generated_text"])

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        from transformers import pipeline

        device_arg: int | str | None
        if self.device in (None, "", "auto"):
            device_arg = None
        elif self.device.startswith("cuda:"):
            device_arg = int(self.device.split(":", 1)[1])
        else:
            device_arg = self.device
        kwargs: dict[str, Any] = {"model": self.model_id}
        if device_arg is not None:
            kwargs["device"] = device_arg
        self._pipeline = pipeline("text-generation", **kwargs)
        return self._pipeline


def _parse_llm_decision(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    taxonomy = str(data.get("taxonomy") or "").strip().upper()
    ground_truth = str(data.get("ground_truth") or "").strip()
    if taxonomy not in {t.value for t in Taxonomy}:
        return None
    gt_lookup = {gt.value.upper(): gt.value for gt in OwnershipLabel}
    ground_truth = gt_lookup.get(ground_truth.upper())
    if ground_truth is None:
        return None
    try:
        confidence = float(data.get("confidence"))
    except (TypeError, ValueError):
        return None
    confidence = max(0.0, min(1.0, confidence))
    rationale = str(data.get("rationale") or "").strip() or "LLM decision (no rationale given)."
    return {
        "taxonomy": taxonomy,
        "ground_truth": ground_truth,
        "confidence": confidence,
        "rationale": rationale,
    }


def _caption_cues(row: dict[str, Any]) -> dict[str, bool]:
    text = " ".join(
        str(row.get(key) or "")
        for key in ("object_caption", "dense_caption_en", "transcript_en", "qwen_translation")
    ).lower()
    verb = normalize_token(str(row.get("verb") or ""))
    transfer = _contains_text_term(
        text,
        ("give", "gave", "pass", "passed", "handed", "receive", "received", "borrow", "borrowed", "return", "returned", "lend", "lent"),
    )
    serving = _contains_text_term(
        text,
        ("server", "waiter", "waitress", "staff", "served", "serves", "served to", "placed a new", "puts a new"),
    )
    prior_possession = _contains_text_term(
        text,
        ("originally", "previously", "prior possession", "belongs to", "still belongs to", "owner remains"),
    )
    shared_origin = any(
        p in text
        for p in (
            "from the center",
            "from the shared object",
            "from the shared container",
            "common container",
            "communal",
            "temporary use",
            "moved from the center",
            "pushed to the center",
            "left in the center",
        )
    )
    ownership_history = (
        verb in _OWNERSHIP_HISTORY_VERBS
        or transfer
        or serving
        or prior_possession
        or shared_origin
        or any(p in text for p in _OWNERSHIP_HISTORY_PATTERNS)
    )
    conflict_cue = any(p in text for p in _CONFLICT_PATTERNS)
    return {
        "ego_actor": any(p in text for p in _EGO_PATTERNS),
        "other_actor": any(p in text for p in _OTHER_PATTERNS),
        "ambiguous": any(p in text for p in _AMBIGUOUS_PATTERNS),
        "shared_area": any(p in text for p in ("shared area", "shared table", "shared zone", "table center", "center of the table")),
        "shared_use": any(p in text for p in ("common", "everyone", "together", "communal", "shared object", "for everyone", "used by everyone")),
        "temporal": bool(re.search(r"\bfrom\b.+\bto\b", text)),
        "moved_or_state_changed": any(p in text for p in ("moves", "moved", "picked", "placed", "put", "held", "turns", "changes", "final status")),
        "transfer": transfer,
        "serving": serving,
        "prior_possession": prior_possession,
        "shared_origin": shared_origin,
        "ownership_history": ownership_history,
        "conflict_cue": conflict_cue,
    }


def _target_zone(bbox: BBox, zones: Any) -> str:
    cx, cy = bbox.center
    for person_id, zone in zones.person_zones.items():
        if zone.x_min <= cx <= zone.x_max and zone.y_min <= cy <= zone.y_max:
            return "other_person_zone"
    if cy >= zones.mine_y_min:
        return "ego_zone"
    if zones.shared_x_min <= cx <= zones.shared_x_max:
        return "shared_zone"
    return "background_or_ambiguous_zone"


def _nearest_person(bbox: BBox, persons: list[PersonDetection]) -> dict[str, Any] | None:
    if not persons:
        return None
    cx, cy = bbox.center
    best = None
    best_dist = 999.0
    for person in persons:
        px, py = person.bbox.center
        dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        if dist < best_dist:
            best = person
            best_dist = dist
    if best is None:
        return None
    return {
        "person_id": best.person_id,
        "distance": best_dist,
        "bbox": best.bbox.model_dump(mode="json"),
        "score": best.score,
    }


def _object_type(label: str) -> str:
    norm = normalize_token(label)
    parts = set(norm.split("_"))
    if norm in _SHARED_OBJECTS or parts & _SHARED_OBJECTS:
        return "shared"
    if norm in _PERSONAL_OBJECTS or parts & _PERSONAL_OBJECTS:
        return "personal"
    return "generic_object"


def _target_label(row: dict[str, Any]) -> str:
    obj = row.get("object") or {}
    for value in (
        obj.get("target_noun"),
        obj.get("label"),
        *((row.get("nouns") or [])[:1]),
    ):
        text = str(value or "").strip()
        if text and text.lower() not in {"sam2_object", "sam2 object", "object"}:
            return text
    return "object"


def _contains_text_term(text: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        if " " in term:
            if term in text:
                return True
            continue
        if re.search(rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])", text):
            return True
    return False


def _bbox_from_dict(data: dict[str, Any]) -> BBox:
    return BBox(
        x_min=float(data.get("x_min") or 0.0),
        y_min=float(data.get("y_min") or 0.0),
        x_max=float(data.get("x_max") or 0.0),
        y_max=float(data.get("y_max") or 0.0),
    )


def _frame_path_for_tag(row: dict[str, Any], tag: str) -> Path | None:
    key = _TEMPORAL_FRAME_PATH_KEYS.get(tag)
    raw = row.get(key) if key else None
    if not raw:
        raw = (row.get("frame_paths") or {}).get(tag)
    if tag == "t" and not raw:
        raw = row.get("first_frame_path")
    if not raw:
        return None
    return Path(str(raw))


def _build_temporal_evidence(
    row: dict[str, Any],
    target_label: str,
    *,
    person_detector: PersonDetector | None,
    cfg: Any,
) -> dict[str, Any]:
    temporal_objects = dict(row.get("temporal_target_objects") or {})
    frame_times = dict(row.get("frame_times_sec") or {})
    frames: list[FrameDetections] = []
    detector_error: str | None = None
    analyzed_tags: list[str] = []

    for tag in _TEMPORAL_FRAME_TAGS:
        obj = temporal_objects.get(tag)
        if tag == "t" and not obj:
            obj = row.get("object") or {}
        if not isinstance(obj, dict):
            continue
        bbox = _bbox_from_dict(obj.get("bbox") or {})
        if bbox.x_max <= bbox.x_min:
            continue
        frame_path = _frame_path_for_tag(row, tag)
        timestamp = float(
            frame_times.get(tag, row.get("first_frame_sec") or row.get("catv_start_sec") or row.get("start_sec") or 0.0)
        )
        persons: list[PersonDetection] = []
        if person_detector is not None and frame_path is not None and frame_path.exists():
            try:
                persons = person_detector(frame_path)
            except Exception as exc:  # noqa: BLE001
                detector_error = f"{type(exc).__name__}: {str(exc)[:300]}"
        zones = person_relative_zones(persons, cfg.zones) if persons else static_zones(cfg.zones)
        frames.append(
            FrameDetections(
                tag=tag,
                frame_path=str(frame_path) if frame_path else "",
                timestamp_sec=timestamp,
                objects=[
                    ObjectDetection(
                        label=target_label,
                        bbox=bbox,
                        score=obj.get("score"),
                        instance_id="target",
                    )
                ],
                persons=persons,
                zones=zones,
                narration=row.get("object_caption") or row.get("dense_caption_en"),
            )
        )
        analyzed_tags.append(tag)

    if len(frames) < 2:
        return {
            "frames_analyzed": analyzed_tags,
            "frame_snapshots": {},
            "frame_zones": {},
            "cross_frame_relations": [],
            "object_moved": False,
            "held_by_changed": False,
            "zone_changed": False,
            "ownership_trajectory": None,
            "past_frame_ownership_clue": False,
            "current_frame_ownership_clue": False,
            "contextual_requires_history": False,
            "detector_error": detector_error,
        }

    graph_frames = build_scene_graph(frames)
    snapshots: dict[str, dict[str, Any]] = {}
    frame_zones: dict[str, str] = {}
    for fd in graph_frames:
        target_obj = next((obj for obj in fd.objects if obj.instance_id == "target"), None)
        if target_obj is None:
            continue
        zone = _target_zone(target_obj.bbox, fd.zones)
        relations = [
            rel.model_dump(mode="json")
            for rel in fd.relations
            if rel.subject_id == "target" or rel.object_id == "target"
        ]
        snapshots[fd.tag] = {
            "tag": fd.tag,
            "target_zone": zone,
            "held_by": _held_by_person(relations),
            "person_count": len(fd.persons),
            "relations": relations,
            "persons": [person.model_dump(mode="json") for person in fd.persons],
        }
        frame_zones[fd.tag] = zone

    cross_moved = [
        rel.model_dump(mode="json")
        for rel in graph_frames[-1].relations
        if rel.predicate == "moved_to" and rel.subject_id == "target"
    ]
    signals = _derive_temporal_ownership_signals(snapshots, cross_moved)
    return {
        "frames_analyzed": analyzed_tags,
        "frame_snapshots": snapshots,
        "frame_zones": frame_zones,
        "cross_frame_relations": cross_moved,
        "detector_error": detector_error,
        **signals,
    }


def _persons_from_snapshot(snapshot: dict[str, Any]) -> list[PersonDetection]:
    persons: list[PersonDetection] = []
    for raw in snapshot.get("persons") or []:
        if not isinstance(raw, dict):
            continue
        bbox = raw.get("bbox") or {}
        persons.append(
            PersonDetection(
                bbox=_bbox_from_dict(bbox),
                person_id=str(raw.get("person_id") or ""),
                score=raw.get("score"),
            )
        )
    return persons


def _held_by_person(relations: list[dict[str, Any]]) -> str | None:
    for rel in relations:
        if rel.get("predicate") != "held_by":
            continue
        holder = str(rel.get("object_id") or "")
        if holder.startswith("person_"):
            return holder
    return None


def _snapshot_ownership_clue(snapshot: dict[str, Any] | None) -> bool:
    """Return True when a single frame gives a clear actor/zone ownership cue."""
    if not snapshot:
        return False
    zone = str(snapshot.get("target_zone") or "")
    if zone in _OWNERSHIP_CLUE_ZONES:
        return True
    if snapshot.get("held_by"):
        return True
    return any(rel.get("predicate") == "held_by" for rel in (snapshot.get("relations") or []))


def _zone_bucket(zone: str) -> str:
    if zone == "ego_zone":
        return "ego"
    if zone in _SHARED_ZONES:
        return "shared"
    if zone == "other_person_zone":
        return "other"
    return "ambiguous"


def _classify_ownership_trajectory(
    start_zone: str,
    end_zone: str,
    start_held: str | None,
    end_held: str | None,
) -> str:
    start_bucket = _zone_bucket(start_zone)
    end_bucket = _zone_bucket(end_zone)
    if start_bucket != end_bucket and start_bucket != "ambiguous" and end_bucket != "ambiguous":
        return f"{start_bucket}_to_{end_bucket}"
    if start_held != end_held and (start_held or end_held):
        return "held_by_changed"
    return "position_changed"


def _derive_temporal_ownership_signals(
    snapshots: dict[str, dict[str, Any]],
    cross_moved: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered = [snapshots[tag] for tag in _TEMPORAL_FRAME_TAGS if tag in snapshots]
    if len(ordered) < 2:
        return {
            "object_moved": False,
            "held_by_changed": False,
            "zone_changed": False,
            "ownership_trajectory": None,
            "past_frame_ownership_clue": False,
            "current_frame_ownership_clue": False,
            "contextual_requires_history": False,
            "current_frame_zone": snapshots.get("t", {}).get("target_zone"),
        }

    zones = [snap["target_zone"] for snap in ordered]
    held = [snap.get("held_by") for snap in ordered]
    zone_changed = len(set(zones)) > 1
    held_by_changed = held[0] != held[-1] and bool(held[0] or held[-1])
    object_moved = bool(cross_moved)
    trajectory = (
        _classify_ownership_trajectory(zones[0], zones[-1], held[0], held[-1])
        if zone_changed or held_by_changed or object_moved
        else None
    )
    past_clue = any(_snapshot_ownership_clue(snapshots.get(tag)) for tag in ("t-2", "t-1"))
    current_snapshot = snapshots.get("t")
    current_clue = _snapshot_ownership_clue(current_snapshot)
    contextual_requires_history = past_clue and not current_clue
    return {
        "object_moved": object_moved,
        "held_by_changed": held_by_changed,
        "zone_changed": zone_changed,
        "ownership_trajectory": trajectory,
        "past_frame_ownership_clue": past_clue,
        "current_frame_ownership_clue": current_clue,
        "contextual_requires_history": contextual_requires_history,
        "current_frame_zone": (current_snapshot or {}).get("target_zone"),
    }

