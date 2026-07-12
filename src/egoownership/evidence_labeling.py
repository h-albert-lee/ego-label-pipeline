"""Merge object captions with lightweight visual evidence.

This module is intentionally narrower than the original ``detect`` pipeline:
the target object box already comes from SAM-3/SAM-2, so we only add the
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


PersonDetector = Callable[[Path], tuple[list[PersonDetection], BBox | None]]
DecisionFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
_HELD_BY_WEARER_CONTAINMENT = 0.5

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
    "cutlery",
    "utensil",
    "teapot",
    "kettle",
    "pan",
    "frypan",
    "saucepan",
    "cooker",
    "hotpot",
    "chopboard",
    "tablecloth",
    "coaster",
    "spatula",
    "opener",
}
_PERSONAL_OBJECTS = {
    "phone",
    "smartphone",
    "cellphone",
    "iphone",
    "ipad",
    "mac",
    "laptop",
    "computer",
    "tablet",
    "notebook",
    "pen",
    "pencil",
    "wallet",
    "purse",
    "handbag",
    "key",
    "keys",
    "card",
    "bag",
    "watch",
    "eyeglass",
    "charger",
    "cable",
    "usb",
    "device",
    "camera",
    "microphone",
    "headphone",
    "earphone",
    "keyboard",
    "mouse",
    "monitor",
    "comb",
    "razor",
    "lotion",
    "perfume",
    "cigarette",
    "lunchbox",
    "money",
    "ticket",
    # Drinkware — a personal-function item like a phone or wallet: no
    # ownership signal from function alone (rule 3), so it defaults to
    # AMBIGUOUS absent a clue rather than SHARED. An individual cup/bottle
    # can still resolve to SHARED via an explicit "shared_use" caption cue.
    "cup",
    "mug",
    "tumbler",
    "bottle",
    "jar",
    "straw",
}
# Ownership follows current physical location once the action completes —
# whoever's zone the object ends up in at t is the new owner.
_TRANSFER_VERBS = {
    "give",
    "pass",
    "hand",
    "receive",
    "serve",
    "offer",
    "return",
}
# Ownership stays with whoever *lent* the object — physical possession at t
# is the opposite of who owns it (holding ≠ owning).
_TEMPORARY_USE_VERBS = {
    "borrow",
    "lend",
    "loan",
}
_OWNERSHIP_HISTORY_VERBS = _TRANSFER_VERBS | _TEMPORARY_USE_VERBS
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
    ego_hand_bbox: BBox | None = None
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
            persons, ego_hand_bbox = person_detector(frame_path) if person_detector is not None else ([], None)
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
        wearer_relation = _held_by_wearer_relation(target_bbox, ego_hand_bbox)
        if wearer_relation is not None:
            relation_summary.append(wearer_relation)

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

    key = _build_key_evidence(row, evidence, fallback_label=label)

    _zone_desc = {
        "ego_zone": "the camera-wearer's near zone",
        "other_person_zone": "another person's spatial zone",
        "shared_zone": "the shared table zone",
        "center_table": "the central table area",
    }.get(zone, f"the {zone} area")
    _obj_type_desc = {
        "shared": "a shared-use object",
        "personal": "a personal object",
        "generic_object": "a generic object",
    }.get(object_type, object_type)

    temporal = evidence.get("temporal") or {}
    contextual = verb in _OWNERSHIP_HISTORY_VERBS or bool(temporal.get("contextual_requires_history"))
    conflict = cues["conflict_cue"]
    if object_type == "shared" and zone == "ego_zone":
        conflict = True
    if object_type == "personal" and zone in {"shared_zone", "other_person_zone"}:
        conflict = True
    if cues["ego_actor"] and cues["other_actor"]:
        # Caption describes both the wearer and another person acting on the
        # target — a genuine contradiction, not just "no signal", so it
        # shouldn't look identical to a caption that said nothing at all.
        conflict = True

    gt: OwnershipLabel | None = None
    rationale = ""

    # ---- Tier 0: functional invariant — outranks everything below,
    # including an explicitly-ambiguous caption, since a shared object's
    # label doesn't depend on who's currently touching it.
    explicit_shared_gt = cues["shared_use"] or object_type == "shared"
    if explicit_shared_gt:
        gt = OwnershipLabel.SHARED
        rationale = (
            f"The target is {_obj_type_desc} and the description indicates shared or communal use, "
            f"placing it in {_zone_desc}."
        )

    if gt is None and cues["ambiguous"] and not (cues["ego_actor"] or cues["other_actor"]):
        return _decision(
            Taxonomy.AMBIGUOUS, OwnershipLabel.AMBIGUOUS, key,
            f"The description mentions {_obj_type_desc} but actor identity is explicitly ambiguous "
            f"with no clear ownership signal.",
        )

    # ---- Tier 2: caption history-verb, resolved via *zone* rather than the
    # (noisier) actor-cue text — checked before the plainer relation/actor
    # tiers below, since an explicit transfer/loan verb is the most specific
    # signal available when present. Transfer verbs (give/pass/receive/
    # return/...) mean ownership follows wherever the object ends up;
    # temporary-use verbs (borrow/lend/loan) mean the opposite — physical
    # possession right now is on loan, not owned.
    if gt is None and verb in _TRANSFER_VERBS:
        if zone == "ego_zone":
            gt = OwnershipLabel.MINE
            rationale = (
                f"The description describes a transfer ('{verb}'), and the target now sits in {_zone_desc}, "
                f"so the camera wearer is the resulting owner."
            )
        elif zone == "other_person_zone":
            gt = OwnershipLabel.PERSON_K
            rationale = (
                f"The description describes a transfer ('{verb}'), and the target now sits in {_zone_desc}, "
                f"so that visible person is the resulting owner."
            )
    elif gt is None and verb in _TEMPORARY_USE_VERBS:
        if zone == "ego_zone":
            gt = OwnershipLabel.PERSON_K
            rationale = (
                f"The description describes temporary use ('{verb}'); the camera wearer currently holds the "
                f"target in {_zone_desc}, but holding is not owning — it appears to be on loan from another person."
            )
        elif zone == "other_person_zone":
            gt = OwnershipLabel.MINE
            rationale = (
                f"The description describes temporary use ('{verb}'); another person currently holds the target "
                f"in {_zone_desc}, but holding is not owning — it appears to be the camera wearer's own item, lent out."
            )

    # ---- Tier 1: direct relation-graph possession. A specific person's (or
    # the wearer's own, via the dedicated ego-hand box) hand currently
    # overlapping the target is stronger, more current evidence than a bare
    # caption mention of who "the actor" is.
    if gt is None:
        # The dedicated ego-hand box is a more specific, purpose-built signal
        # than the generic person-hand-zone heuristic (lower 40% of a whole
        # detected body), so it takes priority if both happen to overlap the
        # target — not just whichever relation appears first in the list.
        held_by_wearer = any(
            rel.get("predicate") == "held_by" and rel.get("object_id") == "wearer" for rel in relations
        )
        held_by_person = any(
            rel.get("predicate") == "held_by" and str(rel.get("object_id") or "").startswith("person_")
            for rel in relations
        )
        if held_by_wearer:
            gt = OwnershipLabel.MINE
            rationale = (
                f"The target spatially overlaps the camera wearer's own detected hand region, "
                f"suggesting the wearer is holding or primarily using it."
            )
        elif held_by_person:
            gt = OwnershipLabel.PERSON_K
            rationale = (
                f"The target spatially overlaps with another person's hand region, "
                f"suggesting it is held or primarily used by that person."
            )

    # ---- Plain caption actor cue (no history verb involved).
    if gt is None and cues["other_actor"] and not cues["ego_actor"]:
        gt = OwnershipLabel.PERSON_K
        rationale = (
            f"Another visible person is the primary actor interacting with the target, "
            f"while the camera wearer is not described as acting on it. "
            f"The object appears in {_zone_desc}."
        )
    elif gt is None and cues["ego_actor"] and not cues["other_actor"]:
        gt = OwnershipLabel.MINE
        rationale = (
            f"The camera wearer is the primary actor in the description and is directly interacting with the target. "
            f"The object is located in {_zone_desc}."
        )

    # ---- Tier 3: zone fallback, with a temporal abandonment check: a
    # personal object relocated to the shared zone with no current actor
    # cue stays MINE if it was in the wearer's zone earlier in the clip —
    # relocating it doesn't relinquish ownership.
    if gt is None:
        # object_type == "shared" can't reach here — Tier 0 already resolved
        # it unconditionally above — so the shared-zone branches below only
        # ever see "personal" or "generic_object".
        if zone == "other_person_zone":
            gt = OwnershipLabel.PERSON_K
            rationale = (
                f"The target is located in {_zone_desc}, close to another visible person, "
                f"with no strong actor cue pointing to the camera wearer."
            )
        elif zone == "ego_zone":
            gt = OwnershipLabel.MINE
            rationale = (
                f"The target is in {_zone_desc}, close to the camera wearer, "
                f"and no actor cue points to another person."
            )
        elif zone in {"shared_zone", "center_table"} and object_type == "personal" and temporal.get("ownership_trajectory") == "ego_to_shared":
            gt = OwnershipLabel.MINE
            contextual = True
            rationale = (
                f"The target is {_obj_type_desc} now resting in {_zone_desc} with no current actor cue, but it "
                f"was in the camera-wearer's zone earlier in the clip — relocating it to a shared surface doesn't "
                f"relinquish ownership."
            )
        elif zone in {"shared_zone", "center_table"}:
            gt = OwnershipLabel.AMBIGUOUS
            rationale = (
                f"The target lies in {_zone_desc} but actor evidence is weak, "
                f"making ownership difficult to determine from spatial position alone."
            )
        else:
            gt = OwnershipLabel.AMBIGUOUS
            rationale = (
                f"There is insufficient actor, object-type, or spatial evidence to assign ownership confidently. "
                f"The target ({_obj_type_desc}) does not fall clearly into any person's zone."
            )

    # ---- Past-frame fallback: only reached if everything above still left
    # us with AMBIGUOUS.
    used_past_frame_fallback = False
    if gt is OwnershipLabel.AMBIGUOUS:
        past = _gt_from_past_frame_clue(temporal)
        if past is not None:
            gt, past_tag = past
            used_past_frame_fallback = True
            contextual = True
            past_zone_desc = "the camera-wearer's zone" if gt is OwnershipLabel.MINE else "another visible person's zone"
            rationale = (
                f"The target gives no clear zone or actor evidence at the current frame ({_zone_desc}), "
                f"but it was in {past_zone_desc} at frame {past_tag}, so ownership is carried forward from that clue."
            )

    if gt is OwnershipLabel.AMBIGUOUS:
        tax = Taxonomy.AMBIGUOUS
    elif contextual:
        tax = Taxonomy.CONTEXTUAL
        if (
            not used_past_frame_fallback
            and temporal.get("contextual_requires_history")
            and verb not in _OWNERSHIP_HISTORY_VERBS
        ):
            past_tags = [tag for tag in ("t-2", "t-1") if tag in (temporal.get("frame_snapshots") or {})]
            rationale = (
                f"{rationale} An ownership clue is visible in past frames ({', '.join(past_tags)}) "
                f"but not at the action moment (current zone: {temporal.get('current_frame_zone', zone)})."
            )
    elif conflict:
        tax = Taxonomy.CONFLICT
    else:
        tax = Taxonomy.BASELINE
    return _decision(tax, gt, key, rationale)


def _decision(
    taxonomy: Taxonomy,
    gt: OwnershipLabel,
    key: dict[str, Any],
    rationale: str,
) -> dict[str, Any]:
    return {
        "taxonomy": taxonomy.value,
        "ground_truth": gt.value,
        "key_evidence": {**key, "rationale": _sentence(rationale) or rationale},
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
    '"rationale": "one or two sentences"}'
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
    rationale = str(data.get("rationale") or "").strip() or "LLM decision (no rationale given)."
    return {
        "taxonomy": taxonomy,
        "ground_truth": ground_truth,
        "rationale": rationale,
    }


def _compact_prose(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _sentence(value: Any) -> str:
    text = _compact_prose(value)
    if not text:
        return ""
    if text[-1] not in ".!?":
        text += "."
    return text


def _is_ego4d_row(row: dict[str, Any]) -> bool:
    return str(row.get("dataset") or row.get("source_dataset") or "").lower() in {"ego4d", "ego4d_fho"}


def _narration_actor_prefix(text: Any) -> str | None:
    """Ego4D narration convention: a leading ``#C`` tag means the camera
    wearer is the narrated actor; a leading ``#O`` (also seen as ``#O2``,
    ``#OO``, ... for multiple observed people) means another visible person
    is. This is a direct, reliable signal — stronger than scanning the text
    for English phrases like "camera wearer", which raw Ego4D narrations
    (e.g. "#O man E moves bag on the table") never actually contain.
    """
    match = re.match(r"^\s*#([A-Za-z]+)", str(text or ""))
    if not match:
        return None
    tag = match.group(1).upper()
    if tag == "C":
        return "ego"
    if tag.startswith("O"):
        return "other"
    return None


def _object_caption_interaction_sentence(row: dict[str, Any]) -> str:
    """First sentence of object_caption's "(3) who interacts with HO" section,
    or "" if absent/unparseable."""
    object_caption = str(row.get("object_caption") or "")
    match = re.search(r"\(3\)\s*(.*?)(?=\n\s*\(\d+\)|$)", object_caption, flags=re.DOTALL)
    if not match:
        return ""
    first_line = next((line.strip() for line in match.group(1).splitlines() if line.strip()), "")
    if not first_line:
        return ""
    first_sentence = re.split(r"(?<=[.!?])\s+", first_line, maxsplit=1)[0]
    return _sentence(first_sentence)


def _caption_interaction_evidence(row: dict[str, Any]) -> str:
    # object_caption (catv caption) is primary: it's sometimes correct even
    # on ego4d, where the narration is used as a fallback only when
    # object_caption gives nothing parseable.
    text = _object_caption_interaction_sentence(row)
    if text:
        return text

    for key in ("dense_caption_en", "transcript"):
        text = _sentence(row.get(key))
        if text:
            return text

    return "No caption interaction evidence was recorded for this target."


def _summarize_relations(relations: list[dict[str, Any]]) -> str:
    if not relations:
        return "No explicit relation graph edge was recorded for the target object."
    parts: list[str] = []
    for rel in relations[:6]:
        predicate = rel.get("predicate") or "related_to"
        obj = rel.get("object_id") or "unknown"
        note = rel.get("note")
        if predicate == "held_by" and obj == "wearer":
            phrase = "The target object is held by or overlaps the camera wearer's own detected hand"
        elif predicate == "held_by":
            phrase = f"The target object is held by or overlaps the hand zone of {obj}"
        elif predicate == "on_shared_band":
            phrase = "The target object lies in the shared table band"
        elif predicate == "next_to":
            phrase = f"The target object is spatially next to {obj}"
        else:
            phrase = f"The target object has relation '{predicate}' with {obj}"
        if note:
            phrase += f", with note '{note}'"
        parts.append(phrase)
    return _sentence(" ".join(_sentence(part) for part in parts))


def _summarize_context_change(temporal: dict[str, Any]) -> str:
    snapshots = temporal.get("frame_snapshots") if isinstance(temporal, dict) else None
    if not isinstance(snapshots, dict):
        return "No separate t-2/t-1 context change evidence was recorded."

    ordered = [(tag, snapshots.get(tag)) for tag in ("t-2", "t-1", "t")]
    snippets: list[str] = []
    for tag, snap in ordered:
        # A missing snapshot means that frame was never analyzed (e.g. SAM-2
        # tracking lost the object) — distinct from an analyzed frame with an
        # unresolved zone, so it must be omitted rather than reported as
        # "unknown_zone" / "not held", which would fabricate evidence for a
        # frame we have no data for at all.
        if not isinstance(snap, dict) or not snap:
            continue
        zone = snap.get("target_zone") or "unknown_zone"
        held_by = snap.get("held_by") or "not_held"
        snippets.append(f"At {tag}, the target is in {zone} and is {held_by if held_by != 'not_held' else 'not held by a detected person'}")

    changes: list[str] = []
    t2 = snapshots.get("t-2")
    t1 = snapshots.get("t-1")
    t = snapshots.get("t")
    # Truthiness (not just isinstance-dict) matters here too — an empty dict
    # from a frame that was never analyzed must not be compared as if it were
    # real (unchanged) data.
    if isinstance(t2, dict) and t2 and isinstance(t, dict) and t:
        if t2.get("target_zone") != t.get("target_zone"):
            changes.append(f"Across the sparse frames, the zone changes from {t2.get('target_zone') or 'unknown'} to {t.get('target_zone') or 'unknown'}")
        if t2.get("held_by") != t.get("held_by"):
            changes.append(f"Across the sparse frames, the holder changes from {t2.get('held_by') or 'not_held'} to {t.get('held_by') or 'not_held'}")
    if isinstance(t1, dict) and t1 and isinstance(t, dict) and t and t1.get("held_by") != t.get("held_by"):
        changes.append(f"Between t-1 and t, the holder changes from {t1.get('held_by') or 'not_held'} to {t.get('held_by') or 'not_held'}")

    base = " ".join(_sentence(snippet) for snippet in snippets)
    if changes:
        base += " " + " ".join(_sentence(change) for change in changes)
    return base


def _build_key_evidence(
    row: dict[str, Any], evidence: dict[str, Any], *, fallback_label: str = "object"
) -> dict[str, Any]:
    """Prose-form summary of the evidence actually available for a decision.

    Same shape the review server builds on the fly for its evidence panel —
    generated once here so ``auto_key_evidence`` in labels.jsonl is already
    complete and human-readable, instead of a decision-agnostic snapshot of
    a few raw fields that leaves out exactly the evidence (relations,
    temporal) that decides many rows.
    """
    object_type = evidence.get("object_type", "generic_object")
    target_zone = evidence.get("target_zone", "background_or_ambiguous_zone")
    relations = evidence.get("relations") or []
    temporal = evidence.get("temporal") or {}
    target_object = evidence.get("target_object", fallback_label)

    nearest = evidence.get("nearest_other_person") or {}
    nearest_text = ""
    if isinstance(nearest, dict) and nearest:
        nearest_text = (
            f" The nearest other-person cue is {nearest.get('person_id', 'unknown')} "
            f"at distance {nearest.get('distance', 'unknown')}."
        )

    zone_evidence = _sentence(
        f"The target object is assigned to {target_zone}."
        f"{nearest_text} Visible other person is {evidence.get('visible_other_person', False)}"
    )
    object_type_evidence = _sentence(f"The target object '{target_object}' is categorized as {object_type}")

    return {
        "target_object": target_object,
        "object_type": object_type,
        "object_type_evidence": object_type_evidence,
        "target_zone": target_zone,
        "zone_evidence": zone_evidence,
        "caption_evidence": _caption_interaction_evidence(row),
        "relation_graph_evidence": _summarize_relations(relations if isinstance(relations, list) else []),
        "context_change_evidence": _summarize_context_change(temporal if isinstance(temporal, dict) else {}),
        "verb": normalize_token(str(row.get("verb") or "")),
        "person_count": evidence.get("person_count", 0),
        "caption_cues": evidence.get("caption_cues") or {},
        "relations": relations,
        "temporal": temporal,
    }


def _object_caption_actor(object_caption_text: str) -> str | None:
    """Actor claim from object_caption's own keyword match, or None if it
    makes no claim (or claims both, which is itself not a clean signal)."""
    has_ego = any(p in object_caption_text for p in _EGO_PATTERNS)
    has_other = any(p in object_caption_text for p in _OTHER_PATTERNS)
    if has_ego and not has_other:
        return "ego"
    if has_other and not has_ego:
        return "other"
    return None


def _caption_cues(row: dict[str, Any]) -> dict[str, bool]:
    is_ego4d = _is_ego4d_row(row)
    narration_text = " ".join(str(row.get(key) or "") for key in ("dense_caption_en", "transcript")).lower()
    object_caption_text = str(row.get("object_caption") or "").lower()
    ego_actor = other_actor = False
    if is_ego4d:
        # Cross-check narration's #C/#O tag against object_caption's own
        # keyword-based actor claim, computed independently. When they
        # agree, that's a high-confidence signal (two independently
        # generated sources converging). When they disagree, neither is
        # trusted here -- gt falls through to the zone tier instead (see
        # _decide_taxonomy_gt). When only one has a signal, use it.
        text = " ".join([object_caption_text, narration_text])
        narration_actor = _narration_actor_prefix(row.get("dense_caption_en"))
        object_actor = _object_caption_actor(object_caption_text)
        if narration_actor and object_actor:
            if narration_actor == object_actor:
                ego_actor = narration_actor == "ego"
                other_actor = narration_actor == "other"
            # else: disagreement -- leave both False, deferring to zone.
        elif narration_actor:
            ego_actor = narration_actor == "ego"
            other_actor = narration_actor == "other"
        elif object_actor:
            ego_actor = object_actor == "ego"
            other_actor = object_actor == "other"
    else:
        text = " ".join(
            str(row.get(key) or "") for key in ("object_caption", "dense_caption_en", "transcript_en", "qwen_translation")
        ).lower()
        ego_actor = any(p in text for p in _EGO_PATTERNS)
        other_actor = any(p in text for p in _OTHER_PATTERNS)
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
        "ego_actor": ego_actor,
        "other_actor": other_actor,
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
    ego_hand_by_tag: dict[str, BBox | None] = {}
    target_bbox_by_tag: dict[str, BBox] = {}

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
        ego_hand_bbox: BBox | None = None
        if person_detector is not None and frame_path is not None and frame_path.exists():
            try:
                persons, ego_hand_bbox = person_detector(frame_path)
            except Exception as exc:  # noqa: BLE001
                detector_error = f"{type(exc).__name__}: {str(exc)[:300]}"
        ego_hand_by_tag[tag] = ego_hand_bbox
        target_bbox_by_tag[tag] = bbox
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
        wearer_relation = _held_by_wearer_relation(
            target_bbox_by_tag.get(fd.tag, target_obj.bbox), ego_hand_by_tag.get(fd.tag)
        )
        if wearer_relation is not None:
            relations.append(wearer_relation)
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
    """Return the holder of a ``held_by`` relation — a ``person_*`` id, or
    ``"wearer"`` when the target overlaps the camera wearer's own hand."""
    for rel in relations:
        if rel.get("predicate") != "held_by":
            continue
        holder = str(rel.get("object_id") or "")
        if holder.startswith("person_") or holder == "wearer":
            return holder
    return None


def _held_by_wearer_relation(target_bbox: BBox, ego_hand_bbox: BBox | None) -> dict[str, Any] | None:
    """Containment check against the wearer's own detected hand/arm box.

    The ego-hand box comes from Grounding DINO's "a person." prompt firing on
    an arm reaching into frame — it's a loose region covering the whole
    reach, not a tight hand-only box, so it's often much larger than a small
    held object. Plain IoU penalizes that size mismatch too harshly (a phone
    fully inside a wide forearm box can score IoU < 0.05); what actually
    matters is how much of the *target* sits inside it.
    """
    if ego_hand_bbox is None or target_bbox.area <= 0:
        return None
    ix1, iy1 = max(target_bbox.x_min, ego_hand_bbox.x_min), max(target_bbox.y_min, ego_hand_bbox.y_min)
    ix2, iy2 = min(target_bbox.x_max, ego_hand_bbox.x_max), min(target_bbox.y_max, ego_hand_bbox.y_max)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    containment = inter / target_bbox.area
    if containment <= _HELD_BY_WEARER_CONTAINMENT:
        return None
    return {"subject_id": "target", "object_id": "wearer", "predicate": "held_by", "score": containment}


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


def _gt_from_past_frame_clue(temporal: dict[str, Any]) -> tuple[OwnershipLabel, str] | None:
    """When frame ``t`` alone gives no ownership clue, fall back to the
    nearest past frame (``t-1``, then ``t-2``) that had one, using that
    frame's zone/holder rather than giving up as AMBIGUOUS. Requires
    ``--sam2-track`` to have populated real t-2/t-1 boxes; absent that,
    ``frame_snapshots`` only ever has (at most) a ``"t"`` entry and this
    always returns ``None``.
    """
    snapshots = temporal.get("frame_snapshots") or {}
    for tag in ("t-1", "t-2"):
        snapshot = snapshots.get(tag)
        if not _snapshot_ownership_clue(snapshot):
            continue
        held_by = snapshot.get("held_by")
        if held_by == "wearer":
            return OwnershipLabel.MINE, tag
        if held_by:
            return OwnershipLabel.PERSON_K, tag
        zone = str(snapshot.get("target_zone") or "")
        if zone == "ego_zone":
            return OwnershipLabel.MINE, tag
        if zone == "other_person_zone":
            return OwnershipLabel.PERSON_K, tag
    return None


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
