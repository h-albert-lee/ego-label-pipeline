"""Persistence layer for the annotation server.

Backed by a single JSONL file plus a JSON activity log. We use a simple
``filelock`` so multiple workers — and multiple browser tabs — can hit the
same file without trampling each other.

The file format on disk is one ``SceneRecord`` per line. Reads stream the
file; writes load → mutate → atomic-rewrite (cheap because the pilot dataset
is on the order of hundreds of records, not millions).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from typing import Any

from egoownership.schema import AnnotationEdit, OwnershipLabel, SceneRecord, Taxonomy


def _record_matches_clip_id(rec: SceneRecord, clip_id: str) -> bool:
    if rec.clip.clip_id == clip_id:
        return True
    source = rec.clip.source or {}
    if str(source.get("label_row_id") or "") == clip_id:
        return True
    if str(source.get("base_clip_id") or "") == clip_id:
        return True
    return False


class SceneStore:
    def __init__(self, jsonl_path: Path, activity_path: Path | None = None):
        self.path = Path(jsonl_path)
        self.activity_path = Path(activity_path) if activity_path else self.path.with_suffix(
            ".activity.jsonl"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.activity_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")
        if not self.activity_path.exists():
            self.activity_path.write_text("", encoding="utf-8")
        self._lock = threading.Lock()

    # ---------- reads ----------

    def iter_records(self) -> Iterator[SceneRecord]:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield SceneRecord.model_validate(json.loads(line))

    def list_summaries(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for rec in self.iter_records():
            vlm_agrees: bool | None = None
            if rec.vlm_judgements and rec.vlm_majority_label:
                vlm_agrees = rec.vlm_majority_label == rec.scene_label
            out.append(
                {
                    "clip_id": rec.clip.clip_id,
                    "dataset": rec.clip.dataset,
                    "video_id": rec.clip.video_id,
                    "verb": rec.clip.verb,
                    "nouns": rec.clip.nouns,
                    "taxonomy": (rec.scene_taxonomy or rec.clip.taxonomy).value,
                    "scene_label": rec.scene_label.value if rec.scene_label else None,
                    "review_status": rec.review_status,
                    "auto_label_confidence": rec.auto_label_confidence,
                    "n_objects": sum(len(f.objects) for f in rec.frames),
                    "n_edits": len(rec.edits),
                    "has_vlm_judgement": bool(rec.vlm_judgements),
                    "vlm_agrees": vlm_agrees,
                    "vlm_agreement_ratio": rec.vlm_agreement_ratio,
                }
            )
        return out

    def get(self, clip_id: str) -> SceneRecord | None:
        for rec in self.iter_records():
            if _record_matches_clip_id(rec, clip_id):
                return rec
        return None

    # ---------- writes ----------

    def _atomic_rewrite(self, records: list[SceneRecord]) -> None:
        # Write to a temp file in the same directory, then rename.
        fd, tmp = tempfile.mkstemp(prefix=".scenes_", suffix=".jsonl", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(r.model_dump_json() + "\n")
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def update(
        self,
        clip_id: str,
        *,
        annotator: str,
        scene_label: OwnershipLabel | None = None,
        scene_taxonomy: Taxonomy | None = None,
        review_status: str | None = None,
        notes: str | None = None,
        object_overrides: dict[str, OwnershipLabel] | None = None,
    ) -> SceneRecord | None:
        with self._lock:
            records = list(self.iter_records())
            target_idx = next((i for i, r in enumerate(records) if _record_matches_clip_id(r, clip_id)), None)
            if target_idx is None:
                return None
            target = records[target_idx]
            edits: list[AnnotationEdit] = list(target.edits)

            if scene_label is not None and target.scene_label != scene_label:
                edits.append(
                    AnnotationEdit(
                        annotator=annotator,
                        field="scene_label",
                        old_value=target.scene_label.value if target.scene_label else None,
                        new_value=scene_label.value,
                    )
                )
                target = target.model_copy(update={"scene_label": scene_label})

            if scene_taxonomy is not None and (
                target.scene_taxonomy != scene_taxonomy
                and target.clip.taxonomy != scene_taxonomy
            ):
                edits.append(
                    AnnotationEdit(
                        annotator=annotator,
                        field="scene_taxonomy",
                        old_value=(target.scene_taxonomy or target.clip.taxonomy).value,
                        new_value=scene_taxonomy.value,
                    )
                )
                target = target.model_copy(update={"scene_taxonomy": scene_taxonomy})

            if review_status is not None and target.review_status != review_status:
                edits.append(
                    AnnotationEdit(
                        annotator=annotator,
                        field="review_status",
                        old_value=target.review_status,
                        new_value=review_status,
                    )
                )
                target = target.model_copy(update={"review_status": review_status})

            if notes is not None and target.notes != notes:
                edits.append(
                    AnnotationEdit(
                        annotator=annotator,
                        field="notes",
                        old_value=target.notes,
                        new_value=notes,
                    )
                )
                target = target.model_copy(update={"notes": notes})

            if object_overrides:
                new_frames = []
                for fd in target.frames:
                    new_objs = []
                    for o in fd.objects:
                        ov = object_overrides.get(o.instance_id) if o.instance_id else None
                        if ov is not None and o.ownership != ov:
                            edits.append(
                                AnnotationEdit(
                                    annotator=annotator,
                                    field=f"object:{o.instance_id}:ownership",
                                    old_value=o.ownership.value if o.ownership else None,
                                    new_value=ov.value,
                                )
                            )
                            new_objs.append(o.model_copy(update={"ownership": ov}))
                        else:
                            new_objs.append(o)
                    new_frames.append(fd.model_copy(update={"objects": new_objs}))
                target = target.model_copy(update={"frames": new_frames})

            target = target.model_copy(update={"edits": edits})
            records[target_idx] = target
            self._atomic_rewrite(records)
            self._append_activity(target, edits[-1] if edits else None)
            return target

    def update_evidence(
        self,
        clip_id: str,
        *,
        annotator: str,
        object_type: str | None = None,
        target_zone: str | None = None,
        relations: list[dict[str, Any]] | None = None,
        rationale: str | None = None,
        selected_evidence: list[str] | None = None,
    ) -> SceneRecord | None:
        """Update editable evidence fields.

        Object type / zone / relations edits re-derive the automatic label —
        relations is what the review UI uses to drop a specific held_by edge
        (e.g. a bystander's box that only coincidentally overlaps the target)
        without needing to touch zone/object_type to do it. Rationale and
        selected-evidence edits are free-form human edits and do not trigger
        label re-derivation by themselves.
        """
        with self._lock:
            records = list(self.iter_records())
            target_idx = next((i for i, r in enumerate(records) if _record_matches_clip_id(r, clip_id)), None)
            if target_idx is None:
                return None
            target = records[target_idx]

            ev: dict[str, Any] = dict(target.auto_key_evidence)
            old_object_type = ev.get("object_type")
            old_target_zone = ev.get("target_zone")
            old_relations = list(ev.get("relations") or [])
            old_rationale = ev.get("rationale")
            old_selected = list(ev.get("selected_evidence") or [])
            changed_decision_fields = False
            changed_freeform_fields = False
            if object_type is not None and old_object_type != object_type:
                ev["object_type"] = object_type
                changed_decision_fields = True
            if target_zone is not None and old_target_zone != target_zone:
                ev["target_zone"] = target_zone
                changed_decision_fields = True
            if relations is not None and old_relations != relations:
                ev["relations"] = relations
                changed_decision_fields = True
            if rationale is not None and old_rationale != rationale:
                ev["rationale"] = rationale
                changed_freeform_fields = True
            if selected_evidence is not None and old_selected != selected_evidence:
                ev["selected_evidence"] = selected_evidence
                changed_freeform_fields = True

            if not changed_decision_fields and not changed_freeform_fields:
                return target

            edits: list[AnnotationEdit] = list(target.edits)

            try:
                new_label = target.scene_label or OwnershipLabel.AMBIGUOUS
                new_taxonomy = target.scene_taxonomy or target.clip.taxonomy
                new_rationale = ev.get("rationale", "")

                if changed_decision_fields:
                    from egoownership.catv_evidence_label import _decide_taxonomy_gt

                    evidence_for_decision: dict[str, Any] = {
                        "target_object": ev.get("target_object", ""),
                        "object_type": ev["object_type"],
                        "target_zone": ev["target_zone"],
                        "caption_cues": ev.get("caption_cues") or {},
                        "relations": ev.get("relations") or [],
                        "person_count": ev.get("person_count", 0),
                        "temporal": ev.get("temporal") or {},
                    }
                    row_for_decision: dict[str, Any] = {"verb": ev.get("verb", ""), "nouns": []}
                    result = _decide_taxonomy_gt(row_for_decision, evidence_for_decision)

                    new_rationale = rationale if rationale is not None else result.get("rationale", "")
                    new_label_str = result.get("ground_truth", "AMBIGUOUS")
                    new_taxonomy_str = result.get("taxonomy", "D")
                    try:
                        new_label = OwnershipLabel(new_label_str)
                    except ValueError:
                        new_label = OwnershipLabel.AMBIGUOUS
                    try:
                        new_taxonomy = Taxonomy(new_taxonomy_str)
                    except ValueError:
                        new_taxonomy = Taxonomy.AMBIGUOUS
                ev["rationale"] = new_rationale

                new_evidence_strings = [new_rationale] if new_rationale else []
                for k in ("object_type", "target_zone", "verb", "person_count"):
                    v = ev.get(k)
                    if v is not None and v != "":
                        new_evidence_strings.append(f"{k}: {v}")

                target_object = ev.get("target_object", "target")
                ev["object_type_evidence"] = (
                    f"The target object '{target_object}' is categorized as {ev['object_type']}."
                )
                ev["zone_evidence"] = f"The target object is assigned to {ev['target_zone']}."
                if relations is not None:
                    from egoownership.catv_evidence_label import _summarize_relations

                    ev["relation_graph_evidence"] = _summarize_relations(ev.get("relations") or [])

                new_frames = []
                for fd in target.frames:
                    new_objs = [
                        o.model_copy(update={"ownership": new_label, "ownership_evidence": new_evidence_strings})
                        for o in fd.objects
                    ]
                    new_frames.append(fd.model_copy(update={"objects": new_objs}))

                edit_parts = []
                if changed_decision_fields:
                    edit_parts.append(
                        f"object_type={old_object_type}->{ev['object_type']}, "
                        f"target_zone={old_target_zone}->{ev['target_zone']}"
                    )
                if relations is not None and old_relations != relations:
                    edit_parts.append(f"relations edited ({len(old_relations)} -> {len(relations)} edges)")
                if rationale is not None and old_rationale != rationale:
                    edit_parts.append("rationale edited")
                if selected_evidence is not None and old_selected != selected_evidence:
                    edit_parts.append(f"selected_evidence={','.join(selected_evidence)}")
                edits.append(AnnotationEdit(
                    annotator=annotator,
                    field="auto_key_evidence",
                    old_value=f"object_type={old_object_type}, target_zone={old_target_zone}",
                    new_value="; ".join(edit_parts),
                ))

                target = target.model_copy(update={
                    "auto_key_evidence": ev,
                    "scene_label": new_label,
                    "scene_taxonomy": new_taxonomy,
                    "frames": new_frames,
                    "edits": edits,
                })
            except Exception:
                # Decision re-derivation failed — just persist the evidence change
                edits.append(AnnotationEdit(
                    annotator=annotator,
                    field="auto_key_evidence",
                    old_value=f"object_type={old_object_type}, target_zone={old_target_zone}",
                    new_value=(
                        f"object_type={ev.get('object_type')}, target_zone={ev.get('target_zone')}, "
                        f"rationale={'edited' if rationale is not None else 'unchanged'}"
                    ),
                ))
                target = target.model_copy(update={"auto_key_evidence": ev, "edits": edits})

            records[target_idx] = target
            self._atomic_rewrite(records)
            self._append_activity(target, edits[-1] if edits else None)
            return target

    def _append_activity(self, target: SceneRecord, edit: AnnotationEdit | None) -> None:
        if edit is None:
            return
        with self.activity_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "clip_id": target.clip.clip_id,
                        "annotator": edit.annotator,
                        "field": edit.field,
                        "old": edit.old_value,
                        "new": edit.new_value,
                    }
                )
                + "\n"
            )

    def recent_activity(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.activity_path.exists():
            return []
        lines = self.activity_path.read_text(encoding="utf-8").splitlines()
        out: list[dict[str, Any]] = []
        for line in lines[-limit:][::-1]:
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def stats(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        by_label: dict[str, int] = {}
        by_taxonomy: dict[str, int] = {}
        by_vlm_agreement: dict[str, int] = {}
        total = 0
        for rec in self.iter_records():
            total += 1
            by_status[rec.review_status] = by_status.get(rec.review_status, 0) + 1
            label = rec.scene_label.value if rec.scene_label else "UNLABELED"
            by_label[label] = by_label.get(label, 0) + 1
            tax = (rec.scene_taxonomy or rec.clip.taxonomy).value
            by_taxonomy[tax] = by_taxonomy.get(tax, 0) + 1
            if not rec.vlm_judgements or not rec.vlm_majority_label:
                vlm_key = "no_data"
            elif rec.vlm_majority_label == rec.scene_label:
                vlm_key = "agree"
            else:
                vlm_key = "disagree"
            by_vlm_agreement[vlm_key] = by_vlm_agreement.get(vlm_key, 0) + 1
        return {
            "total": total,
            "by_status": by_status,
            "by_label": by_label,
            "by_taxonomy": by_taxonomy,
            "by_vlm_agreement": by_vlm_agreement,
        }
