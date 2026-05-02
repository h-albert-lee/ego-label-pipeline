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

from egoownership.schema import AnnotationEdit, OwnershipLabel, SceneRecord, Taxonomy


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
                }
            )
        return out

    def get(self, clip_id: str) -> SceneRecord | None:
        for rec in self.iter_records():
            if rec.clip.clip_id == clip_id:
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
            target_idx = next(
                (i for i, r in enumerate(records) if r.clip.clip_id == clip_id), None
            )
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
        total = 0
        for rec in self.iter_records():
            total += 1
            by_status[rec.review_status] = by_status.get(rec.review_status, 0) + 1
            label = rec.scene_label.value if rec.scene_label else "UNLABELED"
            by_label[label] = by_label.get(label, 0) + 1
            tax = (rec.scene_taxonomy or rec.clip.taxonomy).value
            by_taxonomy[tax] = by_taxonomy.get(tax, 0) + 1
        return {
            "total": total,
            "by_status": by_status,
            "by_label": by_label,
            "by_taxonomy": by_taxonomy,
        }
