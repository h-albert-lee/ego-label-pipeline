"""review_queue.jsonl(ego4d) + labels_v2_eval.jsonl → SceneRecord JSONL (egoown serve 입력).

egoown serve는 SceneRecord를 먹고 승인/수정을 파일락 write-through + edits
(annotator+timestamp) + activity 로그로 남긴다. 정적 HTML(결정 휘발)의 대체.

매핑:
  scene_label      ← 제안 라벨(proposed; judge majority). 없으면 auto GT.
  review_status    ← "in_review"(제안승인: 제안 라벨 검수 대기)
                     "draft"(사람판정: 두 judge 불일치, 제안 없음)
  frames[].frame_path ← crosscheck_eval_reconstructed_frames/{id}__{tag}.jpg
  objects[0].bbox  ← labels_v2_eval의 target bbox (t) / temporal (t-1,t-2)
  notes            ← tier/gt_confidence/votes/evidence_mode/원 GT JSON 직렬화
                     (검수 UI에서 판단 근거로 보이게)
"""
import argparse
import json
from pathlib import Path

FRAME_SUFFIX = {"t-2": "__t_2", "t-1": "__t_1", "t": "__t"}
VALID = {"MINE", "PERSON_k", "SHARED", "AMBIGUOUS"}


def build(qrow: dict, ev: dict, frames_dir_name: str) -> dict:
    rid = qrow["id"]
    fid = rid.replace("#", "__")
    times = ev.get("frame_times_sec") or {}
    ref_bbox = (ev.get("object") or {}).get("bbox")
    tobj = ev.get("temporal_target_objects") or {}
    noun = (ev.get("object") or {}).get("label") or (ev.get("nouns") or ["object"])[0]

    frames = []
    for tag in ("t-2", "t-1", "t"):
        bbox = ref_bbox if tag == "t" else (tobj.get(tag) or {}).get("bbox")
        objs = []
        if bbox:
            objs = [{"bbox": bbox, "label": noun, "instance_id": f"target_{tag}"}]
        frames.append({
            "tag": tag,
            "timestamp_sec": float(times.get(tag, {"t-2": 0.0, "t-1": 1.0, "t": 2.0}[tag])),
            "frame_path": f"{fid}{FRAME_SUFFIX[tag]}.jpg",
            "objects": objs,
        })

    proposed = qrow.get("proposed")
    scene_label = proposed if proposed in VALID else (qrow.get("gt") if qrow.get("gt") in VALID else None)
    review_status = "in_review" if not qrow.get("needs_human") else "draft"

    notes = json.dumps({
        "queue_group": "approve-only" if not qrow.get("needs_human") else "human-decide",
        "auto_gt": qrow.get("gt"),
        "tier": qrow.get("tier"),
        "gt_confidence": qrow.get("conf"),
        "proposed": proposed,
        "judges": {"claude_frames": qrow.get("claude"), "gpt_frames": qrow.get("gpt"),
                   "jupiter_narration": qrow.get("judge3_narration")},
        "votes": qrow.get("votes"),
        "resolved_by": qrow.get("resolved_by"),
        "evidence_mode": "frames-only(1,2)/with-narration(3)",
    }, ensure_ascii=False)

    return {
        "clip": {
            "dataset": qrow.get("source_dataset", "ego4d"),
            "clip_id": rid,
            "video_id": ev.get("video_id"),
            "taxonomy": ev.get("auto_taxonomy") or "C",
            "t_minus_2_sec": float(times.get("t-2", 0.0)),
            "t_minus_1_sec": float(times.get("t-1", 1.0)),
            "t_sec": float(times.get("t", 2.0)),
            "verb": ev.get("verb"),
            "nouns": ev.get("nouns") or [],
            "narration": ev.get("dense_caption_en"),
        },
        "frames": frames,
        "scene_label": scene_label,
        "notes": notes,
        "review_status": review_status,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", type=Path, default=Path("outputs/ego4d/review_queue.jsonl"))
    ap.add_argument("--eval", type=Path, default=Path("outputs/ego4d/labels_v2_eval.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("outputs/ego4d/review_scenerecords.jsonl"))
    ap.add_argument("--frames-dir-name", default="crosscheck_eval_reconstructed_frames")
    ap.add_argument("--source", default="ego4d", help="source_dataset to include")
    args = ap.parse_args()

    ev = {json.loads(l)["id"]: json.loads(l) for l in open(args.eval) if l.strip()}
    n = skipped = 0
    from collections import Counter
    rs = Counter()
    with args.out.open("w", encoding="utf-8") as fh:
        for line in open(args.queue):
            if not line.strip():
                continue
            q = json.loads(line)
            if q.get("source_dataset", "ego4d") != args.source:
                continue
            if q["id"] not in ev:
                skipped += 1
                continue
            rec = build(q, ev[q["id"]], args.frames_dir_name)
            rs[rec["review_status"]] += 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"변환 {n}행 (skip {skipped}) → {args.out}")
    print(f"review_status: {dict(rs)}")


if __name__ == "__main__":
    main()
