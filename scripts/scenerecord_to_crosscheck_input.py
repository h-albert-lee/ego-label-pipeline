"""SceneRecord JSONL(egoown label 출력) → vlm-crosscheck 평평 입력 포맷.

crosscheck는 Ego4D labels_v2 같은 평평한 행(id, auto_ground_truth, object.bbox,
frame_t*_path, temporal_target_objects, dense_caption_en)을 먹는다. EPIC/HD-EPIC은
egoown label이 SceneRecord(clip/frames/scene_label)를 내므로 여기서 변환한다.

target object: 각 SceneRecord의 frame t에는 라벨 대상 인스턴스가 정확히 1개
(scene_label == 그 object.ownership). 이를 target으로 삼는다.
frame_path는 frames_root(HF frames) 하위 상대경로 그대로 둔다.
"""
import argparse
import json
from pathlib import Path


def frame_by_tag(rec, tag):
    for f in rec.get("frames", []):
        if f.get("tag") == tag:
            return f
    return None


def convert(rec: dict) -> dict | None:
    clip = rec.get("clip", {})
    ft = frame_by_tag(rec, "t")
    if ft is None or not ft.get("objects"):
        return None
    obj = ft["objects"][0]  # target: frame t의 유일 인스턴스
    out = {
        "id": clip["clip_id"],
        "clip_id": clip["clip_id"],
        "video_id": clip.get("video_id"),
        "source_dataset": clip.get("dataset"),
        "dataset": clip.get("dataset"),
        "auto_ground_truth": rec.get("scene_label"),
        "auto_taxonomy": rec.get("scene_taxonomy"),
        "dense_caption_en": clip.get("narration"),
        "verb": clip.get("verb"),
        "nouns": clip.get("nouns"),
        "object": {"label": obj.get("label"), "bbox": obj.get("bbox")},
        "auto_key_evidence": {"rationale": rec.get("notes")},
        "temporal_target_objects": {},
    }
    # frame paths + t-1/t-2 bbox
    for tag, key in [("t-2", "frame_t_minus_2_path"),
                     ("t-1", "frame_t_minus_1_path"),
                     ("t", "frame_t_path")]:
        f = frame_by_tag(rec, tag)
        if f is not None:
            out[key] = f.get("frame_path")
            if tag != "t" and f.get("objects"):
                out["temporal_target_objects"][tag] = {"bbox": f["objects"][0].get("bbox")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    n = skipped = 0
    with args.input.open() as fin, args.out.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = convert(json.loads(line))
            if row is None:
                skipped += 1
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"변환 {n}행 (skip {skipped}) → {args.out}")


if __name__ == "__main__":
    main()
