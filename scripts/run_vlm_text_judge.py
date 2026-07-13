"""Text-only VLM judge — for candidates where we have rich narrations but no frames.

Ego4D FHO Observer-mode narrations explicitly name other people ("#O Man A
Moves a disposable plate"). Sending those + clip metadata (no images) to
Claude is enough to recover an ownership label for most clips.

Output schema matches the vision-judge runner: emits a SceneRecord-like
JSONL with `vlm_judgement` populated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import anthropic  # noqa: E402

from egoownership.schema import (  # noqa: E402
    ClipCandidate,
    FrameDetections,
    OwnershipLabel,
    SceneRecord,
    Taxonomy,
    VLMJudgement,
)

_SYS_PROMPT = (
    "You are an expert annotator for the Egocentric Implicit Ownership "
    "benchmark. Given only a narration sentence + clip metadata from a "
    "first-person video (NO IMAGES this time), decide who owns the target "
    "object referenced in the narration.\n\n"
    "Label space:\n"
    "  MINE     — owned by the camera wearer\n"
    "  PERSON_k — owned by another person mentioned in the narration\n"
    "  SHARED   — communal/table-center, not personally owned\n"
    "  AMBIGUOUS — symmetric/under-specified text, cannot decide\n\n"
    "Ego4D narration conventions:\n"
    "  '#O' prefix = observer mode (wearer watches someone else act)\n"
    "  '#C' prefix = camera-wearer is the actor\n"
    "  'Man A' / 'woman B' = visible non-wearer participants\n\n"
    "Cues for PERSON_k: object is held/moved/used by Man A/Woman B/etc.\n"
    "Cues for MINE: '#C', wearer's hand mentioned, wearer-side description.\n"
    "Cues for SHARED: communal table items (bread basket, salt, condiments).\n"
    "Output JSON strictly matching the schema. Confidence in [0,1]. Rationale "
    "must cite the exact narration phrase that drove your decision."
)

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["MINE", "PERSON_k", "SHARED", "AMBIGUOUS"]},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["label", "confidence", "rationale"],
    "additionalProperties": False,
}

_model = os.environ.get("EGOOWN_VLM_MODEL", "claude-jupiter-v1-p")


def _system_blocks(text: str) -> list[dict]:
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def judge_one(cand: ClipCandidate, client) -> tuple[str, dict | None, str | None]:
    user = (
        f"clip_id: {cand.clip_id}\n"
        f"dataset: {cand.dataset}  taxonomy: {cand.taxonomy.value}\n"
        f"verb: {cand.verb or '—'}\n"
        f"nouns: {', '.join(cand.nouns) or '—'}\n"
        f"narration: {cand.narration or '—'}\n"
        f"timestamps_sec: t-2={cand.t_minus_2_sec:.2f} t-1={cand.t_minus_1_sec:.2f} t={cand.t_sec:.2f}\n\n"
        "Output the JSON ownership decision now."
    )
    try:
        resp = client.messages.create(
            model=_model,
            max_tokens=500,
            system=_system_blocks(_SYS_PROMPT),
            messages=[{"role": "user", "content": [{"type": "text", "text": user}]}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        # No structured-output param on this SDK — strip fences and parse the
        # JSON the prompt asks for.
        import re as _re
        text = _re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=_re.MULTILINE)
        payload = json.loads(text)
        return cand.clip_id, payload, None
    except Exception as e:  # noqa: BLE001
        return cand.clip_id, None, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--side-jsonl", type=Path, default=None)
    args = ap.parse_args()

    side = args.side_jsonl or args.out.with_suffix(args.out.suffix + ".partial.jsonl")

    cands = []
    with args.candidates.open() as f:
        for line in f:
            cands.append(ClipCandidate.model_validate(json.loads(line)))
    if args.limit:
        cands = cands[: args.limit]
    print(f"Loaded {len(cands)} candidates")

    # Resume
    judged: dict[str, dict] = {}
    errors: dict[str, str] = {}
    if side.exists():
        with side.open() as f:
            for l in f:
                try:
                    rec = json.loads(l)
                    cid = rec["clip_id"]
                    if rec.get("judgement") is not None:
                        judged[cid] = rec["judgement"]
                    elif rec.get("error"):
                        errors[cid] = rec["error"]
                except Exception:  # noqa: BLE001
                    continue
    print(f"Resume: {len(judged)} done, {len(errors)} prior errors")

    todo = [c for c in cands if c.clip_id not in judged]
    print(f"To judge: {len(todo)}")

    client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY
    if todo:
        side_lock = threading.Lock()
        t0 = time()
        done = 0
        with side.open("a") as side_f, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(judge_one, c, client) for c in todo]
            for fut in as_completed(futures):
                cid, judgement, err = fut.result()
                with side_lock:
                    side_f.write(json.dumps({"clip_id": cid, "judgement": judgement, "error": err}) + "\n")
                    side_f.flush()
                done += 1
                if judgement:
                    judged[cid] = judgement
                elif err:
                    errors[cid] = err
                if done % 100 == 0 or done == len(todo):
                    elapsed = time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta_min = (len(todo) - done) / rate / 60 if rate > 0 else 0
                    print(
                        f"  done={done}/{len(todo)} judged={len(judged)} errors={len(errors)} "
                        f"rate={rate:.2f}/s eta_min={eta_min:.1f}",
                        flush=True,
                    )

    # Emit SceneRecord-style JSONL
    out_count = 0
    with args.out.open("w") as f:
        for c in cands:
            j = judged.get(c.clip_id)
            # Make a minimal SceneRecord (no frame data, no rule label)
            frames = [
                FrameDetections(tag="t-2", timestamp_sec=c.t_minus_2_sec, objects=[], persons=[]),
                FrameDetections(tag="t-1", timestamp_sec=c.t_minus_1_sec, objects=[], persons=[]),
                FrameDetections(tag="t", timestamp_sec=c.t_sec, objects=[], persons=[]),
            ]
            scene = SceneRecord(
                clip=c,
                frames=frames,
                scene_label=None,  # no rule cascade (no frames/bboxes)
                notes="text-only VLM judge (no frames)",
                auto_label_confidence=None,
            )
            if j:
                scene = scene.model_copy(update={
                    "vlm_judgement": VLMJudgement(
                        provider="anthropic",
                        model=_model,
                        label=OwnershipLabel(j.get("label", "AMBIGUOUS")),
                        confidence=float(j.get("confidence") or 0.0),
                        rationale=j.get("rationale"),
                    ),
                    "scene_label": OwnershipLabel(j.get("label", "AMBIGUOUS")),
                    "auto_label_confidence": float(j.get("confidence") or 0.0),
                })
            f.write(json.dumps(scene.model_dump(mode="json"), ensure_ascii=False) + "\n")
            out_count += 1
    print(f"\nWrote {out_count} scene records → {args.out}")
    print(f"  with VLM judgement: {len(judged)}, errors: {len(errors)}")


if __name__ == "__main__":
    main()
