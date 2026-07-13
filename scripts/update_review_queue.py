"""3차 재adjudication 결과로 review_queue.jsonl 갱신.

해소된 행(1627): needs_human=False, proposed=3judge majority, resolved_by 스탬프.
여전히 갈린 행(510): needs_human=True 유지, 3차 표(J3) 기록.
evidence_mode 필드 추가.
"""
import json
from pathlib import Path

D = Path("outputs/ego4d")
resolved = json.loads(Path("/tmp/claude-1000/-home-gpuadmin-albert/2d6f65bb-9ea0-4f24-bd19-b48dca9a8ac3/scratchpad/resolved3.json").read_text())
cc3 = {json.loads(l)["id"]: json.loads(l) for l in open(D / "crosscheck_eval_narr.jsonl") if l.strip()}
J3 = "claude-jupiter-v1-p"

rows = [json.loads(l) for l in open(D / "review_queue.jsonl") if l.strip()]
out = []
for r in rows:
    i = r["id"]
    # 3차 표 기록 (미해결이었던 것만 cc3에 존재)
    if i in cc3:
        r["judge3_narration"] = (cc3[i]["judges"].get(J3) or {}).get("label")
        r["evidence_mode_stage3"] = "with-narration"
    if i in resolved:
        info = resolved[i]
        r["proposed"] = info["proposed"]
        r["needs_human"] = False
        r["resolved_by"] = "3judge-majority"
        r["votes"] = info["votes"]  # [J1 frames, J2 frames, J3 narration]
    out.append(r)

with (D / "review_queue.jsonl").open("w", encoding="utf-8") as fh:
    for r in out:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

from collections import Counter
c = Counter(("사람판정" if r["needs_human"] else "승인만") for r in out)
print(f"갱신 완료: {len(out)}행 → review_queue.jsonl")
print(f"  {dict(c)}")
print(f"  resolved_by=3judge-majority: {sum(1 for r in out if r.get('resolved_by'))}")
