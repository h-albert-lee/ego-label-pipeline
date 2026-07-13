"""human 검수 시트 — eval셋 기준. judge majority를 '제안 라벨'로 표시.

- 전체 검수 큐(3201): outputs/ego4d/review_queue.jsonl (승인/수정 툴링용)
- HTML 시트: 우선순위 상위 CAP개에 프레임 임베드 + 승인/수정 라디오
  우선순위: P2(low) 먼저 → 두 judge 합의(제안 명확) 먼저.
"""
import base64
import html
import json
from collections import Counter
from pathlib import Path

D = Path("outputs/ego4d")
FR = D / "crosscheck_eval_reconstructed_frames"
CAP = 80

ev = {json.loads(l)["id"]: json.loads(l) for l in open(D / "labels_v2_eval.jsonl") if l.strip()}
cc1 = {json.loads(l)["id"]: json.loads(l) for l in open(D / "crosscheck_eval.jsonl") if l.strip()}
cc2 = {json.loads(l)["id"]: json.loads(l) for l in open(D / "crosscheck_eval_gpt.jsonl") if l.strip()}
JC, JG = "claude-jupiter-v1-p", "gpt-5.4-mini"


def jl(row, jn):
    return (row["judges"].get(jn) or {}).get("label") if row else None


queue = []
for i, c1 in cc1.items():
    if c1["majority_agrees"]:
        continue  # claude가 GT와 일치 → 검수 불필요 (단, low는 아래서 포함)
    e = ev[i]
    claude, gpt = jl(c1, JC), jl(cc2.get(i), JG)
    agree2 = gpt is not None and claude == gpt
    proposed = claude if agree2 else (None if gpt else claude)
    queue.append({
        "id": i, "conf": e["gt_confidence"], "tier": e["subset_tier"],
        "gt": c1["auto_ground_truth"], "claude": claude, "gpt": gpt,
        "judges_agree": agree2, "proposed": proposed,
        "needs_human": not agree2,
        "caption": e.get("dense_caption_en"),
        "object": (e.get("object") or {}).get("label"),
    })
# low-conf 중 claude가 GT와 일치했던 것도 검수 큐(i)에 포함
for i, c1 in cc1.items():
    if c1["majority_agrees"] and ev[i]["gt_confidence"] == "low":
        e = ev[i]
        queue.append({
            "id": i, "conf": "low", "tier": e["subset_tier"], "gt": c1["auto_ground_truth"],
            "claude": jl(c1, JC), "gpt": None, "judges_agree": False,
            "proposed": c1["auto_ground_truth"], "needs_human": False,
            "caption": e.get("dense_caption_en"), "object": (e.get("object") or {}).get("label"),
        })

# 전체 큐 JSONL
with (D / "review_queue.jsonl").open("w", encoding="utf-8") as fh:
    for r in queue:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

# HTML 우선순위: low 먼저, 그다음 judges_agree(제안 명확) 먼저
prio = sorted(queue, key=lambda r: (r["conf"] != "low", not r["judges_agree"]))
sheet = prio[:CAP]


def frame_uri(rid, suffix):
    fp = FR / f"{rid.replace('#','__')}__{suffix}.jpg"
    if fp.exists():
        return "data:image/jpeg;base64," + base64.standard_b64encode(fp.read_bytes()).decode()
    return None


cards = []
for k, r in enumerate(sheet, 1):
    imgs = ""
    for suf, cap in [("t_2", "t-2"), ("t_1", "t-1"), ("t", "t")]:
        u = frame_uri(r["id"], suf)
        if u:
            imgs += f'<figure><img src="{u}"><figcaption>{cap}</figcaption></figure>'
    prop = r["proposed"] or "— (judge 불일치, 사람이 판정)"
    badge = "합의" if r["judges_agree"] else ("사람필수" if r["needs_human"] else "GT일치")
    opts = "".join(
        f'<label><input type=radio name=v{k}>{L}</label>'
        for L in ["MINE", "PERSON_k", "SHARED", "AMBIGUOUS"]
    )
    cards.append(f"""
    <div class=card>
      <h3>#{k} <span class=t>{html.escape(r['tier'])}/{html.escape(r['conf'])}</span>
          <span class=b>{badge}</span></h3>
      <div class=meta><code>{html.escape(r['id'])}</code><br>
        object: <b>{html.escape(str(r['object']))}</b> &nbsp;|&nbsp; caption: {html.escape(str(r['caption']))}</div>
      <div class=frames>{imgs}</div>
      <table>
        <tr><td class=k>파이프라인 GT</td><td>{html.escape(str(r['gt']))}</td></tr>
        <tr><td class=k>claude judge</td><td>{html.escape(str(r['claude']))}</td></tr>
        <tr><td class=k>gpt judge</td><td>{html.escape(str(r['gpt']))}</td></tr>
        <tr><td class=k>제안 라벨</td><td class=prop>{html.escape(str(prop))}</td></tr>
      </table>
      <div class=verdict>검수:
        <label><input type=radio name=v{k} checked>제안 승인</label> {opts}
      </div>
    </div>""")

counts = Counter((r["conf"], r["needs_human"]) for r in queue)
doc = f"""<!doctype html><meta charset=utf-8>
<title>EgoOwn 검수 시트 (eval)</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;max-width:900px;margin:1.5rem auto;padding:0 1rem;color:#1a1a1a}}
 .card{{border:1px solid #ddd;border-radius:10px;padding:1rem 1.2rem;margin:1.3rem 0}}
 h3{{margin:.2rem 0 .5rem}} .t{{font-size:12px;color:#666;background:#eef;padding:.1rem .4rem;border-radius:4px}}
 .b{{font-size:12px;background:#ffe9c7;padding:.1rem .4rem;border-radius:4px;margin-left:.3rem}}
 .meta{{font-size:12px;color:#555;margin-bottom:.6rem}} code{{font-size:10.5px}}
 .frames{{display:flex;gap:.5rem}} figure{{margin:0;flex:1}} img{{width:100%;border-radius:6px;border:1px solid #ccc}}
 figcaption{{text-align:center;font-size:11px;color:#777}}
 table{{border-collapse:collapse;width:100%;font-size:13px;margin:.5rem 0}} td{{border:1px solid #eee;padding:.3rem .5rem}}
 td.k{{width:130px;color:#666;font-weight:600}} .prop{{font-weight:700;color:#0a58ca}}
 .verdict{{border-top:1px dashed #ddd;padding-top:.5rem;font-size:13px}} .verdict label{{margin-right:.8rem}}
</style>
<h1>EgoOwn human 검수 시트 (eval 기준)</h1>
<p>전체 검수 큐 <b>{len(queue)}</b>개 (JSONL: review_queue.jsonl). 아래는 우선순위 상위 <b>{len(sheet)}</b>개
(low-conf 채굴건 + 두 judge 합의건 먼저). '제안 라벨'은 두 judge 다수결이며, 승인 또는 수정하세요.</p>
{''.join(cards)}
"""
(D / "review_sheet.html").write_text(doc, encoding="utf-8")
print(f"전체 검수 큐: {len(queue)} → {D/'review_queue.jsonl'}")
print(f"HTML 시트: 상위 {len(sheet)}개 → {D/'review_sheet.html'} ({(D/'review_sheet.html').stat().st_size//1024} KB)")
