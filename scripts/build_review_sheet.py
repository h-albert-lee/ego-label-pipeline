"""human 검수 HTML 시트 — review_queue.jsonl(신뢰소스)을 읽어 렌더링만 한다.

review_queue.jsonl은 3-judge 재adjudication까지 반영된 최신본. 이 스크립트는
queue를 재생성하지 않고 그대로 읽어 우선순위 상위 CAP개에 프레임을 임베드한다.
  우선순위: needs_human(사람판정) 먼저 → low-conf 먼저.
'proposed'를 제안 라벨로 표시하고 승인/수정 UI를 붙인다.
"""
import base64
import html
import json
from pathlib import Path

D = Path("outputs/ego4d")
FR = D / "crosscheck_eval_reconstructed_frames"
CAP = 80

queue = [json.loads(l) for l in open(D / "review_queue.jsonl") if l.strip()]


def frame_uri(rid, suffix):
    fp = FR / f"{rid.replace('#','__')}__{suffix}.jpg"
    if fp.exists():
        return "data:image/jpeg;base64," + base64.standard_b64encode(fp.read_bytes()).decode()
    return None


# 우선순위: 사람판정(무proposal) 먼저 → low-conf 먼저
prio = sorted(queue, key=lambda r: (not r.get("needs_human"), r.get("conf") != "low"))
sheet = prio[:CAP]

cards = []
for k, r in enumerate(sheet, 1):
    imgs = ""
    for suf, cap in [("t_2", "t-2"), ("t_1", "t-1"), ("t", "t")]:
        u = frame_uri(r["id"], suf)
        if u:
            imgs += f'<figure><img src="{u}"><figcaption>{cap}</figcaption></figure>'
    prop = r.get("proposed") or "— (judge 합의 없음, 사람이 판정)"
    badge = "사람필수" if r.get("needs_human") else "제안승인"
    votes = r.get("votes")
    vote_row = (f"<tr><td class=k>3-judge 표</td><td>frames1={html.escape(str(votes[0]))}, "
                f"frames2={html.escape(str(votes[1]))}, narr3={html.escape(str(votes[2]))}</td></tr>"
                if votes else
                (f"<tr><td class=k>narration judge</td><td>{html.escape(str(r.get('judge3_narration')))}</td></tr>"
                 if r.get("judge3_narration") else ""))
    opts = "".join(
        f'<label><input type=radio name=v{k}>{L}</label>'
        for L in ["MINE", "PERSON_k", "SHARED", "AMBIGUOUS"]
    )
    cards.append(f"""
    <div class=card>
      <h3>#{k} <span class=t>{html.escape(r.get('tier','?'))}/{html.escape(r.get('conf','?'))}</span>
          <span class=b>{badge}</span></h3>
      <div class=meta><code>{html.escape(r['id'])}</code><br>
        object: <b>{html.escape(str(r.get('object')))}</b> &nbsp;|&nbsp; caption: {html.escape(str(r.get('caption')))}</div>
      <div class=frames>{imgs}</div>
      <table>
        <tr><td class=k>파이프라인 GT</td><td>{html.escape(str(r.get('gt')))}</td></tr>
        <tr><td class=k>claude (frames)</td><td>{html.escape(str(r.get('claude')))}</td></tr>
        <tr><td class=k>gpt (frames)</td><td>{html.escape(str(r.get('gpt')))}</td></tr>
        {vote_row}
        <tr><td class=k>제안 라벨</td><td class=prop>{html.escape(str(prop))}</td></tr>
      </table>
      <div class=verdict>검수:
        <label><input type=radio name=v{k} checked>제안 승인</label> {opts}
      </div>
    </div>""")

n_human = sum(1 for r in queue if r.get("needs_human"))
n_appr = len(queue) - n_human
doc = f"""<!doctype html><meta charset=utf-8>
<title>EgoOwn 검수 시트 (eval, 3-judge)</title>
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
<h1>EgoOwn human 검수 시트 (eval, 3-judge 반영)</h1>
<p>전체 검수 큐 <b>{len(queue)}</b>개 — 제안승인 <b>{n_appr}</b> / 사람필수 <b>{n_human}</b>
(JSONL: review_queue.jsonl). 아래는 우선순위 상위 <b>{len(sheet)}</b>개
(사람필수 먼저). '제안 라벨'은 judge majority이며, 승인 또는 수정하세요.</p>
{''.join(cards)}
"""
(D / "review_sheet.html").write_text(doc, encoding="utf-8")
print(f"HTML 시트: 상위 {len(sheet)}개 → {D/'review_sheet.html'} "
      f"({(D/'review_sheet.html').stat().st_size//1024} KB)")
print(f"큐: 제안승인 {n_appr} / 사람필수 {n_human}")
