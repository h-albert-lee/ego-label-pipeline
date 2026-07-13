"""disagree 케이스 20개를 3프레임 + GT/judge 라벨 + evidence로 HTML 덤프."""
import base64
import html
import json
from pathlib import Path

CC = Path("outputs/ego4d/crosscheck.jsonl")
LAB = Path("outputs/ego4d/labels_v2.jsonl")
FR = Path("outputs/ego4d/crosscheck_reconstructed_frames")
OUT = Path("outputs/ego4d/disagree_report.html")
N = 20

lab = {json.loads(l)["id"]: json.loads(l) for l in open(LAB) if l.strip()}
cc = [json.loads(l) for l in open(CC) if l.strip()]
jname = list(cc[0]["judges"].keys())[0]

dis = [r for r in cc if not r["majority_agrees"]]
# MINE↔타인 혼동을 우선 노출 (cascade proximity 혐의 케이스 먼저)
dis.sort(key=lambda r: 0 if r["auto_ground_truth"] == "MINE" else 1)
dis = dis[:N]


def frame_uri(rid: str, suffix: str) -> str | None:
    prefix = rid.replace("#", "__")
    fp = FR / f"{prefix}__{suffix}.jpg"
    if fp.exists():
        b = base64.standard_b64encode(fp.read_bytes()).decode()
        return f"data:image/jpeg;base64,{b}"
    return None


cards = []
for i, r in enumerate(dis, 1):
    rid = r["id"]
    src = lab.get(rid, {})
    ake = src.get("auto_key_evidence") or {}
    j = r["judges"].get(jname, {})
    imgs = ""
    for suf, cap in [("t_2", "t-2"), ("t_1", "t-1"), ("t", "t (action)")]:
        uri = frame_uri(rid, suf)
        if uri:
            imgs += f'<figure><img src="{uri}"><figcaption>{cap}</figcaption></figure>'
    ev = "".join(
        f"<tr><td class=k>{html.escape(k)}</td><td>{html.escape(str(j.get(k,'')))}</td></tr>"
        for k in ["object_type_evidence", "zone_evidence",
                  "relation_graph_evidence", "context_change_evidence"]
    )
    cards.append(f"""
    <div class=card>
      <h3>#{i} &nbsp; <span class=gt>GT: {html.escape(str(r['auto_ground_truth']))}</span>
          &nbsp;→&nbsp; <span class=jl>judge: {html.escape(str(j.get('label')))}</span></h3>
      <div class=meta>id: <code>{html.escape(rid)}</code><br>
        caption: {html.escape(str(src.get('dense_caption_en','')))}<br>
        object: {html.escape(str((src.get('object') or {}).get('label','')))} &nbsp;|&nbsp;
        cascade: object_type=<b>{html.escape(str(ake.get('object_type')))}</b>,
        target_zone=<b>{html.escape(str(ake.get('target_zone')))}</b>,
        taxonomy=<b>{html.escape(str(src.get('auto_taxonomy')))}</b></div>
      <div class=frames>{imgs}</div>
      <details open><summary>GT 근거 (auto_rationale)</summary>
        <p class=rat>{html.escape(str(src.get('auto_rationale','')))}</p></details>
      <details open><summary>judge 근거 4필드</summary>
        <table>{ev}</table></details>
      <div class=verdict>사람 판정: &nbsp;
        <label><input type=radio name=v{i}> GT 맞음</label>
        <label><input type=radio name=v{i}> judge 맞음</label>
        <label><input type=radio name=v{i}> 둘 다 애매</label></div>
    </div>""")

doc = f"""<!doctype html><meta charset=utf-8>
<title>Ego ownership crosscheck — disagree {len(dis)}건</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}}
 h1{{font-size:20px}} .card{{border:1px solid #ddd;border-radius:10px;padding:1rem 1.2rem;margin:1.4rem 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
 h3{{margin:.2rem 0 .6rem}} .gt{{color:#0a7d2c}} .jl{{color:#b3261e}}
 .meta{{font-size:12.5px;color:#555;margin-bottom:.7rem}} code{{font-size:11px}}
 .frames{{display:flex;gap:.5rem}} figure{{margin:0;flex:1}} img{{width:100%;border-radius:6px;border:1px solid #ccc}}
 figcaption{{text-align:center;font-size:11px;color:#777}}
 table{{border-collapse:collapse;width:100%;font-size:12.5px}} td{{border:1px solid #eee;padding:.3rem .5rem;vertical-align:top}}
 td.k{{width:190px;color:#666;font-weight:600}} .rat{{background:#f6f8fa;padding:.5rem .7rem;border-radius:6px;font-size:12.5px}}
 summary{{cursor:pointer;font-weight:600;margin:.5rem 0 .3rem}} .verdict{{margin-top:.7rem;padding-top:.6rem;border-top:1px dashed #ddd;font-size:13px}}
 .verdict label{{margin-right:1rem}}
</style>
<h1>Ego ownership crosscheck — 불일치 {len(dis)}건 (judge={html.escape(jname)})</h1>
<p>MINE(GT)이 judge에서 뒤집힌 케이스를 앞쪽에 배치. 각 카드에서 3프레임을 보고 GT와 judge 중 누가 맞는지 판정.</p>
{''.join(cards)}
"""
OUT.write_text(doc, encoding="utf-8")
print(f"wrote {OUT} ({len(dis)} cases, {OUT.stat().st_size//1024} KB)")
