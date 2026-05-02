# Egocentric Implicit Ownership — 프로젝트 종합 문서

본 문서는 본 프로젝트의 **모든 맥락, 설계 의도, 사용법**을 한국어로 정리한
단일 진입점입니다. 새 팀원이 합류했을 때 이 문서 하나만 읽어도 프로젝트 전체
그림을 이해할 수 있도록 구성했습니다.

---

## 목차
1. [프로젝트 목적](#1-프로젝트-목적)
2. [핵심 개념: Taxonomy와 Label space](#2-핵심-개념-taxonomy와-label-space)
3. [대상 데이터셋과 선정 이유](#3-대상-데이터셋과-선정-이유)
4. [전체 파이프라인 흐름](#4-전체-파이프라인-흐름)
5. [각 단계 설계 의도](#5-각-단계-설계-의도)
6. [협업 어노테이션 서버 / UI](#6-협업-어노테이션-서버--ui)
7. [어노테이터 작업 가이드](#7-어노테이터-작업-가이드)
8. [폴더 구조](#8-폴더-구조)
9. [설치 및 실행](#9-설치-및-실행)
10. [확장 가이드](#10-확장-가이드)
11. [열린 질문 / 향후 과제](#11-열린-질문--향후-과제)

---

## 1. 프로젝트 목적

**1인칭 시점 영상에서 물리적 접촉 없이 시각·맥락 단서만으로 물체의 암묵적
소유권을 추론**하는 모델을 위한 벤치마크 데이터셋을 구축하는 것이 목표입니다.

전형적인 시나리오:
- 식당 테이블에서 내 앞의 접시·컵(MINE), 맞은편 사람의 노트(PERSON_k),
  테이블 중앙의 빵 바구니(SHARED).
- 회의실에서 노트북·메모장(MINE), 상대방의 펜(PERSON_k), 공용 물병(SHARED).
- 누군가 펜을 건네는 짧은 시퀀스: 처음에는 MINE, 건네는 중에는 SHARED,
  최종적으로 PERSON_k.

단일 프레임만 보면 위 판단은 쉽게 빗나갑니다. 그래서 **희소한 3프레임
(t-2, t-1, t)** 시퀀스를 사용해 시간적 맥락을 명시적으로 모델에 제공합니다.

---

## 2. 핵심 개념: Taxonomy와 Label space

### Taxonomy (씬 카테고리)
| 코드 | 이름 | 설명 |
|------|------|------|
| **A** | Baseline | 단일 프레임만 봐도 MINE/SHARED/PERSON_k 가 명확한 정적 장면 |
| **B** | Conflict | 시각적 단서와 맥락적 단서가 *불일치* (예: 손이 닿아 있지만 실제로는 상대 물건) |
| **C** | Contextual | 과거 프레임을 봐야 정답이 결정됨 (give / pass / put down 등) |
| **D** | Ambiguous | 대칭이거나 단서가 부족해 정답이 정해지지 않음 |

### Label space (객체별 정답 레이블)
| 레이블 | 의미 |
|--------|------|
| `MINE` | 카메라 착용자(나)의 소유 |
| `PERSON_k` | 다른 참가자의 소유 (k는 person_1, person_2 등) |
| `SHARED` | 공유 / 공용 |
| `AMBIGUOUS` | 정답 미정 |

### 왜 이 두 축인가?
- 모델 평가시 **Taxonomy별 성능을 분리해서** 봐야 합니다. Taxonomy A는 쉬워서
  많이 맞히고, B/C/D에서 갈리는 게 모델 능력 측정의 핵심입니다.
- Label space는 출력 헤드. Taxonomy는 데이터 슬라이스.

---

## 3. 대상 데이터셋과 선정 이유

### 일차 후보
| 데이터셋 | 강점 | 약점 | 본 프로젝트 활용 |
|---------|------|------|---------------|
| **Ego4D FHO** | pre/PNR/post 3프레임 + 손/객체 bbox 1.97M개 | 단발성 인터랙션 위주 | Taxonomy C (give/pass/put_down) 메인 소스 |
| **EPIC-KITCHENS-100** | 39.6k action segment, 454k bbox | 단일 참가자 (주방 혼자) | Taxonomy A (테이블 위 정적 장면) |
| **HD-EPIC** | 19.9k movement track, SAM2 마스크 (IoU 0.82) | 주방 한정 | Taxonomy C 보강 (transfer 트랙) |
| **EgoLife** | 6명 참가자 · dining/meeting · transcript+AV caption | 라이선스/접근성 | Taxonomy A/C/D **사회적 상황** 메인 소스 |

### 보조
- **Ego4D Episodic Memory** — 객체 연속 bbox (장기 트래킹 보강 시).
- **Ego-Exo4D** — 단일 참가자 위주여서 우선순위 낮음.

### 선정 원칙
1. 손/객체 bbox가 이미 라벨돼 있어야 한다 (없으면 자동 추출 비용 큼).
2. dining/meeting처럼 **공용 테이블** 상황이 많아야 한다.
3. 가능하면 **다중 참가자**가 있어야 PERSON_k 신호가 풍부하다.

---

## 4. 전체 파이프라인 흐름

```
[원본 어노테이션]
  └── filter (verb/noun 화이트리스트로 후보 선별)
        └── candidates_*.jsonl
              └── extract-frames (ffmpeg, t-2/t-1/t)
                    └── frames/
                          └── detect (visual evidence 풍부화)
                                ├── Grounding DINO (top-down: clip 명사 프롬프트)
                                ├── RAM (bottom-up: 모든 명사 자동 추출)
                                ├── SAM2 (마스크 정제)
                                ├── Person detector (사람 bbox)
                                ├── Instance tracking (instance_id 부여)
                                ├── Depth Anything v2 (선택)
                                ├── BLIP-2 attribute 추출 (선택)
                                └── Scene graph (next_to / held_by / moved_to)
                                      └── detections.jsonl
                                            └── label (rule cascade)
                                                  └── scene_records.jsonl
                                                        └── serve (협업 UI)
                                                              └── 사람이 검수·수정
                                                                    └── 최종 벤치마크
```

각 스테이지는 **JSONL 파일을 입출력**으로 분리되어 있어 어디서든 멈췄다
이어붙이기 좋습니다.

---

## 4.1 자동라벨링의 세 가지 모드 (A/B/C)

`detect` 단계에서 어떻게 visual evidence를 만들지에 따라 세 경로 중 하나를 선택합니다.

| 모드 | 모델 | 명령 옵션 | 용도 |
|---|---|---|---|
| **A. Native bbox** | 없음 (CPU만) | `egoown detect --source native --annotations <fho_main.json> --dataset ego4d-fho` | GPU 없이 빠르게 시작. FHO/HD-EPIC의 어노테이션 bbox만 사용 → person/RAM/depth/attribute 없음. EPIC-KITCHENS는 native bbox가 없어서 미지원. |
| **B. Local models** | torch + transformers | `egoown detect --source model ...` | 풀 스택 (DINO + RAM + SAM + Person + Depth + BLIP-2). HuggingFace에서 첫 실행 시 가중치 자동 다운로드. GPU 권장 (4090 기준 클립당 1-2초). |
| **C. Remote VLM** | Anthropic / OpenAI API | `egoown detect ... --remote-vlm anthropic` 또는 `egoown label ... --remote-vlm-judge anthropic` | RAM/BLIP-2 자리에 Claude Opus 4.7 또는 GPT-4o가 들어감. 또는 라벨링 단계에서 scene-level second opinion. `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` 필요. |

세 모드는 **혼합 가능**합니다. 가장 흔한 권장 흐름:
1. **A로 빠른 시작** — Ego4D FHO fixture로 어노테이터들이 UI에서 흐름을 익히게 함.
2. **B로 풍부화** — GPU 머신에서 person/depth/attribute을 채워 다시 검수.
3. **C 추가** — attribute 추출만 Claude Opus 4.7로 빼서 비용 효율성과 품질을 동시에. label 단계에서는 `--remote-vlm-judge anthropic`으로 VLM second opinion을 SceneRecord에 함께 저장 (룰 캐스케이드와 비교 가능).

### C 경로의 세 가지 작업
모두 `src/egoownership/detection/remote_vlm.py` (Anthropic) / `remote_vlm_openai.py` (OpenAI) 에 구현.

1. **`caption_object`** — cropped bbox → `ObjectAttributes` (color/material/state/marks). Anthropic의 `output_config.format` (json_schema) 으로 구조화 출력 강제. 시스템 프롬프트는 `cache_control: ephemeral`로 prompt caching → 100개 클립 돌려도 시스템 프롬프트 비용은 5분 윈도우당 1회만 발생.
2. **`tag_frame`** — 프레임 → 명사 리스트 (RAM 대체). 동일한 prompt caching.
3. **`judge_scene`** — 3프레임 + clip 메타 → `{label, confidence, rationale, target_instance_hint}`. **Adaptive thinking 활성화** (Opus 4.7의 추론 능력 사용). 결과는 `SceneRecord.vlm_judgement`에 저장되며 **룰 캐스케이드 라벨은 그대로 유지** — 두 신호를 어노테이터 UI에서 나란히 보여줘 충돌 시 사람이 판단.

기본 모델은 **`claude-opus-4-7`**. 환경변수 `EGOOWN_VLM_PROVIDER=openai`로 OpenAI 백엔드 (기본 `gpt-4o`) 전환 가능.

### 비용 가이드 (대략)
- A 경로: 무료 (CPU만)
- B 경로: GPU 시간 외엔 무료
- C 경로: Opus 4.7 prompt caching 적용 시 attribute 1건 약 $0.02-0.05, judge 1건 약 $0.10-0.20 (3프레임 + thinking). 100 클립이면 attribute $5~, judge $15~.

---

## 5. 각 단계 설계 의도

### 5.1 filter — 메타데이터 1차 필터링
- 어노테이션의 verb/noun을 `configs/taxonomy.yaml`의 화이트리스트와 매칭.
- Taxonomy C에는 `give`, `pass`, `hand_over`, `put_down`, `place`, `drop`,
  `take`, `transfer`, `pickup` 등을 포함.
- noun은 `cup`, `bowl`, `pen`, `notebook`, `plate`, `phone` 등 dining/meeting
  상황 명사 위주.
- HD-EPIC의 `transfer`/`pickup`이 본래 명세에 없어서 추가했습니다 — 같은
  비디오에서 일어나는 객체 이동 중 ownership이 의미 있는 것들이 거기 있음.

### 5.2 extract-frames — 희소 3프레임
- Ego4D FHO는 이미 pre/PNR/post 3프레임 인덱스 제공 → 그대로 사용.
- EPIC은 `start_timestamp` / `stop_timestamp`만 있음 → t-2 = start, t = stop,
  t-1 = midpoint으로 매핑.
- HD-EPIC movement track은 `start_frame` / `end_frame` → fps로 초로 변환.
- ffmpeg를 `-ss <시간> -i <비디오> -frames:v 1`로 호출. imageio backend도
  fallback으로 제공.

### 5.3 detect — 다단계 시각 증거 수집
설계 핵심: **여러 신호 소스를 결합해 한 프레임에서 최대한 많은 evidence를
뽑는다**. 어떤 신호 하나가 약해도 다른 신호가 보완하도록.

#### a) Grounding DINO (top-down)
- 어노테이션의 명사를 프롬프트로 넣어 **알려진 관심 객체만 빠르게** 검출.
- 프롬프트 형식: `"a cup. a plate. a hand."` (DINO 표준).

#### b) RAM (bottom-up, 선택)
- 알려지지 않은 객체까지 보강. 어노테이션이 빠뜨린 "맞은편 사람의 노트"
  같은 게 핵심 — 이게 PERSON_k 신호 만들어줌.
- `--use-ram` 플래그로 활성화.

#### c) SAM2 mask refinement (선택)
- DINO bbox를 SAM2에 넣어 마스크의 tight bbox로 교체. 손과 객체 경계가
  뚜렷할 때 효과 큼.
- `--use-sam` 플래그로 활성화.

#### d) Person detector
- DINO에 `"a person."` 프롬프트로 별도 호출 → 사람 bbox 분리.
- 프레임 간 IoU로 person_1, person_2 등 ID 전파.
- **이게 들어가면 zone 정의가 카메라 파라미터 의존성에서 벗어남** (5.4 참고).

#### e) Instance tracking
- 같은 객체가 t-2, t-1, t에서 모두 탐지될 때 `instance_id`를 일관되게 부여.
- 1차: 인접 프레임 IoU greedy 매칭 (threshold 0.30).
- 2차: 같은 class가 두 프레임에 정확히 1개씩 있으면 IoU=0이어도 매칭 — 펜
  건네기 같이 빠르게 움직이는 케이스 대응.
- 3차: 빈 프레임은 건너뛰고 "가장 최근 nonempty 프레임"과 매칭 — 일시적
  가림 대응.
- **Taxonomy D 케이스** (대칭 컵 2개)는 두 객체가 모두 같은 class라서 1차
  greedy로 정확히 분리. 2차 fallback은 이 경우 발동 안 됨 (multiplicity > 1).
- Optional: SAM2 video predictor 사용 가능 (`--use-sam2-video`).

#### f) Depth Anything v2 (선택)
- 단안 깊이 추정으로 객체 평균 depth 추출.
- `--estimate-depth` 활성화 시 zone heuristic이 "near zone (wearer)"를 깊이
  기반으로도 판단.

#### g) VLM attribute (선택, BLIP-2)
- cropped bbox → "Describe the cup including color, material, and state"
  프롬프트로 freeform caption 생성.
- 정규식으로 color/material/state/distinctive_marks를 채워 넣음.
- 어노테이터 UI에서 "두 컵 중 어느 게 누구 것인지" 구별할 때 사용 가능.
- `--extract-attrs` 플래그.

#### h) Scene graph (relations)
- `next_to` — 두 객체 중심 거리 < 0.10
- `held_by` — 객체가 손 bbox와 겹침, 또는 사람 bbox 하단 1/3에 있음
- `in_front_of_wearer` — 객체가 wearer near zone에 있음
- `on_shared_band` — shared 가로 띠 안에 있음
- `moved_to` — 같은 instance가 t-2 → t에서 0.15 이상 이동

### 5.4 zones — 동적 zone 정의 (Q2 응답)
원래는 YAML에 고정값을 박아 놨는데(`mine_near_y_min: 0.55` 등), 이는 카메라
높이/FOV에 강하게 의존합니다. 세 가지 전략을 제공합니다:

1. **`static-yaml`** — 사람 미검출 시 fallback.
2. **`person-relative`** — 검출된 사람 bbox 기준으로 zone 동적 계산.
   - 가장 낮은 사람 bbox 하단 + 0.05 → MINE 시작 y.
   - 사람들의 가로 중심값 사이 → SHARED 띠.
   - 각 사람의 영향력 박스(sax bbox + 좌우 패딩, 위로 화면 끝까지) → person_zone.
3. **`person-relative+depth`** — 위 + depth로 wearer 영역 추가 보정.

### 5.5 ownership.py — 룰 캐스케이드
객체 한 개에 대해 다음 순서로 결정:

```
1. held_by relation:
     - target이 person_X → PERSON_k
     - target이 wearer/hand → MINE
2. person_zones IoU > 0.05  → PERSON_k
3. depth가 wearer band 위쪽 → MINE
4. y >= mine_y_min          → MINE
5. nearest person 거리 0.20 이내 (사람 있을 때) → PERSON_k
6. shared 띠 안              → SHARED
7. y <= person_far_y_max (legacy fallback) → PERSON_k
8. 그 외 → AMBIGUOUS
```

각 룰에 evidence 문자열을 남겨서 어노테이터가 "왜 이 라벨이 나왔는지" 볼
수 있게 합니다 (`o.ownership_evidence`).

### 5.6 scene-level label 도출
- 클립 명사와 매칭되는 instance를 골라 그 instance의 t-2 → t-1 → t
  ownership 시퀀스를 봅니다.
- **stable** (3프레임 모두 같은 라벨) → 그 라벨, confidence 1.0
- **transition** (시작 ≠ 끝) → 끝 프레임의 라벨, confidence 0.65
- **트랙 누락** (어떤 프레임에서 instance 없음) → AMBIGUOUS, confidence 0.10
- **AMBIGUOUS 끼어 있으면** → AMBIGUOUS, confidence 0.20

추가로 같은 class instance가 여러 개 있으면 (Taxonomy D 후보) **자동
플래그**합니다 ("duplicate-cup present (Taxonomy D candidate)").

---

## 6. 협업 어노테이션 서버 / UI

### 왜 만들었는가
자동 라벨은 draft에 불과합니다. 진짜 벤치마크를 만들려면 사람이 매 클립을
검수해야 하고, 5명이 동시에 작업할 수 있어야 빠릅니다. 그래서 FastAPI
백엔드 + vanilla JS 프론트로 가벼운 협업 도구를 직접 만들었습니다.

### 백엔드 (FastAPI)
- `GET /api/scenes` — 씬 요약 리스트 (status, taxonomy, label, sort 필터)
- `GET /api/scenes/{clip_id}` — 풀 SceneRecord
- `POST /api/scenes/{clip_id}` — 부분 업데이트 (라벨, 상태, 노트, 객체별
  override)
- `GET /api/next-draft?after={clip_id}` — 다음 검수 필요 클립 (low confidence
  먼저)
- `GET /api/activity?limit=50` — 모든 어노테이터의 최근 편집 로그
- `GET /api/stats` — status/label/taxonomy별 카운트
- `GET /api/config` — 프론트에 비디오 가용 여부 전달
- `GET /frames/{path}` — 프레임 이미지 정적 서빙
- `GET /video/{video_id}` — 비디오 스트리밍 (HTTP Range 지원으로 t-2
  타임스탬프로 바로 seek 가능)

### 영속성
- `scene_records.jsonl` — 한 줄 = 한 SceneRecord. 매 편집마다 atomic rewrite.
- `scene_records.activity.jsonl` — append-only 편집 로그.
- 동시성은 `threading.Lock`. 다중 프로세스/머신 환경이 되면 SQLite로 옮길 것.

### 프론트엔드 (vanilla JS)
- 빌드 단계 없음. `index.html` + `app.css` + `app.js` 세 파일.
- 5초마다 활동 로그 폴링 — 다른 어노테이터의 작업이 거의 실시간으로 반영.

---

## 7. 어노테이터 작업 가이드

### 화면 구성
```
┌─ 헤더 ───────────────────────────────────────────────────────┐
│  Logo | 진행률 (verified/total · drafts left) | annotator name │
├─ 사이드바 ─┬─ 작업공간 ────────────────────┬─ 활동 / 통계 ──────┤
│ 검색       │ clip_id 메타 (badge들)         │ activity feed      │
│ 필터       │ video player (있으면)          │ ── stats 탭 ───── │
│ 정렬       │ 3프레임 캔버스 (bbox 오버레이) │ status bucket      │
│            │ ✓ Auto-label is correct (V)    │ label bucket       │
│ 씬 리스트  │ ✕ Reject (X)                   │ taxonomy bucket    │
│            │ Scene label / Tax / Status     │                    │
│            │ Notes                          │                    │
│            │ Save (⌘S) Prev(K) Next(J/N)    │                    │
│            │ Per-instance review            │                    │
│            │ Scene graph relations          │                    │
│            │ Edit history                   │                    │
└────────────┴─────────────────────────────────┴────────────────┘
```

### 권장 워크플로
1. **시작**: `annotator` 이름 칸에 본인 이름 입력 (localStorage 저장).
2. **N 키**: 가장 자신 없는 클립부터 (low confidence first 정렬).
3. **3프레임 검토**:
   - 영상 있으면 `↤ t-2` `⇶ t-1` `↦ t` 버튼으로 점프.
   - `Z`로 zone 오버레이 토글 → 왜 자동라벨이 그렇게 나왔는지 확인.
   - `R`로 relation 오버레이 토글 → next_to / held_by 라인 확인.
   - 프레임 클릭 → 확대 모달.
4. **자동 라벨이 맞으면** `V` (Quick Approve) → verified 상태로 저장 + 다음
   클립 자동 이동.
5. **자동 라벨이 틀리면**:
   - `1`/`2`/`3`/`4` 로 scene label 변경.
   - `A`/`B`/`C`/`D` 로 taxonomy 재지정 (예: 자동라벨이 D인데 실제로는 C).
   - per-instance 패널에서 객체별 override.
   - Notes에 이유 메모.
   - `⌘S` 또는 `⌘Enter` → 저장 + 자동 이동.
6. **검수 불가** (영상 화질 안 좋음, 단서 부족 등) → `X` (Reject) → 해당
   클립은 벤치마크에서 제외.
7. **모르겠음**: 상태를 `in_review`로만 바꾸고 다음으로. 다른 어노테이터가
   다시 볼 것.

### 키보드 단축키 (`?` 누르면 도움말)
| 키 | 동작 |
|----|------|
| `1` `2` `3` `4` | scene label · MINE / PERSON_k / SHARED / AMBIGUOUS |
| `A` `B` `C` `D` | taxonomy |
| `V` | quick approve (auto-label confirm + verified + next) |
| `X` | reject (확인 후 + next) |
| `⌘/Ctrl + S` | 저장 + 다음 |
| `J` / `N` | 다음 검수 필요 클립 |
| `K` | 이전 클립 |
| `Z` | zone 오버레이 토글 |
| `R` | relation 오버레이 토글 |
| `F` | 사이드바 검색창 포커스 |
| `?` | 도움말 모달 |
| `Esc` | 모달 닫기 / 입력 포커스 해제 |

### 두 어노테이터가 같은 클립을 만지면?
지금은 마지막 저장이 이김(last write wins). 활동 로그에 누가 무엇을
바꿨는지 다 남아 있어서 충돌 추적은 가능. 검수 정책으로 "한 클립 = 한
어노테이터" 라운드를 돌리고, 두 번째 라운드에서 다른 사람이 검토하는 식이
권장.

---

## 8. 폴더 구조

```
egoownership/
├── pyproject.toml             # 패키지 메타 + extras (frames/detect/serve/dev)
├── README.md                  # 영문 간단 안내 (이 파일은 docs/ 안에 한국어)
├── .gitignore                 # .venv, outputs/, frames*/ 제외
├── configs/
│   └── taxonomy.yaml          # verb/noun 화이트리스트 + zone 임계값
├── src/egoownership/
│   ├── __init__.py
│   ├── schema.py              # SceneRecord, BBox, Person, Relation, ...
│   ├── config.py              # YAML 로더
│   ├── filters.py             # verb+noun → Taxonomy 필터
│   ├── frames.py              # ffmpeg / imageio 프레임 추출
│   ├── pipeline.py            # 스테이지 오케스트레이션
│   ├── cli.py                 # typer CLI (egoown)
│   ├── datasets/              # FHO / EPIC / HD-EPIC / EgoLife 어댑터
│   ├── download/              # 데이터셋 다운로드 헬퍼
│   ├── detection/
│   │   ├── grounding_dino.py
│   │   ├── sam.py
│   │   ├── ram.py
│   │   ├── persons.py
│   │   ├── tracking.py
│   │   ├── zones.py
│   │   ├── depth.py
│   │   ├── attributes.py
│   │   ├── relations.py
│   │   └── ownership.py
│   └── server/
│       ├── app.py             # FastAPI
│       ├── store.py           # JSONL 저장소 + 파일락
│       ├── entry.py           # uvicorn reload용 진입점
│       └── static/            # HTML / CSS / JS
├── tests/                     # 40+ pytest (parsers, tracking, zones, server …)
└── docs/
    └── PROJECT_CONTEXT.md     # 이 문서
```

---

## 9. 설치 및 실행

### 9.1 패키지 설치
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[frames,detect,serve,dev]"
```

extras:
- `frames` — ffmpeg/imageio (프레임 추출)
- `detect` — torch + transformers (DINO/SAM/RAM/Depth/BLIP-2)
- `serve` — fastapi + uvicorn (어노테이션 서버)
- `dev` — pytest + ruff + httpx

### 9.2 데이터셋 어노테이션 받기
```bash
egoown download epic --out data/epic            # GitHub에서 직접 다운로드
egoown download ego4d --out data/ego4d          # 라이선스 동의 후 명령어 안내
egoown download hd-epic --out data/hd_epic      # 안내 출력
# EgoLife는 프로젝트 사이트 공식 절차 따라 받기
```

### 9.3 파이프라인 실행

**A. GPU 없이 빠른 시작 (모델 0개)**
```bash
egoown filter ego4d-fho --annotations data/ego4d/v2/annotations/fho_main.json \
    --taxonomy C --out outputs/cands.jsonl

egoown detect --candidates outputs/cands.jsonl \
    --source native --dataset ego4d-fho \
    --annotations data/ego4d/v2/annotations/fho_main.json \
    --out outputs/detections.jsonl

egoown label --detections outputs/detections.jsonl --out outputs/scenes.jsonl
egoown serve --scenes outputs/scenes.jsonl --frames-root frames/
```

**B. 로컬 풀스택 (GPU 권장)**
```bash
pip install -e ".[detect]"

egoown extract-frames --candidates outputs/cands.jsonl \
    --videos-root data/ego4d/videos --out frames/

egoown detect --candidates outputs/cands.jsonl --frames frames/ \
    --out outputs/detections.jsonl \
    --use-ram --extract-attrs --estimate-depth

egoown label --detections outputs/detections.jsonl --out outputs/scenes.jsonl
```

**C. Anthropic Claude Opus 4.7 보조**
```bash
pip install -e ".[remote-anthropic]"
export ANTHROPIC_API_KEY=sk-ant-...

# attribute / RAM tagging을 Claude로 위임
egoown detect --candidates outputs/cands.jsonl --frames frames/ \
    --out outputs/detections.jsonl \
    --use-ram --extract-attrs --remote-vlm anthropic

# label 단계에서 scene-level second opinion 추가
egoown label --detections outputs/detections.jsonl \
    --out outputs/scenes.jsonl \
    --remote-vlm-judge anthropic --frames-root frames/

# OpenAI GPT-4o 사용:
# pip install -e ".[remote-openai]"; export OPENAI_API_KEY=...
# egoown label ... --remote-vlm-judge openai --frames-root frames/
```

**서버 실행 (모든 경로 공통)**
```bash
egoown serve --scenes outputs/scenes.jsonl --frames-root frames/ \
    --videos-root data/ego4d/videos \
    --host 0.0.0.0 --port 8000
```

브라우저로 `http://<서버>:8000` 접속.

### 9.4 테스트
```bash
.venv/bin/pytest                    # 40+ tests (under 1s)
```

---

## 10. 확장 가이드

### 10.1 새 데이터셋 추가하기
1. `src/egoownership/datasets/<name>.py` 작성. `iter_<name>_candidates(path)`
   제너레이터를 노출하고 각 이벤트를 `ClipCandidate`로 yield.
2. `datasets/__init__.py`와 `pipeline._LOADER_MAP`에 등록.
3. `tests/test_<name>.py`에 mini fixture + 파서 테스트 추가.

### 10.2 새 검출 모듈 추가하기
- `detection/<name>.py`에 lazy-load 패턴 (`_load_<name>` + `lru_cache`)으로
  작성. 모델 weights가 없으면 graceful degrade.
- `pipeline.stage_detect`의 옵션 플래그에 추가.
- CLI `detect_cmd`에 `typer.Option`으로 노출.

### 10.3 새 zone 전략 추가
- `detection/zones.py`에 새 함수 추가 (입력: persons/depth/clip 맥락, 출력:
  `FrameZones`).
- `derivation` 필드에 새 식별자 ("hybrid-<X>") 부여.
- pipeline.stage_detect 분기에 옵션 추가.

### 10.4 ownership 룰 캐스케이드 수정
- `detection/ownership._classify_with_zones` 한 함수 안에 모두 모여 있음.
- 각 룰에 evidence 문자열 추가하면 UI에서 표시됨.

### 10.5 어노테이션 UI 컴포넌트 추가
- 새 패널이면 `index.html`에 `<details>` 블록, `app.css`에 스타일, `app.js`에
  `render*` 함수 추가.
- 새 키 단축키는 `app.js`의 `keydown` 핸들러 switch에 추가.

---

## 11. 열린 질문 / 향후 과제

### 단기
- **EgoLife 실데이터 검증** — 어노테이션 포맷이 실제로 어떤지(공식 문서)
  확인하고 어댑터 미세 조정.
- **충돌 해결 정책** — 두 어노테이터가 같은 클립을 다르게 라벨한 경우 UI
  수준에서 시각화 (현재는 활동 로그로만 추적 가능).
- **Reject된 클립의 활용** — 아예 폐기할지, "hard negative"로 별도 저장할지.
- **Conflict (Taxonomy B) 자동 검출** — 자동 라벨과 시각적 단서(held_by)가
  불일치하는 경우 Taxonomy B로 자동 플래그.

### 중기
- **3D 정보** — HD-EPIC 디지털 트윈 / monocular 3D detection 통합으로 zone
  정의를 진짜 3D 거리 기반으로 전환.
- **VLM evaluation** — 본 벤치마크가 완성되면 GPT-4o, Qwen2.5-VL, InternVL3
  같은 모델을 평가하는 evaluator 모듈을 별도 리포로 분리.
- **Active learning** — 어노테이터의 수정 패턴을 학습해서 다음 라벨 제안의
  자동 confidence를 보정.

### 장기
- **Cross-cultural ownership 시그널** — 한국/일본의 식사 예절 (어른 앞에
  먼저 두기, 공용 잔 회전 등)이 PERSON_k / SHARED 분포에 미치는 영향.
- **Privacy-preserving 평가** — egocentric 데이터의 얼굴/PII 처리 정책.

---

## 부록 A. 데이터 형식 요약

### ClipCandidate (filter 단계 산출물)
```json
{
  "dataset": "ego4d_fho",
  "clip_id": "clip_dining_001:ann_1",
  "video_id": "video_A",
  "taxonomy": "C",
  "t_minus_2_sec": 10.0,
  "t_minus_1_sec": 10.4,
  "t_sec": 10.8,
  "verb": "put_down",
  "nouns": ["cup"],
  "narration": "Put down the cup on the table",
  "source": {...}
}
```

### detections.jsonl (detect 단계 산출물)
```json
{
  "clip": { ... ClipCandidate ... },
  "frames": [
    {
      "tag": "t-2",
      "timestamp_sec": 10.0,
      "width": 1920, "height": 1080,
      "frame_path": "ego4d_fho/video_A/clip_dining_001_ann_1__t-2.jpg",
      "objects": [
        {
          "label": "cup",
          "bbox": {"x_min": 0.5, "y_min": 0.78, "x_max": 0.64, "y_max": 0.94},
          "score": 0.93,
          "instance_id": "cup_1",
          "ownership": null,
          "ownership_evidence": [],
          "attributes": {"color": "white", "material": "ceramic", ...},
          "mean_depth": 0.82
        }
      ],
      "persons": [
        {"bbox": {...}, "person_id": "person_1", "score": 0.91}
      ],
      "relations": [
        {"subject_id": "cup_1", "object_id": "wearer", "predicate": "in_front_of"}
      ],
      "zones": {"mine_y_min": 0.55, "shared_x_min": 0.30, ..., "derivation": "person-relative"}
    },
    ...
  ],
  "depth_bands": [...]
}
```

### SceneRecord (label 단계 산출물 = 최종 벤치마크 row)
```json
{
  "clip": { ... },
  "frames": [ ... 위와 동일 + ownership 채워짐 ... ],
  "scene_label": "PERSON_k",
  "scene_taxonomy": null,
  "notes": "cup_1: transition MINE → PERSON_k",
  "auto_label_confidence": 0.65,
  "review_status": "draft",
  "edits": [
    {
      "annotator": "alice",
      "when": "2026-05-02T00:42:14Z",
      "field": "scene_label",
      "old_value": "SHARED",
      "new_value": "PERSON_k",
      "note": null
    }
  ]
}
```

---

## 부록 B. Git / 배포

- 리포지토리: <https://github.com/h-albert-lee/ego-label-pipeline>
- 메인 브랜치: `main`
- 데이터·영상·체크포인트는 절대 커밋 금지 (`.gitignore`로 차단됨).
- 배포 방식 (현재): `egoown serve` 직접 실행. 프로덕션이라면 `uvicorn` 뒤에
  nginx + 인증 레이어 추가.
