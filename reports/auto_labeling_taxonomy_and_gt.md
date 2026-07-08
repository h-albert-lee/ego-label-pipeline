# Auto-Labeling Method: Ground Truth and Taxonomy

This document describes how `one-pass-labels` automatically assigns an ownership ground-truth
label (`auto_ground_truth`) and a taxonomy tag (`auto_taxonomy`) to each candidate object.

Implementation: `src/egoownership/catv_evidence_label.py` (`build_evidence_label`,
`_decide_taxonomy_gt`). Ground-truth values: `MINE`, `PERSON_k`, `SHARED`, `AMBIGUOUS`.
Taxonomy values: `A` (baseline), `B` (conflict), `C` (contextual), `D` (ambiguous).

## 1. Evidence collection

Five categories of evidence are assembled before any labeling decision is made.

### Object type

Each target is classified by a keyword lookup on its noun (`_object_type`) into one of three
categories — there is no longer an intermediate "weak-shared" tier (see §5, Rule 3):

- **shared** — functionally communal items whose ownership never depends on who's currently
  touching them: serving-ware and condiments (pot, tray, plate, bowl, dish, napkin, tissue,
  chopstick/spoon/fork/knife, cutlery, utensil, teapot, kettle, pan, frypan, saucepan, cooker,
  hotpot, chopboard, tablecloth, coaster, spatula, opener, salt, pepper, sugar, oil, water, food,
  trash, bin).
- **personal** — individually-owned belongings, including electronics/effects (phone, wallet,
  keys, laptop, camera, etc.) *and* drinkware (cup, mug, tumbler, bottle, jar, straw) — drinkware
  has no shared-by-default assumption; see §5.
- **generic_object** — anything not covered above.

### Spatial zone

The target's bounding box is classified into one of four zones by `_target_zone`:

- **ego_zone** — the region below all detected persons (the wearer's near zone).
- **other_person_zone** — inside a specific detected person's "influence rectangle."
- **shared_zone** / **center_table** — the band between detected people (or fixed dataset
  thresholds when no persons are detected).
- **background_or_ambiguous_zone** — none of the above.

Zone boundaries come from `person_relative_zones()` when people are visible, falling back to
fixed YAML thresholds (`static_zones()`) otherwise.

### Person detection and the ego-hand box

People are localized with an open-vocabulary detector (`detect_persons`, Grounding DINO,
`"a person."` prompt). Because the camera wearer's own hand/arm frequently produces a false
"person" hit in egocentric footage, a second pass checks for a face inside each candidate box:

- A box **with** a face is kept as a real person.
- Among **faceless** boxes, only the single one that most looks like the wearer's own hand
  (touches a frame border, lowest in the frame) is excluded from the persons list — other
  faceless boxes are kept, since a real bystander's face may simply be turned away or occluded.
- The excluded box is not discarded — it's returned separately as the **ego-hand bbox**, so
  downstream logic can tell "this object is in the wearer's own hand" apart from "this object is
  in person_k's hand," which the persons list alone cannot represent.

### Relations

A lightweight scene graph (`build_scene_graph`) records whether the target overlaps a detected
person's hand region (the padded lower 40% of their bbox), producing a `held_by=person_k`
relation. Separately, `_held_by_wearer_relation` checks the target against the ego-hand bbox
using **containment** (`intersection / target_area`), not IoU — the ego-hand box is a loose
region covering the wearer's whole reach, often much larger than the held object, so IoU
under-counts a small object fully inside it. Containment > 0.5 produces a `held_by=wearer`
relation. When both a wearer relation and a bystander relation are present, the wearer relation
always wins (it's the more specific, purpose-built signal).

### Caption cues

The auto-generated object caption is scanned (`_caption_cues`) for boolean signals: whether the
wearer is described as the actor (**ego_actor**), whether another person is (**other_actor**),
whether the caption explicitly states the actor is unclear (**ambiguous**), whether it states
communal/shared use (**shared_use**), whether it contains a physical-contradiction phrase like a
misoriented ID badge or screen (**conflict_cue**), and the specific history verb present, if any
(see Tier 2 below).

### Temporal evidence

When `--sam2-track` is enabled, the target's box is propagated backward from the current frame
(`t`) to two earlier sample points (`t-2`, `t-1`) via SAM-2, giving a zone/holder snapshot at
each. From this, `_derive_temporal_ownership_signals` computes whether the zone or holder
changed across the window (`ownership_trajectory`), and whether an ownership clue existed
earlier but is absent now (`contextual_requires_history`). Without `--sam2-track`, only the
current frame has a box, and all temporal signals are inert defaults (`None`/`False`).

## 2. Ground-truth assignment

GT is decided by a fixed sequence of tiers, each checked only if nothing higher has resolved it
yet (`gt is None`). This ordering encodes one principle: **the more specific and current a piece
of evidence is, the earlier it's allowed to decide the label.**

| Tier | Condition | Resolution |
|---|---|---|
| **0. Functional invariant** | Caption says shared/communal use, or `object_type == "shared"` | **SHARED** — unconditional, ahead of everything below, including an explicitly-ambiguous caption |
| *(hard stop)* | Caption says actor identity is explicitly ambiguous, and neither actor cue fired | **AMBIGUOUS** |
| **2. Verb-classified zone resolution** | Caption has a *transfer* verb (give/pass/hand/receive/serve/offer/return) | Ownership follows **where the object currently sits**: `ego_zone` → MINE, `other_person_zone` → PERSON_k. Zone inconclusive → falls through. |
| | Caption has a *temporary-use* verb (borrow/lend/loan) | Ownership is the **inverse** of where it currently sits: `ego_zone` → PERSON_k (holding ≠ owning — on loan), `other_person_zone` → MINE (lent out). Zone inconclusive → falls through. |
| **1. Relation graph** | Target overlaps the wearer's own ego-hand box | **MINE** |
| | Target overlaps a specific other person's hand region | **PERSON_k** |
| **2b. Plain actor cue** | Caption names exactly one of {wearer, other person} as sole actor (no history verb involved) | MINE / PERSON_k accordingly |
| **3. Zone fallback** | Zone is another person's zone | **PERSON_k** |
| | Zone is the wearer's zone | **MINE** |
| | Personal object in the shared zone, but temporal trajectory shows it was in the wearer's zone earlier (`ego_to_shared`) | **MINE** — relocating it doesn't relinquish ownership (abandonment case) |
| | Object sits in the shared zone otherwise | **AMBIGUOUS** |
| | No zone signal at all (background) | **AMBIGUOUS** |
| **Past-frame fallback** | Result of the above is still AMBIGUOUS, and an earlier frame (`t-1`, then `t-2`) had a clear zone/holder clue | Use **that** frame's implied owner instead of giving up |

Notes on specific tiers:

- Tier 0 is truly unconditional for `object_type == "shared"` — it fires before the
  ambiguous-caption check, so a shared object's label never depends on caption clarity or who's
  touching it.
- Tier 3's shared-zone branches never see `object_type == "shared"` — Tier 0 already consumed
  that case — so they only distinguish `personal` (eligible for the abandonment check) from
  `generic_object`.
- The past-frame fallback only ever changes a result that would otherwise be AMBIGUOUS; it never
  overrides a conclusion already reached by an earlier tier, and it does nothing when temporal
  tracking wasn't enabled (no earlier-frame data exists to fall back to).

## 3. Taxonomy assignment

Taxonomy is not an independent judgment — it's read off flags collected while computing GT:

- **D (Ambiguous)** — whenever the final GT is AMBIGUOUS (including after the past-frame
  fallback fails to find anything), regardless of any other flag.
- **C (Contextual)** — otherwise, if a history verb fired (Tier 2 or its inverse), the temporal
  abandonment case fired, the past-frame fallback fired, or temporal evidence shows an ownership
  clue existed earlier but not now.
- **B (Conflict)** — otherwise, if any of: a physical-contradiction caption phrase, a shared-type
  object sitting in the wearer's zone, a personal-type object sitting outside the wearer's zone,
  or the caption naming *both* the wearer and another person as actors simultaneously (a genuine
  contradiction, not merely missing signal).
- **A (Baseline)** — none of the above; a clean, uncontested read.

## 4. Design principle

Each evidence source has a designated scope of authority, and decisions never average or score
sources against each other — the pipeline always takes the highest-authority source that
actually has an opinion:

1. An explicit functional/communal-use signal (Tier 0) — because some objects' ownership status
   is simply not a function of who's touching them.
2. An explicit caption verb describing a transfer or loan, resolved against current zone (Tier 2)
   — because a stated history is more informative than raw geometry alone, but "who ends up with
   it" is best read from where it currently sits, not from noisy actor-cue text.
3. Direct physical-contact evidence (Tier 1: the relation graph, wearer-hand-first) — because
   current physical possession is stronger, more current evidence than a bare caption mention of
   who "the actor" is.
4. A plain, single-actor caption statement (Tier 2b).
5. Spatial position alone (Tier 3), with one exception for very recent history (temporal
   abandonment).
6. The immediate past, only as a last resort before giving up (past-frame fallback).

Taxonomy is a byproduct of *which* tier fired, not a separate judgment — so a downstream reviewer
can tell at a glance whether a label came from a clean caption sentence, spatial inference, or a
flagged disagreement between sources.

## 5. Rule compliance

The design was checked against seven ownership-labeling rules. Current status:

| # | Rule (summary) | Status |
|---|---|---|
| 1 | Table settings placed at person k's seat are OTHERS even before contact (setting = first occupancy) | ❌ **Not implemented.** Requires a persistent per-seat assignment that survives frames where that person isn't currently detected. Zones are recomputed fresh each frame from currently-visible people; there's no seat memory across a clip. |
| 2 | Shared-function objects (serving spoon, communal bottle) stay SHARED regardless of who holds them | ✅ Tier 0 is unconditional on `object_type == "shared"`, checked before any actor/relation evidence. |
| 3 | Shared-function → SHARED; personal-function (cup, phone) with insufficient clue → AMBIGUOUS, not SHARED | ✅ **Fixed.** Originally violated by a "weak-shared" drinkware category that defaulted cup/mug/etc. to SHARED absent an actor cue — removed; these items are now `personal` and default to AMBIGUOUS like any other personal object, matching the rule exactly. An individual cup can still resolve to SHARED via an explicit `shared_use` caption cue. |
| 4 | MINE = wearer ownership; holding ≠ owning; wearer holding someone else's object → OTHERS | ✅ (with a caveat) Tier 2's temporary-use verbs correctly invert MINE/OTHERS regardless of current holder. Caveat: if the wearer holds someone else's object with **no verb cue at all**, there's no signal to detect that, so it defaults to MINE via Tier 1 — this is a fundamental information limit, not a logic gap. |
| 5 | Record a specific k when identifiable; otherwise OTHERS only (evaluation collapses to OTHERS) | ❌ **Not implemented.** `PERSON_k` is still a literal placeholder string, never filled with an actual detected id or name — every case collapses to the same value. Requires person re-identification/tracking tied to a naming source, which doesn't exist yet. |
| 6 | Pushing your own object into the shared zone doesn't relinquish ownership → MINE | ✅ (conditional) Tier 3's `ownership_trajectory == "ego_to_shared"` check. Only fires when `--sam2-track` produced real `t-2`/`t-1` boxes; without it, this case still falls to AMBIGUOUS as before. |
| 7 | Third-party-hand cases decompose as: shared+contact → SHARED; personal+contact → OTHERS; personal+no-contact+owner-off-screen → AMBIGUOUS | ✅ Shared+contact is Tier 0 (trivial). Personal+contact is Tier 1's `held_by=person_k`, now reliable since the ego-hand fix stopped it misattributing bystander contact to the wearer. Personal+no-contact+off-screen is Tier 3's final AMBIGUOUS fallback. |

### Known open gaps (rules 1 and 5)

Both remaining gaps need a genuinely new capability rather than a logic change:

- **Rule 1** needs persistent, cross-frame seat/ownership-zone tracking per person — a
  fundamentally different mechanism from the current per-frame zone recomputation.
- **Rule 5** needs person re-identification across a clip tied to a naming source (e.g.
  transcript speaker attribution), to turn a generic `person_k` into a specific, trackable
  identity.

Neither is scoped or implemented as of this report.
