# Egocentric Implicit Ownership Labeling

Pipeline for building and reviewing implicit object-ownership annotations from
egocentric videos.

## Stages

```text
extract-bbox -> object-caption -> auto-label -> vlm-crosscheck -> serve
```

- `extract-bbox`: normalize dataset captions, filter tabletop object nouns, sample
  the caption reference frame, and extract one target bbox per object noun.
- `object-caption`: run SAM-2 tracking from the reference bbox and caption the
  highlighted object with Qwen3-VL.
- `auto-label`: choose sparse `t-2`, `t-1`, `t` benchmark frames and assign
  taxonomy / ownership labels from object descriptions and visual evidence.
- `vlm-crosscheck`: optionally ask independent VLM judges to audit labels.
- `serve`: launch the human review UI.

## Runtime Setup

Install the package in the environment used for the lightweight CLI stages:

```bash
cd /home/jhlee/ego-label-pipeline
pip install -e .
```

Recommended environment split:

- `extract-bbox`: run in the SAM/SAM-3 environment when using `--sam-backend sam3`.
- `object-caption`: run in the environment that can load SAM-2 and Qwen3-VL.
- `auto-label`, `vlm-crosscheck`, `serve`: can run in the normal project/test environment.

All stages write JSONL incrementally and support resume by default. Use
`--overwrite` when you intentionally want to regenerate an output from scratch.

## Label Space

`MINE`, `PERSON_k`, `SHARED`, `AMBIGUOUS`

## Taxonomy

| Taxonomy | Name | Description |
| --- | --- | --- |
| A | Baseline | Static cues are aligned; ownership is clear from spatial/object evidence. |
| B | Conflict | Spatial, semantic, or relational cues disagree. |
| C | Contextual | Past frames/action history are needed to override current appearance. |
| D | Ambiguous | Evidence is insufficient or symmetric. |

## Example Commands

EgoLife:

```bash
cd /home/jhlee/ego-label-pipeline

egoown extract-bbox \
  --dataset egolife \
  --input data/egolife/tabletop_object_annotations.jsonl \
  --videos-root /data/video_datasets/EgoLife \
  --out outputs/egolife/bbox_objects.jsonl \
  --sam-backend sam3 \
  --sam-model-id facebook/sam3 \
  --sam-device cuda:0

egoown object-caption \
  --dataset egolife \
  --input outputs/egolife/bbox_objects.jsonl \
  --out outputs/egolife/captions.jsonl \
  --caption-device cuda:0 \
  --mask-model-path facebook/sam2.1-hiera-base-plus

egoown auto-label \
  --dataset egolife \
  --input outputs/egolife/captions.jsonl \
  --out outputs/egolife/labels.jsonl \
  --sam2-track facebook/sam2.1-hiera-base-plus \
  --sam2-device cuda:0

egoown vlm-crosscheck \
  --dataset egolife \
  --input outputs/egolife/labels.jsonl \
  --out outputs/egolife/gpt_crosscheck.jsonl \
  --judge openai:gpt-4o \
  --frames-root outputs/egolife/auto_label_sparse_frames

egoown serve \
  --input outputs/egolife/labels.jsonl \
  --crosscheck outputs/egolife/gpt_crosscheck.jsonl \
  --frames-root outputs/egolife/auto_label_sparse_frames \
  --videos-root /data/video_datasets/EgoLife \
  --host 0.0.0.0 \
  --port 8000
```

Ego4D:

```bash
cd /home/jhlee/ego-label-pipeline

egoown extract-bbox \
  --dataset ego4d \
  --input data/ego4d/v2/annotations/narration.json \
  --videos-root /data/video_datasets/Ego4D/v2/full_scale \
  --out outputs/ego4d/bbox_objects.jsonl \
  --sam-backend sam3 \
  --sam-model-id facebook/sam3 \
  --sam-device cuda:0
```

Then run the same `object-caption`, `auto-label`, `vlm-crosscheck`, and `serve`
stages with `--dataset ego4d`.

## A2PD (Aria Gen 2 Pilot Dataset)

A2PD has no per-clip narration, so the target-object list was derived from the
video instead (a vision model, gpt-4.1-mini, over 15 s clips, keeping only clips
where someone besides the wearer speaks). That step is already done and checked
in, so a run starts at `extract-bbox`:

```text
resources/a2pd/objects.jsonl       176 clips, 10 multi-person sequences, nouns per clip
resources/a2pd/object_nouns.jsonl  reviewed vocabulary (78 kept / 13 rejected, with reasons)
```

1. Get access at https://explorer.projectaria.com/gen2pilot (CC BY-NC 4.0) and save the
   download manifest as `data/gen2pilot/gen2pilot_download_urls.json`. Its links expire.
2. Download the 10 preview videos (about 8 GB). The layout matters: `video_path` in
   `objects.jsonl` is `data/gen2pilot/a2pd/<seq>/<file>.mp4`.

```bash
python3 scripts/fetch_a2pd.py data/gen2pilot/gen2pilot_download_urls.json \
  --categories video_main_rgb --out data/gen2pilot/a2pd \
  --sequences eat_0 eat_1 eat_2 eat_3 play_0 play_1 play_2 play_3 walk_0 walk_1
```

3. Run the stages (from the repo root, so `.env` is picked up; `HF_HUB_OFFLINE=1` uses the
   cached SAM-3 weights, which are a gated repo):

```bash
egoown extract-bbox --dataset generic \
  --input resources/a2pd/objects.jsonl --videos-root data/gen2pilot/a2pd \
  --object-nouns resources/a2pd/object_nouns.jsonl \
  --out outputs/a2pd/bbox_objects.jsonl --frames-dir outputs/a2pd/reference_frames \
  --sam-backend sam3 --sam-model-id facebook/sam3 --sam-device cuda:0

egoown object-caption --dataset generic \
  --input outputs/a2pd/bbox_objects.jsonl --out outputs/a2pd/captions.jsonl \
  --caption-device cuda:0 --mask-model-path facebook/sam2.1-hiera-base-plus

egoown auto-label --dataset generic \
  --input outputs/a2pd/captions.jsonl --out outputs/a2pd/labels.jsonl \
  --sam2-track facebook/sam2.1-hiera-base-plus --sam2-device cuda:0
```

Every record carries `dataset: a2pd`, which switches on the place-setting rules: each
diner's plate, napkin and coaster is judged by where it sits (the wearer's own place, another
person's, or the middle of the table) instead of being treated as communal, and the wearer's
zone is not pulled down by a person seated at the table's side. Other datasets are unaffected.

## Outputs

Default per-dataset outputs:

```text
outputs/{dataset}/bbox_objects.jsonl
outputs/{dataset}/captions.jsonl
outputs/{dataset}/labels.jsonl
outputs/{dataset}/crosscheck.jsonl
outputs/{dataset}/auto_label_sparse_frames/
```

Useful review files:

- `bbox_objects.jsonl`: reference object bboxes from the caption frame.
- `captions.jsonl`: SAM-2 highlighted-object descriptions.
- `labels.jsonl`: taxonomy, GT label, selected frames, and evidence fields.
- `crosscheck.jsonl`: optional independent VLM judge outputs.

## Branch

This cleaned pipeline lives on:

```bash
object-ownership-labeling-pipeline
```
