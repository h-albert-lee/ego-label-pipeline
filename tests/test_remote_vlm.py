"""Tests for the remote VLM client (mocked — no actual API calls)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Skip if anthropic SDK isn't installed locally — tests aren't required for
# users who only run the local model path.
pytest.importorskip("anthropic", reason="anthropic SDK not installed")

from egoownership.detection import remote_vlm
from egoownership.schema import BBox, ClipCandidate, ObjectDetection, Taxonomy


def _png_bytes() -> bytes:
    """Return a minimal valid PNG (1×1 white pixel)."""
    import zlib, struct

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = b"IHDR" + struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr_chunk = struct.pack(">I", len(ihdr) - 4) + ihdr + struct.pack(">I", zlib.crc32(ihdr))
    raw = b"\x00\xff\xff\xff"
    comp = zlib.compress(raw)
    idat = b"IDAT" + comp
    idat_chunk = struct.pack(">I", len(idat) - 4) + idat + struct.pack(">I", zlib.crc32(idat))
    iend = b"IEND"
    iend_chunk = struct.pack(">I", 0) + iend + struct.pack(">I", zlib.crc32(iend))
    return sig + ihdr_chunk + idat_chunk + iend_chunk


@pytest.fixture
def fake_image(tmp_path: Path) -> Path:
    p = tmp_path / "demo.png"
    p.write_bytes(_png_bytes())
    return p


def _mock_response(json_text: str) -> MagicMock:
    """Build a fake Anthropic Messages response with one text block."""
    resp = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = json_text
    resp.content = [text_block]
    return resp


def test_caption_object_parses_structured_response(fake_image: Path):
    remote_vlm._client.cache_clear()
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _mock_response(
        '{"color": "white", "material": "ceramic", "state": "empty",'
        ' "text_on_object": null, "fine_grained_label": "ceramic mug",'
        ' "distinctive_marks": null}'
    )
    with patch.object(remote_vlm, "_client", return_value=fake_client):
        client = remote_vlm.RemoteVLM()
        det = ObjectDetection(
            label="cup",
            bbox=BBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0),
            score=0.9,
        )
        attrs = client.caption_object(fake_image, det)
    assert attrs.color == "white"
    assert attrs.material == "ceramic"
    assert attrs.fine_grained_label == "ceramic mug"
    # Verify the request shape: model is opus-4-7, system is cached, structured output configured.
    args = fake_client.messages.create.call_args.kwargs
    assert args["model"] == "claude-opus-4-7"
    assert args["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert args["output_config"]["format"]["type"] == "json_schema"


def test_tag_frame_returns_lowercase_list(fake_image: Path):
    remote_vlm._client.cache_clear()
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _mock_response(
        '{"tags": ["Cup", "Plate", "Hand"]}'
    )
    with patch.object(remote_vlm, "_client", return_value=fake_client):
        client = remote_vlm.RemoteVLM()
        tags = client.tag_frame(fake_image)
    assert tags == ["cup", "plate", "hand"]


def test_judge_scene_passes_three_frames_and_uses_adaptive_thinking(fake_image: Path):
    remote_vlm._client.cache_clear()
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _mock_response(
        '{"label": "PERSON_k", "confidence": 0.78,'
        ' "rationale": "Pen leaves wearer hand and lands in person_1 area by t.",'
        ' "target_instance_hint": "pen_1"}'
    )
    with patch.object(remote_vlm, "_client", return_value=fake_client):
        client = remote_vlm.RemoteVLM()
        clip = ClipCandidate(
            dataset="ego4d_fho",
            clip_id="demo:give_pen",
            video_id="vid",
            taxonomy=Taxonomy.CONTEXTUAL,
            t_minus_2_sec=0.0,
            t_minus_1_sec=0.5,
            t_sec=1.0,
            verb="give",
            nouns=["pen"],
        )
        result = client.judge_scene(clip, [fake_image, fake_image, fake_image])
    assert result["label"] == "PERSON_k"
    assert "person_1" in result["rationale"]
    args = fake_client.messages.create.call_args.kwargs
    # Must enable adaptive thinking for scene judgment.
    assert args["thinking"] == {"type": "adaptive"}
    # The user message must contain 3 frame markers + image blocks.
    user_blocks = args["messages"][0]["content"]
    image_blocks = [b for b in user_blocks if b.get("type") == "image"]
    assert len(image_blocks) == 3


def test_factory_returns_anthropic_client_by_default():
    client = remote_vlm.get_client()
    assert isinstance(client, remote_vlm.RemoteVLM)


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError):
        remote_vlm.get_client(provider="bogus")
