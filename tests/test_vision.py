"""Images: PDF pages with figures or scans, and image files, read by a (fake) vision model."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ask import config, usage
from ask.sources import loaders, vision
from ask.sources.loaders import LoadError, load_path


@pytest.fixture
def fake_vision(monkeypatch):
    """Vision on, with a model that reports the size of the image it was shown."""
    monkeypatch.setattr(config, "READ_IMAGES", True)
    seen = []

    def describe(png, model=None):
        seen.append(len(png))
        return f"IMAGE TEXT #{len(seen)}: Gini 0.684 chart"
    monkeypatch.setattr(vision, "describe_png", describe)
    return seen


def _pdf(path, pages):
    """pages: list of (text or None, with_figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(path) as pdf:
        for text, figure in pages:
            fig = plt.figure(figsize=(6, 8))
            if text:
                fig.text(0.1, 0.9, text)
            if figure:
                fig.figimage(np.random.default_rng(0).random((300, 300)), xo=50, yo=50)
            pdf.savefig(fig)
            plt.close(fig)
    return path


def test_pages_with_figures_or_no_text_are_read(tmp_path, fake_vision):
    p = _pdf(tmp_path / "report.pdf", [("Section 1. Discrimination results are shown in the chart below.", True),
                                         ("Section 2. Text only, nothing to look at on this page at all.", False),
                                         (None, True)])                      # a scanned page
    [src] = load_path(p)
    text = src.files["report.pdf"]
    assert text.count("read by the vision model") == 2 and len(fake_vision) == 2
    assert "[page 1 · images, read by the vision model]\nIMAGE TEXT" in text
    assert "[page 2 · images" not in text and "[page 3 · images" in text


def test_scanned_pdf_without_vision_says_how_to_enable_it(tmp_path):
    p = _pdf(tmp_path / "scan.pdf", [(None, True)])
    with pytest.raises(LoadError, match="ASK_READ_IMAGES"):
        load_path(p)


def test_vision_page_limit(tmp_path, fake_vision, monkeypatch):
    monkeypatch.setattr(config, "MAX_VISION_PAGES", 2)
    p = _pdf(tmp_path / "scans.pdf", [(None, True)] * 4)
    text = load_path(p)[0].files["scans.pdf"]
    assert len(fake_vision) == 2 and "2 more pages with images were not read" in text


def test_folder_with_images_and_documents(tmp_path, fake_vision):
    from PIL import Image
    d = tmp_path / "evidence"
    d.mkdir()
    (d / "notes.md").write_text("# Notes\nSee the screenshots.\n", encoding="utf-8")
    for name in ("roc.png", "calibration.jpg"):
        Image.fromarray((np.random.default_rng(1).random((200, 300, 3)) * 255).astype("uint8")).save(d / name)
    (d / "broken.png").write_bytes(b"not an image")
    [src] = load_path(d)
    assert set(src.files) == {"notes.md", "roc.png", "calibration.jpg"}
    assert src.files["roc.png"].startswith("[image: roc.png · read by the vision model]")
    assert any(s.startswith("broken.png") for s in src.skipped)


def test_images_only_folder_and_single_image(tmp_path, fake_vision):
    from PIL import Image
    d = tmp_path / "shots"
    d.mkdir()
    Image.new("RGB", (400, 300), "white").save(d / "a.png")
    [src] = load_path(d)
    assert src.kind == "docs" and list(src.files) == ["a.png"]
    [one] = load_path(d / "a.png")
    assert one.kind == "docs" and "IMAGE TEXT" in one.files["a.png"]


def test_folder_images_skipped_when_vision_is_off(tmp_path):
    from PIL import Image
    d = tmp_path / "mixed"
    d.mkdir()
    (d / "readme.md").write_text("# Hi\n", encoding="utf-8")
    Image.new("RGB", (200, 200)).save(d / "chart.png")
    [src] = load_path(d)
    assert list(src.files) == ["readme.md"] and any("vision model switched off" in s for s in src.skipped)


def test_describe_png_is_cached_and_metered(monkeypatch):
    calls = []

    def fake_create(**kw):
        calls.append(kw)
        return SimpleNamespace(output_text="Table: PD 1.2%", usage=SimpleNamespace(
            input_tokens=900, output_tokens=40, input_tokens_details=SimpleNamespace(cached_tokens=0)))
    from ask.agent import llm_call
    monkeypatch.setattr(llm_call, "_responses_create", fake_create)
    m = usage.Meter()
    with usage.metering(m):
        assert vision.describe_png(b"\x89PNG fake", "gpt-5.6-luna") == "Table: PD 1.2%"
        assert vision.describe_png(b"\x89PNG fake", "gpt-5.6-luna") == "Table: PD 1.2%"   # from disk
    assert len(calls) == 1 and m.by_model["gpt-5.6-luna"]["input"] == 900
    content = calls[0]["input"][0]["content"]
    assert content[1]["type"] == "input_image" and content[1]["image_url"].startswith("data:image/png;base64,")


def test_text_cache_separates_vision_on_and_off(tmp_path, monkeypatch):
    p = _pdf(tmp_path / "r.pdf", [("Some text on the page that is long enough to count.", True)])
    off = load_path(p)[0].files["r.pdf"]
    monkeypatch.setattr(config, "READ_IMAGES", True)
    monkeypatch.setattr(vision, "describe_png", lambda png, model=None: "FIGURE")
    loaders._TEXT_CACHE.clear()
    on = load_path(p)[0].files["r.pdf"]
    assert "FIGURE" not in off and "FIGURE" in on
