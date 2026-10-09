"""Tests for DatasetManager."""

from pathlib import Path
import pytest

from quantforge.engines.dataset_manager import DatasetManager


def test_list_and_get_presets():
    mgr = DatasetManager()
    presets = mgr.list_presets()
    assert len(presets) >= 4
    names = [p.name for p in presets]
    assert "general-wiki" in names
    assert "code-multilang" in names
    assert "reasoning-math" in names
    assert "multilingual-mixed" in names

    preset = mgr.get_preset("general-wiki")
    assert preset.category == "general"
    assert "algorithm" in preset.sample_text


def test_generate_corpus(tmp_path: Path):
    mgr = DatasetManager()
    out = tmp_path / "wiki_corpus.txt"
    mgr.generate_corpus("general-wiki", target_path=out, repeats=3)

    assert out.is_file()
    content = out.read_text(encoding="utf-8")
    assert "algorithm" in content
    assert len(content) > 1000


def test_prepare_corpus(tmp_path: Path):
    f1 = tmp_path / "part1.txt"
    f1.write_text("Paragraph A\n\nParagraph B")
    f2 = tmp_path / "part2.txt"
    f2.write_text("Paragraph B\n\nParagraph C")

    out = tmp_path / "merged.txt"
    mgr = DatasetManager()
    mgr.prepare_corpus([f1, f2], output_path=out, deduplicate=True)

    assert out.is_file()
    merged = out.read_text(encoding="utf-8")
    assert "Paragraph A" in merged
    assert "Paragraph B" in merged
    assert "Paragraph C" in merged
    # Duplicate Paragraph B should be included only once
    assert merged.count("Paragraph B") == 1
