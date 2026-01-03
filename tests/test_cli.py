from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest

from agents import crawler
from agents.cli import build_dataset


FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.mark.asyncio
async def test_build_dataset_writes_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seeds_path = FIXTURE_DIR / "seeds.txt"
    output_path = tmp_path / "out.jsonl"

    markdown_map: Dict[str, str] = {
        "https://example.com/camera": (FIXTURE_DIR / "camera.md").read_text(encoding="utf-8"),
        "https://example.com/fpv": (FIXTURE_DIR / "fpv.md").read_text(encoding="utf-8"),
    }

    async def fake_fetch_markdown(url: str) -> str:
        return markdown_map[url]

    monkeypatch.setattr(crawler, "fetch_markdown", fake_fetch_markdown)

    await build_dataset(seeds_path=seeds_path, output_path=output_path, llm_mapper=None, max_attempts=1)

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    records = [json.loads(line) for line in lines]
    categories = [record["data"]["category"] for record in records]
    assert set(categories) == {"camera", "fpv"}
