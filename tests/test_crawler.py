from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Callable, Dict

import pytest

from agents import crawler
from config.schemas import CameraDrone, EnterpriseDrone, FPVDrone


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def _load_expected(name: str) -> Dict[str, object]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "markdown_fixture", "expected_fixture", "expected_cls"),
    [
        (
            "https://example.com/camera",
            "camera.md",
            "camera.json",
            CameraDrone,
        ),
        (
            "https://example.com/fpv",
            "fpv.md",
            "fpv.json",
            FPVDrone,
        ),
        (
            "https://example.com/enterprise",
            "enterprise.md",
            "enterprise.json",
            EnterpriseDrone,
        ),
    ],
)
async def test_extract_with_self_healing_parses_categories_and_brand_model(
    monkeypatch: pytest.MonkeyPatch,
    url: str,
    markdown_fixture: str,
    expected_fixture: str,
    expected_cls: type,
) -> None:
    markdown_payload = _load_fixture(markdown_fixture)
    expected_json = _load_expected(expected_fixture)

    async def fake_fetch_markdown(target_url: str) -> str:
        assert target_url == url
        return markdown_payload

    def stub_parser(prompt: str, target_url: str) -> str:
        assert target_url == url
        return json.dumps({**expected_json, "link": target_url})

    monkeypatch.setattr(crawler, "fetch_markdown", fake_fetch_markdown)

    result = await crawler.extract_with_self_healing(url, parser=stub_parser)

    assert isinstance(result.parsed, expected_cls)
    assert result.parsed.brand == expected_json["brand"]
    assert result.parsed.model == expected_json["model"]
    assert result.parsed.category == expected_json["category"]
    assert result.parsed.link == url
    assert result.metadata["healed"] is False


@pytest.mark.asyncio
async def test_extract_with_self_healing_invokes_lookup_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    primary_url = "https://example.com/missing-brand"
    alt_url = "https://example.com/fallback"

    primary_md = _load_fixture("camera.md")
    fallback_md = _load_fixture("enterprise.md")

    async def fake_fetch_markdown(target_url: str) -> str:
        if target_url == primary_url:
            return primary_md
        if target_url == alt_url:
            return fallback_md
        raise AssertionError(f"Unexpected fetch target: {target_url}")

    def parser_factory() -> Callable[[str, str], str]:
        calls: Dict[str, Dict[str, object]] = {}

        def stub_parser(prompt: str, target_url: str) -> str:
            # Simulate missing brand on the first parse and populate on fallback.
            if target_url == primary_url:
                payload = {
                    "brand": None,
                    "model": "Aero X",
                    "category": "camera",
                    "max_flight_time": 31,
                }
            else:
                payload = {
                    "brand": "HealBrand",
                    "model": "Aero X",
                    "category": "enterprise",
                    "ingress_protection": "IP43",
                }
            calls[target_url] = payload
            return json.dumps({**payload, "link": target_url})

        stub_parser.calls = calls  # type: ignore[attr-defined]
        return stub_parser

    parser = parser_factory()

    def lookup_strategy(field: str, ctx: Dict[str, object]) -> str:
        assert field == "brand"
        assert ctx["url"] == primary_url
        return alt_url

    monkeypatch.setattr(crawler, "fetch_markdown", fake_fetch_markdown)

    result = await crawler.extract_with_self_healing(
        primary_url,
        parser=parser,
        lookup_strategy=lookup_strategy,
    )

    assert result.parsed is not None
    assert result.parsed.brand == "HealBrand"
    assert result.parsed.model == "Aero X"
    assert result.parsed.category == "camera"
    assert result.metadata["healed"] is True
