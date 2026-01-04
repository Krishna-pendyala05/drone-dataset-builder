from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Callable, Dict

import pytest
from playwright.async_api import Error as PlaywrightError

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

    async def fake_fetch_snapshot(target_url: str, **_kwargs) -> crawler.PageSnapshot:
        assert target_url == url
        return crawler.PageSnapshot(
            url=target_url,
            final_url=target_url,
            markdown=markdown_payload,
            full_html="",
            pruned_html="",
            title="",
            spec_text="",
            status=200,
        )

    def stub_parser(prompt: str, target_url: str) -> str:
        assert target_url == url
        return json.dumps({**expected_json, "link": target_url})

    monkeypatch.setattr(crawler, "fetch_snapshot", fake_fetch_snapshot)

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

    async def fake_fetch_snapshot(target_url: str, **_kwargs) -> crawler.PageSnapshot:
        if target_url == primary_url:
            payload = primary_md
        elif target_url == alt_url:
            payload = fallback_md
        else:
            raise AssertionError(f"Unexpected fetch target: {target_url}")
        return crawler.PageSnapshot(
            url=target_url,
            final_url=target_url,
            markdown=payload,
            full_html="",
            pruned_html="",
            title="",
            spec_text="",
            status=200,
        )

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

    monkeypatch.setattr(crawler, "fetch_snapshot", fake_fetch_snapshot)

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


@pytest.mark.asyncio
async def test_safe_evaluate_recovers_after_context_destruction(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"evaluate": 0, "load": 0, "timeout": 0}

    class DummyMouse:
        async def wheel(self, *_args, **_kwargs):
            return None

    class DummyKeyboard:
        async def press(self, *_args, **_kwargs):
            return None

    class DummyPage:
        mouse = DummyMouse()
        keyboard = DummyKeyboard()

        async def wait_for_load_state(self, *_args, **_kwargs):
            calls["load"] += 1
            return None

        async def wait_for_timeout(self, *_args, **_kwargs):
            calls["timeout"] += 1
            return None

        async def evaluate(self, *_args, **_kwargs):
            calls["evaluate"] += 1
            if calls["evaluate"] == 1:
                raise PlaywrightError("Execution context was destroyed, retry")
            return {"ok": True}

    page = DummyPage()
    result = await crawler._safe_evaluate(page, "(sel) => sel", ["a"], timeout_ms=1000)  # type: ignore[attr-defined]

    assert result == {"ok": True}
    assert calls["evaluate"] == 2


@pytest.mark.asyncio
async def test_extract_with_self_healing_rejects_invalid_records(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "https://www.dji.com/global/fake"

    async def fake_fetch_snapshot(target_url: str, **_kwargs) -> crawler.PageSnapshot:
        assert target_url == url
        return crawler.PageSnapshot(
            url=target_url,
            final_url=target_url,
            markdown="## Placeholder content",
            full_html="",
            pruned_html="",
            title="",
            spec_text="",
        )

    def stub_parser(prompt: str, target_url: str) -> str:
        assert target_url == url
        payload = {
            "brand": "OtherBrand",
            "model": "x",
            "category": "camera",
            "max_flight_time": None,
            "max_speed": None,
            "link": target_url,
        }
        return json.dumps(payload)

    monkeypatch.setattr(crawler, "fetch_snapshot", fake_fetch_snapshot)

    result = await crawler.extract_with_self_healing(url, parser=stub_parser, max_attempts=1)

    assert result.parsed is None
    assert result.metadata.get("invalid") is True
    assert "brand_mismatch_for_domain" in result.metadata.get("reason", "")


@pytest.mark.asyncio
async def test_extract_with_self_healing_marks_404_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "https://www.dji.com/mavic-3/specs"
    html_404 = _load_fixture("404.html")

    async def fake_fetch_snapshot(target_url: str, **_kwargs) -> crawler.PageSnapshot:
        assert target_url == url
        return crawler.PageSnapshot(
            url=target_url,
            final_url=target_url,
            markdown=html_404,
            full_html=html_404,
            pruned_html=html_404,
            title="404 | Page not found",
            spec_text="",
            status=404,
            invalid=True,
            invalid_reason="status_404",
        )

    def stub_parser(prompt: str, target_url: str) -> str:  # noqa: ARG001
        return json.dumps({})

    monkeypatch.setattr(crawler, "fetch_snapshot", fake_fetch_snapshot)

    result = await crawler.extract_with_self_healing(url, parser=stub_parser, max_attempts=1)

    assert result.parsed is None
    assert result.metadata.get("invalid") is True
    assert result.metadata.get("quality") == "D"
