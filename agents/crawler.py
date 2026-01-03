from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

from config.schemas import CameraDrone, DroneBase, EnterpriseDrone, FPVDrone
from agents.parser import deterministic_parser_factory

DEFAULT_TIMEOUT_MS = 90000


class CrawlContentError(RuntimeError):
    def __init__(self, message: str, markdown_snapshot: str | None = None):
        super().__init__(message)
        self.markdown_snapshot = markdown_snapshot or ""

# Optional Markdown conversion helper. If markdownify is not installed the
# fallback will still produce a readable, Markdown-compatible string.
markdownify_spec = importlib.util.find_spec("markdownify")
if markdownify_spec:
    from markdownify import markdownify as html_to_markdown  # type: ignore
else:
    def html_to_markdown(html: str) -> str:
        return "\n".join(segment.strip() for segment in html.splitlines() if segment.strip())


MarkdownParser = Callable[[str, str], str]
SearchStrategy = Callable[[str, Dict[str, Any]], Optional[str]]


@dataclass
class CrawlResult:
    url: str
    markdown: str
    parsed: Optional[DroneBase] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


SHADOW_PROBES: Tuple[str, ...] = (
    "table",
    "[role='table']",
    "[data-spec]",
    "dji-specs-table",
    "specs-table",
    "specification-table",
)


def _mapping_prompt(markdown: str) -> str:
    """Instruct a downstream model to bridge semantic gaps."""

    return f"""
You are a hardware-spec extraction agent. Convert the provided Markdown into normalized JSON.

Mapping rules:
- Map 'Endurance', 'Flight time', or 'Hover time' -> max_flight_time (minutes, numeric when possible).
- Map 'Sensor', 'Camera', 'Imager', or 'Payload sensor' -> sensor_type.
- Map 'Payload', 'Max payload', or 'Payload weight' -> payload_capacity in kilograms.
- Always populate brand and model if present anywhere in the text, even in prose.
- Preserve the source URL in the 'link' field.
- Output MUST be a single JSON object matching one of CameraDrone, FPVDrone, or EnterpriseDrone.
- If category is unclear, default to 'camera'.

Return ONLY JSON with no prose.

Markdown payload:
----------------
{markdown}
""".strip()


async def _deep_wait(page, timeout_ms: int = 12000) -> None:
    """Wait for JS-rendered spec tables, including Shadow DOM content."""

    await page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    await page.wait_for_timeout(600)

    async def has_shadow_tables(selectors: Sequence[str]) -> bool:
        return await _safe_evaluate(
            page,
            """
            (selectors) => {
              const seen = [];
              const crawl = (root) => {
                for (const sel of selectors) {
                  root.querySelectorAll(sel).forEach((el) => seen.push(el));
                }
                const shadowHosts = root.querySelectorAll('*');
                shadowHosts.forEach((node) => {
                  if (node.shadowRoot) {
                    crawl(node.shadowRoot);
                  }
                });
              };
              crawl(document);
              return seen.length > 0;
            }
            """,
            list(selectors),
            timeout_ms=timeout_ms,
        )

    deadline = timeout_ms
    interval = 400
    elapsed = 0
    while elapsed < deadline:
        if await has_shadow_tables(SHADOW_PROBES):
            return
        await page.wait_for_timeout(interval)
        elapsed += interval

    # Final attempt to ensure content is hydrated.
    await page.wait_for_load_state("networkidle", timeout=timeout_ms)


async def _wait_for_settled(page, timeout_ms: int) -> None:
    await page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    await page.wait_for_load_state("networkidle", timeout=timeout_ms)


async def _safe_evaluate(page, script: str, *args, timeout_ms: int) -> Any:
    attempts = 0
    while attempts < 2:
        attempts += 1
        try:
            await _wait_for_settled(page, timeout_ms=timeout_ms)
            return await page.evaluate(script, *args)
        except PlaywrightError as exc:
            message = str(exc)
            destroyed = "Execution context was destroyed" in message
            if destroyed and attempts < 2:
                await page.wait_for_timeout(300)
                continue
            raise


async def _progressive_scroll(page, timeout_ms: int) -> None:
    """Scroll using Playwright primitives to trigger lazy loading without brittle eval."""

    _ = timeout_ms
    for _ in range(6):
        await page.mouse.wheel(0, 1600)
        await page.wait_for_timeout(250)
    for _ in range(2):
        await page.keyboard.press("End")
        await page.wait_for_timeout(250)


async def fetch_markdown(
    url: str,
    wait_selectors: Iterable[str] = SHADOW_PROBES,
    viewport: Tuple[int, int] = (1400, 900),
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> str:
    """Navigate with Playwright and capture the page as Markdown after deep waits."""

    logger = logging.getLogger(__name__)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
        html = ""
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            await _wait_for_settled(page, timeout_ms=timeout_ms)
            await _progressive_scroll(page, timeout_ms=timeout_ms)
            await _deep_wait(page, timeout_ms=timeout_ms)
            for selector in wait_selectors:
                try:
                    await page.wait_for_selector(selector, state="visible", timeout=2000)
                except PlaywrightTimeoutError:
                    continue
            html = await page.content()
            return html_to_markdown(html)
        except Exception as exc:  # noqa: BLE001
            if not html:
                try:
                    html = await page.content()
                except Exception:  # noqa: BLE001
                    html = ""
            markdown_snapshot = html_to_markdown(html) if html else ""
            logger.warning("crawl.partial_capture url=%s error=%s", url, exc)
            raise CrawlContentError(str(exc), markdown_snapshot=markdown_snapshot) from exc
        finally:
            await browser.close()


def parse_with_agent(markdown: str, url: str, parser: MarkdownParser) -> DroneBase:
    """Send Markdown to a secondary parsing agent with a semantic mapping prompt."""

    prompt = _mapping_prompt(markdown)
    raw_response = parser(prompt, url)
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Parser returned invalid JSON for {url}") from exc

    link = payload.get("link") or url
    payload.pop("link", None)

    category = payload.get("category", "camera")
    if category == "fpv":
        return FPVDrone(**payload, link=link)
    if category == "enterprise":
        return EnterpriseDrone(**payload, link=link)
    return CameraDrone(**payload, link=link)


async def extract_with_self_healing(
    url: str,
    parser: MarkdownParser,
    lookup_strategy: Optional[SearchStrategy] = None,
    max_attempts: int = 2,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> CrawlResult:
    """
    Crawl, parse, and self-heal missing critical fields.

    If brand or model is empty, a secondary targeted fetch is triggered using the
    provided lookup_strategy to locate an alternative URL.
    """

    logger = logging.getLogger(__name__)
    attempts = 0
    errors = []
    markdown = ""
    parsed: Optional[DroneBase] = None
    total_elapsed_ms = 0

    while attempts < max_attempts and parsed is None:
        attempts += 1
        attempt_start = perf_counter()
        logger.info("crawl.start url=%s attempt=%s", url, attempts)
        try:
            markdown = await fetch_markdown(url, timeout_ms=timeout_ms)
            parsed = parse_with_agent(markdown, url, parser)
            attempt_elapsed_ms = int((perf_counter() - attempt_start) * 1000)
            total_elapsed_ms += attempt_elapsed_ms
            logger.info("crawl.success url=%s attempt=%s elapsed_ms=%s", url, attempts, attempt_elapsed_ms)
        except Exception as exc:  # noqa: BLE001
            attempt_elapsed_ms = int((perf_counter() - attempt_start) * 1000)
            total_elapsed_ms += attempt_elapsed_ms
            error_text = f"{type(exc).__name__}: {exc}"
            metadata_partial = isinstance(exc, CrawlContentError)
            if metadata_partial and isinstance(exc, CrawlContentError) and exc.markdown_snapshot:
                markdown = exc.markdown_snapshot
            log_action = "crawl.partial_success" if metadata_partial else "crawl.failed"
            logger.warning(
                "%s url=%s attempt=%s elapsed_ms=%s error=%s",
                log_action,
                url,
                attempts,
                attempt_elapsed_ms,
                error_text,
            )
            errors.append(error_text)
            if attempts >= max_attempts:
                break
            continue

    if parsed is None:
        return CrawlResult(
            url=url,
            markdown=markdown,
            parsed=None,
            metadata={
                "healed": False,
                "attempts": attempts,
                "errors": errors,
                "total_elapsed_ms": total_elapsed_ms,
                "partial": bool(markdown),
            },
        )

    missing_core = [field for field in ("brand", "model") if not getattr(parsed, field)]
    healed = False
    if missing_core and lookup_strategy:
        for field in missing_core:
            alt_url = lookup_strategy(field, {"url": url, "parsed": parsed.model_dump()})
            if not alt_url:
                continue
            logger.info("heal.lookup", extra={"field": field, "url": url, "alt_url": alt_url})
            alt_markdown = await fetch_markdown(alt_url)
            alt_parsed = parse_with_agent(alt_markdown, alt_url, parser)
            if getattr(parsed, field) is None and getattr(alt_parsed, field):
                setattr(parsed, field, getattr(alt_parsed, field))
                healed = True

    return CrawlResult(
        url=url,
        markdown=markdown,
        parsed=parsed,
        metadata={
            "healed": healed or bool(missing_core),
            "missing_fields": missing_core,
            "attempts": attempts,
            "errors": errors,
            "total_elapsed_ms": total_elapsed_ms,
            "partial": bool(errors),
        },
    )


def run_sample():
    """
    Example usage.

    Supply a parser callable that accepts (prompt, url) and returns a JSON string.
    """

    async def _runner():
        deterministic_parser = deterministic_parser_factory()

        result = await extract_with_self_healing(
            "https://example.com/drone",
            parser=deterministic_parser,
            lookup_strategy=lambda field, ctx: ctx["url"],
        )
        return result

    return asyncio.run(_runner())


__all__ = [
    "DEFAULT_TIMEOUT_MS",
    "CrawlResult",
    "fetch_markdown",
    "parse_with_agent",
    "extract_with_self_healing",
    "CrawlContentError",
    "run_sample",
]
