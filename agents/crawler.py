from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

from config.schemas import CameraDrone, DroneBase, EnterpriseDrone, FPVDrone
from agents.parser import MIN_FIELD_COUNT, deterministic_parser_factory

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


MarkdownParser = Callable[[Any, str], str]
SearchStrategy = Callable[[str, Dict[str, Any]], Optional[str]]


@dataclass
class CrawlResult:
    url: str
    markdown: str
    parsed: Optional[DroneBase] = None
    raw_specs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PageSnapshot:
    url: str
    final_url: str
    markdown: str
    full_html: str
    pruned_html: str
    title: str
    spec_text: str
    status: Optional[int] = None
    redirected: bool = False
    invalid: bool = False
    invalid_reason: Optional[str] = None


@dataclass
class PageSnapshot:
    url: str
    final_url: str
    markdown: str
    full_html: str
    pruned_html: str
    title: str
    spec_text: str
    status: Optional[int] = None
    redirected: bool = False
    invalid: bool = False
    invalid_reason: Optional[str] = None


SHADOW_PROBES: Tuple[str, ...] = (
    "table",
    "[role='table']",
    "[data-spec]",
    "dji-specs-table",
    "specs-table",
    "specification-table",
)


def _validate_result(parsed: DroneBase, url: str, parser_metadata: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    meta: Dict[str, Any] = {"parser_metadata": parser_metadata}
    reasons = []
    fields = parsed.model_dump()
    host = ""
    try:
        from urllib.parse import urlsplit

        host = urlsplit(url).netloc.lower()
    except Exception:  # noqa: BLE001
        host = ""
    if "dji.com" in host and (parsed.brand or "").strip().upper() != "DJI":
        reasons.append("brand_mismatch_for_domain")
    model = parsed.model or ""
    if len(model.strip()) < 2 or re.search(r"(https?:|//)", model, flags=re.IGNORECASE):
        reasons.append("invalid_model")
    field_count = sum(1 for _, value in fields.items() if value not in (None, ""))
    meta["field_count"] = field_count
    raw_specs = parser_metadata.get("raw_specs") or {}
    quality = parser_metadata.get("quality")
    meta["quality"] = quality
    if quality == "D" or (not raw_specs and not model.strip()):
        meta["invalid"] = True
        meta["reason"] = ";".join(reasons) if reasons else "insufficient_data"
        return False, meta
    if "invalid_model" in reasons and not raw_specs:
        meta["invalid"] = True
        meta["reason"] = ";".join(reasons)
        return False, meta
    if reasons:
        meta["reason"] = ";".join(reasons)
    return True, meta


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


async def _extract_content_html(page, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> str:
    script = """
    () => {
      const exclusionSelectors = [
        'header',
        'nav',
        'footer',
        'aside',
        '[role="navigation"]',
        '[aria-label*="cookie" i]',
        '[id*="cookie" i]',
        '[class*="cookie" i]',
        '[class*="consent" i]',
        '[class*="banner" i]',
      ];
      const isExcluded = (el) => exclusionSelectors.some((sel) => el.matches(sel));
      const isVisible = (el) => {
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity || '1') === 0) {
          return false;
        }
        const rect = el.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
      };
      const textLength = (el) => ((el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim().length);
      const candidates = [];
      document.querySelectorAll('main, article, section, div').forEach((el) => {
        if (!el || isExcluded(el) || !isVisible(el)) return;
        const len = textLength(el);
        if (len > 40) {
          candidates.push({ el, len });
        }
      });
      let target = null;
      const main = document.querySelector('main');
      if (main && isVisible(main) && !isExcluded(main)) {
        target = main;
      }
      if (!target && candidates.length) {
        target = candidates.reduce((max, current) => (current.len > max.len ? current : max)).el;
      }
      if (!target) {
        target = document.body || document.documentElement;
      }
      const clone = target.cloneNode(true);
      const innerRemove = exclusionSelectors.concat([
        '[class*="menu" i]',
        '[class*="sidebar" i]',
        '[class*="nav" i]',
        '[role="banner"]',
        '[data-testid*="cookie" i]',
      ]);
      innerRemove.forEach((sel) => clone.querySelectorAll(sel).forEach((node) => node.remove()));
      clone.querySelectorAll('script,style,noscript').forEach((node) => node.remove());
      return clone.outerHTML;
    }
    """
    return await _safe_evaluate(page, script, timeout_ms=timeout_ms)


async def _collect_spec_text(page, timeout_ms: int) -> str:
    """Extract inner text and Shadow DOM text from custom spec elements."""

    selectors = ["spec", "specs", "specification", "parameters"]
    script = """
    (tokens) => {
      const matches = [];
      const normalize = (text) => (text || "").replace(/\\s+/g, " ").trim();
      const crawl = (root) => {
        const all = root.querySelectorAll("*");
        all.forEach((el) => {
          const tag = (el.tagName || "").toLowerCase();
          const className = (el.className || "").toString().toLowerCase();
          if (!tag) return;
          const haystack = `${tag} ${className}`;
          if (!tokens.some((token) => haystack.includes(token))) return;
          if (el.shadowRoot) {
            matches.push({
              tag,
              hasShadow: true,
              text: normalize(el.shadowRoot.innerText || ""),
              html: el.shadowRoot.innerHTML || "",
            });
          } else {
            matches.push({
              tag,
              hasShadow: false,
              text: normalize(el.innerText || ""),
              html: el.innerHTML || "",
            });
          }
        });
      };
      crawl(document);
      return matches;
    }
    """
    results = await _safe_evaluate(page, script, selectors, timeout_ms=timeout_ms)
    texts = []
    if isinstance(results, list):
        for item in results:
            text = ""
            if isinstance(item, dict):
                text = item.get("text") or ""
                html = item.get("html") or ""
                if html and not text:
                    text = html
            if text:
                texts.append(text)
    combined = "\n".join(t for t in texts if t)
    return combined


async def fetch_snapshot(
    url: str,
    wait_selectors: Iterable[str] = SHADOW_PROBES,
    viewport: Tuple[int, int] = (1400, 900),
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> PageSnapshot:
    """Navigate with Playwright and capture both full and pruned HTML plus spec text."""

    logger = logging.getLogger(__name__)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
        full_html = ""
        pruned_html = ""
        page_title = ""
        spec_text = ""
        status_code: Optional[int] = None
        final_url = url
        try:
            response = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            if response is not None:
                status_code = response.status
            await _wait_for_settled(page, timeout_ms=timeout_ms)
            final_url = page.url
            try:
                page_title = await page.title()
            except Exception:  # noqa: BLE001
                page_title = ""
            await _progressive_scroll(page, timeout_ms=timeout_ms)
            await _deep_wait(page, timeout_ms=timeout_ms)
            spec_text = await _collect_spec_text(page, timeout_ms=timeout_ms)
            for selector in wait_selectors:
                try:
                    await page.wait_for_selector(selector, state="visible", timeout=2000)
                except PlaywrightTimeoutError:
                    continue
            full_html = await page.content()
            pruned_html = await _extract_content_html(page, timeout_ms=timeout_ms)
        except Exception as exc:  # noqa: BLE001
            logger.warning("crawl.partial_capture url=%s error=%s", url, exc)
            if not pruned_html:
                try:
                    pruned_html = await _extract_content_html(page, timeout_ms=timeout_ms)
                except Exception:  # noqa: BLE001
                    pruned_html = ""
            if not full_html:
                try:
                    full_html = await page.content()
                except Exception:  # noqa: BLE001
                    full_html = ""
            raise CrawlContentError(str(exc), markdown_snapshot=html_to_markdown(pruned_html or full_html)) from exc
        finally:
            await browser.close()

    redirected = final_url.rstrip("/") != url.rstrip("/")
    invalid = False
    invalid_reason = None
    title_lower = (page_title or "").lower()
    body_text = " ".join(re.sub(r"<[^>]+>", " ", full_html).split()).lower() if full_html else ""
    if status_code and status_code >= 400:
        invalid = True
        invalid_reason = f"status_{status_code}"
    elif any(token in title_lower for token in ["404", "page not found"]) or any(
        token in body_text for token in ["404", "page not found"]
    ):
        invalid = True
        invalid_reason = "page_not_found"

    markdown = html_to_markdown(full_html or pruned_html)
    if page_title:
        markdown = f"# {page_title}\n\n{markdown}"

    return PageSnapshot(
        url=url,
        final_url=final_url,
        markdown=markdown,
        full_html=full_html,
        pruned_html=pruned_html,
        title=page_title,
        spec_text=spec_text,
        status=status_code,
        redirected=redirected,
        invalid=invalid,
        invalid_reason=invalid_reason,
    )


def parse_with_agent(snapshot: PageSnapshot, url: str, parser: MarkdownParser) -> Tuple[DroneBase, Dict[str, Any]]:
    """Send page content to a secondary parsing agent with a semantic mapping prompt."""

    prompt = {
        "markdown": snapshot.markdown,
        "full_html": snapshot.full_html,
        "pruned_html": snapshot.pruned_html,
        "spec_text": snapshot.spec_text,
        "title": snapshot.title,
    }
    raw_response = parser(prompt, url)
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Parser returned invalid JSON for {url}") from exc

    parser_metadata: Dict[str, Any] = {}
    raw_specs: Dict[str, Any] = {}
    if isinstance(payload, dict):
        parser_metadata = payload.get("metadata", {}) or {}
        raw_specs = payload.get("raw_specs")
        if raw_specs is not None:
            parser_metadata["raw_specs"] = raw_specs
        if "canonical" in payload:
            payload = payload.get("canonical", {})
        elif "data" in payload:
            payload = payload.get("data", {})

    link = payload.get("link") or snapshot.final_url or url
    payload.pop("link", None)

    category = payload.get("category", "camera")
    if category == "fpv":
        return FPVDrone(**payload, link=link), parser_metadata
    if category == "enterprise":
        return EnterpriseDrone(**payload, link=link), parser_metadata
    return CameraDrone(**payload, link=link), parser_metadata


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
    parser_metadata: Dict[str, Any] = {}
    raw_specs: Dict[str, Any] = {}
    total_elapsed_ms = 0

    snapshot: Optional[PageSnapshot] = None
    while attempts < max_attempts and parsed is None:
        attempts += 1
        attempt_start = perf_counter()
        logger.info("crawl.start url=%s attempt=%s", url, attempts)
        try:
            snapshot = await fetch_snapshot(url, timeout_ms=timeout_ms)
            markdown = snapshot.markdown
            if snapshot.invalid:
                errors.append(snapshot.invalid_reason or "invalid_page")
                break
            parsed, parser_metadata = parse_with_agent(snapshot, url, parser)
            raw_specs = parser_metadata.pop("raw_specs", {}) if parser_metadata else {}
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

    snapshot_meta: Dict[str, Any] = {}
    if snapshot:
        snapshot_meta = {
            "final_url": snapshot.final_url,
            "redirected": snapshot.redirected,
            "status": snapshot.status,
            "invalid": snapshot.invalid,
            "invalid_reason": snapshot.invalid_reason,
        }

    if parsed is None:
        return CrawlResult(
            url=url,
            markdown=markdown,
            parsed=None,
            raw_specs=raw_specs,
            metadata={
                "healed": False,
                "attempts": attempts,
                "errors": errors,
                "total_elapsed_ms": total_elapsed_ms,
                "partial": bool(markdown),
                **snapshot_meta,
                "quality": "D" if snapshot and snapshot.invalid else None,
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
            alt_snapshot = await fetch_snapshot(alt_url)
            alt_parsed, alt_meta = parse_with_agent(alt_snapshot, alt_url, parser)
            if getattr(parsed, field) is None and getattr(alt_parsed, field):
                setattr(parsed, field, getattr(alt_parsed, field))
                healed = True
                parser_metadata.update(alt_meta)

    valid, validation_meta = _validate_result(parsed, url, {**parser_metadata, "raw_specs": raw_specs})
    combined_metadata = {
        "healed": healed or bool(missing_core),
        "missing_fields": missing_core,
        "attempts": attempts,
        "errors": errors,
        "total_elapsed_ms": total_elapsed_ms,
        "partial": bool(errors),
        **snapshot_meta,
        **validation_meta,
    }
    if not valid:
        return CrawlResult(
            url=url,
            markdown=markdown,
            parsed=None,
            raw_specs=raw_specs,
            metadata=combined_metadata,
        )

    return CrawlResult(
        url=url,
        markdown=markdown,
        parsed=parsed,
        raw_specs=raw_specs,
        metadata=combined_metadata,
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
    "fetch_snapshot",
    "PageSnapshot",
    "parse_with_agent",
    "extract_with_self_healing",
    "CrawlContentError",
    "run_sample",
]
