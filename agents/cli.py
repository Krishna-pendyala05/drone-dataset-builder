from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from agents.crawler import DEFAULT_TIMEOUT_MS, CrawlResult, extract_with_self_healing
from agents.parser import deterministic_parser_factory

logger = logging.getLogger(__name__)


def _load_urls(path: Path) -> List[str]:
    urls: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("#"):
            continue
        urls.append(trimmed)
    return urls


def _load_llm_mapper(path: Optional[str]) -> Optional[Callable]:
    if not path:
        return None
    if ":" not in path:
        raise ValueError("LLM mapper must be in the form module:function")
    module_name, func_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    mapper = getattr(module, func_name, None)
    if not callable(mapper):
        raise ValueError(f"{path} is not callable")
    return mapper


async def build_dataset(
    seeds_path: Path,
    output_path: Path,
    llm_mapper: Optional[Callable] = None,
    max_attempts: int = 2,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> Path:
    urls = _load_urls(seeds_path)
    parser = deterministic_parser_factory(llm_mapper=llm_mapper)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    success_count = 0
    failed_count = 0
    skipped_count = 0
    failed_urls: List[str] = []

    with output_path.open("w", encoding="utf-8") as outfile:
        for url in urls:
            result: CrawlResult = await extract_with_self_healing(
                url,
                parser=parser,
                lookup_strategy=None,
                max_attempts=max_attempts,
                timeout_ms=timeout_ms,
            )
            if result.parsed is None:
                failed_count += 1
                skipped_count += 1
                failed_urls.append(url)
                logger.error(
                    "crawl.skipped url=%s attempts=%s elapsed_ms=%s errors=%s",
                    url,
                    result.metadata.get("attempts"),
                    result.metadata.get("total_elapsed_ms"),
                    result.metadata.get("errors"),
                )
                continue
            canonical = result.parsed.dict()
            payload = {
                "url": url,
                "data": canonical,
                "canonical": canonical,
                "raw_specs": result.raw_specs,
                "metadata": result.metadata,
            }
            outfile.write(json.dumps(payload) + "\n")
            records += 1
            success_count += 1
            logger.info(
                "crawl.success url=%s attempts=%s elapsed_ms=%s",
                url,
                result.metadata.get("attempts"),
                result.metadata.get("total_elapsed_ms"),
            )

    logger.info(
        "dataset.summary success=%s failed=%s skipped=%s failed_urls=%s",
        success_count,
        failed_count,
        skipped_count,
        failed_urls,
    )
    logger.info(
        "dataset.completed output=%s records=%s success=%s failed=%s skipped=%s",
        output_path,
        records,
        success_count,
        failed_count,
        skipped_count,
    )
    return output_path


def parse_args(args: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drone spec dataset generator")
    parser.add_argument("--seeds", required=True, help="Path to seeds file (one URL per line).")
    parser.add_argument(
        "--output",
        default="dataset.jsonl",
        help="Output JSONL path (default: dataset.jsonl)",
    )
    parser.add_argument(
        "--llm-mapper",
        default=None,
        help="Optional module:function to post-process deterministic fields with an LLM.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum crawl attempts per URL before failing.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=DEFAULT_TIMEOUT_MS,
        help="Navigation and load timeout in milliseconds (default: 90000).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(args=args)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(message)s")
    llm_mapper = _load_llm_mapper(args.llm_mapper)
    output_path = Path(args.output)
    seeds_path = Path(args.seeds)

    import asyncio

    asyncio.run(
        build_dataset(
            seeds_path=seeds_path,
            output_path=output_path,
            llm_mapper=llm_mapper,
            max_attempts=args.max_attempts,
            timeout_ms=args.timeout_ms,
        )
    )


if __name__ == "__main__":
    main()
