from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit

from config.schemas import CameraDrone, DroneBase, EnterpriseDrone, FPVDrone

logger = logging.getLogger(__name__)


@dataclass
class ParsedFields:
    brand: Optional[str] = None
    model: Optional[str] = None
    category: Optional[str] = None
    max_flight_time: Optional[float] = None
    sensor_type: Optional[str] = None
    payload_capacity: Optional[float] = None
    max_speed: Optional[float] = None
    notes: Optional[str] = None
    gimbal: Optional[bool] = None
    camera_resolution: Optional[str] = None
    video_tx_power_mw: Optional[int] = None
    supports_hd_link: Optional[bool] = None
    ingress_protection: Optional[str] = None
    thermal_capability: Optional[bool] = None
    operating_temp_c: Optional[str] = None


SPEC_KEY_MAP: Dict[str, str] = {
    # Core specs
    "flight time": "max_flight_time",
    "endurance": "max_flight_time",
    "hover time": "max_flight_time",
    "max flight time": "max_flight_time",
    "maximum flight time": "max_flight_time",
    "max endurance": "max_flight_time",
    "sensor": "sensor_type",
    "sensor type": "sensor_type",
    "camera sensor": "sensor_type",
    "imager": "sensor_type",
    "max speed": "max_speed",
    "top speed": "max_speed",
    "speed": "max_speed",
    # Payload and extras
    "payload": "payload_capacity",
    "max payload": "payload_capacity",
    "payload weight": "payload_capacity",
    "maximum payload": "payload_capacity",
    "gimbal": "gimbal",
    "camera resolution": "camera_resolution",
    "resolution": "camera_resolution",
    "video tx power": "video_tx_power_mw",
    "vtx power": "video_tx_power_mw",
    "video transmitter power": "video_tx_power_mw",
    "hd link": "supports_hd_link",
    "digital link": "supports_hd_link",
    "ingress protection": "ingress_protection",
    "ip rating": "ingress_protection",
    "thermal": "thermal_capability",
    "thermal capability": "thermal_capability",
    "operating temperature": "operating_temp_c",
}

IDENTITY_KEYS: Dict[str, str] = {
    "brand": "brand",
    "manufacturer": "brand",
    "maker": "brand",
    "model": "model",
    "name": "model",
    "product": "model",
}

MIN_FIELD_COUNT = 5


def _normalize_key(raw: str) -> Optional[str]:
    lowered = raw.strip().lower().replace("\uff1a", ":")
    cleaned = re.sub(r"[\\*`]", "", lowered).strip(" :")
    normalized = cleaned.replace("_", " ")
    for key in sorted({**SPEC_KEY_MAP, **IDENTITY_KEYS}, key=len, reverse=True):
        if key in normalized:
            return ({**SPEC_KEY_MAP, **IDENTITY_KEYS})[key]
    if normalized in {"brand", "model"}:
        return cleaned
    return None


def _normalize_placeholder(value: object) -> Optional[object]:
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed == "":
            return None
        lowered = trimmed.lower()
        if lowered in {"—", "n/a", "na", "-", "–"}:
            return None
    return value


def _coerce_number(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value)
    match = re.search(r"([-+]?[0-9]+(?:\.[0-9]+)?)", text.replace(",", ""))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _coerce_bool(value: str) -> Optional[bool]:
    lowered = value.strip().lower()
    if lowered in {"yes", "true", "y", "1", "included", "standard"}:
        return True
    if lowered in {"no", "false", "n", "0", "not included", "optional"}:
        return False
    return None


def _maybe_category(fields: Dict[str, object], markdown: str) -> str:
    text = markdown.lower()
    if any(k in text for k in ["fpv", "vtx", "video tx", "goggles"]):
        return "fpv"
    if any(k in text for k in ["ingress", "ip", "thermal", "enterprise", "payload capacity"]):
        return "enterprise"
    if "category" in fields and isinstance(fields["category"], str):
        category = fields["category"].lower()
        if category in {"fpv", "enterprise", "camera"}:
            return category
    return "camera"


def _extract_markdown(prompt: str) -> str:
    if "Markdown payload:" not in prompt:
        return prompt
    marker = "Markdown payload:"
    _, remainder = prompt.split(marker, 1)
    return remainder.strip()


def _parse_table(lines: List[str]) -> List[Dict[str, str]]:
    tables: List[Dict[str, str]] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("|") and "|" in lines[i].strip()[1:]:
            header_line = lines[i]
            if i + 2 <= len(lines):
                header = [cell.strip() for cell in header_line.strip("|").split("|")]
                i += 1
                if set(lines[i].strip()) <= {"|", "-", ":", " "}:
                    i += 1
                while i < len(lines) and lines[i].strip().startswith("|"):
                    row_cells = [cell.strip() for cell in lines[i].strip("|").split("|")]
                    if len(row_cells) == len(header):
                        tables.append(dict(zip(header, row_cells)))
                    i += 1
                continue
        i += 1
    return tables


def _parse_key_values(lines: Iterable[str]) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    patterns = [
        re.compile(r"^\s*[-*]\s*([^:]+):\s*(.+)$"),
        re.compile(r"^\s*([^:]{2,}):\s*(.+)$"),
    ]
    for line in lines:
        for pattern in patterns:
            match = pattern.match(line)
            if match:
                key, value = match.group(1).strip(" *`"), match.group(2).strip()
                value = value.lstrip("* ").strip()
                kv[key] = value
                break
    return kv


def _collapse_tables(tables: List[Dict[str, str]]) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    for row in tables:
        if len(row) == 2:
            first_key, second_key = list(row.keys())
            raw_key_candidate = row[first_key]
            raw_value_candidate = row[second_key]
            normalized_key = _normalize_key(raw_key_candidate)
            if normalized_key:
                kv.setdefault(normalized_key, raw_value_candidate)
                continue
        for key, value in row.items():
            normalized_key = _normalize_key(key)
            if normalized_key:
                kv.setdefault(normalized_key, value)
    return kv


def _filter_lines(lines: List[str]) -> List[str]:
    filtered: List[str] = []
    for line in lines:
        trimmed = line.strip()
        if not trimmed:
            continue
        link_only = re.fullmatch(r"[-*]?\s*\[.+\]\(.+\)", trimmed)
        if link_only:
            continue
        if "Suggested searches" in trimmed or trimmed.lower().startswith("support"):
            continue
        filtered.append(line)
    return filtered


def _select_spec_section(markdown: str) -> Tuple[List[str], Dict[str, object]]:
    lines = [line for line in markdown.splitlines()]
    sections: List[Tuple[int, str]] = []
    for idx, line in enumerate(lines):
        heading_match = re.match(r"^(#+)\s*(.+)$", line.strip())
        if heading_match:
            sections.append((idx, heading_match.group(2).strip()))
    spec_indices: List[int] = []
    for idx, title in sections:
        lowered = title.lower()
        if "spec" in lowered:
            spec_indices.append(idx)
    if not spec_indices:
        for idx, line in enumerate(lines):
            if "#spec" in line.lower() or "anchor=\"spec" in line.lower():
                spec_indices.append(idx)
                break
    if not spec_indices:
        return _filter_lines(lines), {"spec_section_found": False, "section_title": None}
    start = min(spec_indices)
    next_headings = [idx for idx, _ in sections if idx > start]
    end = min(next_headings) if next_headings else len(lines)
    return _filter_lines(lines[start:end]), {
        "spec_section_found": True,
        "section_title": lines[start].lstrip("# ").strip() if start < len(lines) else None,
    }


def _map_fields(kv: Dict[str, str]) -> Tuple[Dict[str, object], Dict[str, str]]:
    fields: Dict[str, object] = {}
    unmapped: Dict[str, str] = {}
    for raw_key, raw_val in kv.items():
        normalized_key = _normalize_key(raw_key)
        normalized_value = _normalize_placeholder(raw_val)
        if normalized_key is None or normalized_value is None:
            if raw_key.strip():
                unmapped[raw_key.strip()] = raw_val
            continue
        if isinstance(normalized_value, str) and re.search(r"(https?:|www\\.|//)", normalized_value, re.IGNORECASE):
            if normalized_key != "link":
                unmapped[raw_key.strip()] = raw_val
                continue
        if normalized_key in {"max_flight_time", "payload_capacity", "max_speed"}:
            number = _coerce_number(normalized_value)
            fields[normalized_key] = number
        elif normalized_key in {"gimbal", "supports_hd_link", "thermal_capability"}:
            bool_val = _coerce_bool(normalized_value if isinstance(normalized_value, str) else str(normalized_value))
            fields[normalized_key] = bool_val if bool_val is not None else _normalize_placeholder(normalized_value)
        elif normalized_key in {"video_tx_power_mw"}:
            num = _coerce_number(normalized_value)
            fields[normalized_key] = int(num) if num is not None else _normalize_placeholder(normalized_value)
        else:
            fields[normalized_key] = _normalize_placeholder(normalized_value)
    return fields, unmapped


def _derive_identity(fields: Dict[str, object], markdown: str, url: str) -> None:
    url_host = urlsplit(url).netloc.lower()
    if "dji.com" in url_host:
        fields["brand"] = "DJI"
    heading_match = re.search(r"^#\s*(.+)$", markdown, flags=re.MULTILINE)
    title = heading_match.group(1).strip() if heading_match else None
    if not title:
        secondary_heading = re.search(r"^##\s*(.+)$", markdown, flags=re.MULTILINE)
        if secondary_heading:
            title = secondary_heading.group(1).strip()
    if title and "spec" not in title.lower():
        brand_prefix = str(fields.get("brand", "") or "")
        model_candidate = title
        if brand_prefix and model_candidate.lower().startswith(brand_prefix.lower()):
            model_candidate = model_candidate[len(brand_prefix) :].strip(" -|")
        if model_candidate:
            fields.setdefault("model", model_candidate)
    if "model" not in fields or not fields["model"]:
        path_parts = [part for part in urlsplit(url).path.split("/") if part]
        slug = path_parts[-1] if path_parts else ""
        normalized_slug = slug.replace("-", " ").replace("_", " ").strip()
        if normalized_slug:
            words = [word.upper() if word.isupper() else word.capitalize() for word in normalized_slug.split()]
            fields.setdefault("model", " ".join(words))
    fields["category"] = _maybe_category(fields, markdown)
    fields.setdefault("link", url)


def deterministic_parser_factory(
    llm_mapper: Optional[Callable[[Dict[str, object], str, str], Dict[str, object]]] = None,
) -> Callable[[str, str], str]:
    def parser(prompt: str, url: str) -> str:
        markdown = _extract_markdown(prompt)
        spec_lines, section_info = _select_spec_section(markdown)
        tables = _parse_table(spec_lines)
        table_kv = _collapse_tables(tables)
        kv = {**table_kv, **_parse_key_values(spec_lines)}
        fields, unmapped = _map_fields(kv)
        _derive_identity(fields, markdown, url)
        if llm_mapper:
            try:
                mapped = llm_mapper(fields, markdown, url)
                if isinstance(mapped, dict):
                    fields.update({k: v for k, v in mapped.items() if v is not None})
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM mapper failed; continuing with deterministic fields", exc_info=exc)
        drone = _to_schema(fields)
        metadata = {
            "unmapped_specs": unmapped,
            "mapped_fields": sorted([key for key, value in fields.items() if value is not None]),
            **section_info,
        }
        return json.dumps({"data": drone.dict(), "metadata": metadata})

    return parser


def _to_schema(fields: Dict[str, object]) -> DroneBase:
    category = str(fields.get("category", "camera")).lower()
    if category == "fpv":
        return FPVDrone(**fields)
    if category == "enterprise":
        return EnterpriseDrone(**fields)
    return CameraDrone(**fields)


__all__ = [
    "deterministic_parser_factory",
    "MIN_FIELD_COUNT",
]
