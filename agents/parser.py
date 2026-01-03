from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

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


KEY_MAP: Dict[str, str] = {
    "brand": "brand",
    "manufacturer": "brand",
    "maker": "brand",
    "model": "model",
    "name": "model",
    "product": "model",
    "flight time": "max_flight_time",
    "endurance": "max_flight_time",
    "hover time": "max_flight_time",
    "max flight time": "max_flight_time",
    "maximum flight time": "max_flight_time",
    "sensor": "sensor_type",
    "camera": "sensor_type",
    "payload sensor": "sensor_type",
    "imager": "sensor_type",
    "payload": "payload_capacity",
    "max payload": "payload_capacity",
    "payload weight": "payload_capacity",
    "maximum payload": "payload_capacity",
    "max speed": "max_speed",
    "top speed": "max_speed",
    "speed": "max_speed",
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


def _normalize_key(raw: str) -> str:
    lowered = raw.strip().lower().replace("\uff1a", ":")
    cleaned = re.sub(r"[\\*`_]", "", lowered).strip(" :")
    for key in sorted(KEY_MAP, key=len, reverse=True):
        if key in cleaned:
            return KEY_MAP[key]
    return cleaned.replace(" ", "_")


def _coerce_number(value: str) -> Optional[float]:
    match = re.search(r"([-+]?[0-9]+(?:\.[0-9]+)?)", value.replace(",", ""))
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
        for key, value in row.items():
            clean_key = _normalize_key(key)
            kv.setdefault(clean_key, value)
    return kv


def _cast_fields(kv: Dict[str, str], markdown: str, url: str) -> Dict[str, object]:
    fields: Dict[str, object] = {}
    for raw_key, raw_val in kv.items():
        normalized_key = _normalize_key(raw_key)
        value = raw_val
        if normalized_key in {"max_flight_time", "payload_capacity", "max_speed"}:
            number = _coerce_number(value)
            fields[normalized_key] = number if number is not None else value
        elif normalized_key in {"gimbal", "supports_hd_link", "thermal_capability"}:
            bool_val = _coerce_bool(value)
            fields[normalized_key] = bool_val if bool_val is not None else value
        elif normalized_key in {"video_tx_power_mw"}:
            num = _coerce_number(value)
            fields[normalized_key] = int(num) if num is not None else value
        else:
            fields[normalized_key] = value
    if "brand" not in fields:
        heading_match = re.search(r"^#+\s*(.+)$", markdown, flags=re.MULTILINE)
        if heading_match:
            parts = heading_match.group(1).split(" ", 1)
            if len(parts) == 2:
                fields["brand"] = parts[0]
                fields.setdefault("model", parts[1])
    fields["category"] = _maybe_category(fields, markdown)
    fields.setdefault("link", url)
    return fields


def deterministic_parser_factory(
    llm_mapper: Optional[Callable[[Dict[str, object], str, str], Dict[str, object]]] = None,
) -> Callable[[str, str], str]:
    def parser(prompt: str, url: str) -> str:
        markdown = _extract_markdown(prompt)
        lines = [line for line in markdown.splitlines() if line.strip()]
        tables = _parse_table(lines)
        table_kv = _collapse_tables(tables)
        kv = {**table_kv, **_parse_key_values(lines)}
        fields = _cast_fields(kv, markdown, url)
        if llm_mapper:
            try:
                mapped = llm_mapper(fields, markdown, url)
                if isinstance(mapped, dict):
                    fields.update({k: v for k, v in mapped.items() if v is not None})
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM mapper failed; continuing with deterministic fields", exc_info=exc)
        drone = _to_schema(fields)
        return drone.json()

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
]
