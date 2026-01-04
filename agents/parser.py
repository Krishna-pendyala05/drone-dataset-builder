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
SECTION_KEYWORDS = [
    "spec",
    "specification",
    "technical",
    "parameters",
    "details",
    "aircraft",
    "camera",
    "gimbal",
    "transmission",
]


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


def _normalize_raw_key(raw: str) -> Optional[str]:
    cleaned = raw.strip().strip(":")
    if not cleaned:
        return None
    cleaned = re.sub(r"[\\*`]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower()


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


def _parse_minutes(value: str) -> Optional[float]:
    lowered = value.lower()
    if "min" not in lowered and "minute" not in lowered:
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", lowered.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_speed(value: str) -> Optional[float]:
    lowered = value.lower()
    number_match = None
    if "m/s" in lowered or "meter/second" in lowered:
        number_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:m/s|meter/second)", lowered)
        if number_match:
            return float(number_match.group(1))
    if any(unit in lowered for unit in ["km/h", "kph", "kmh"]):
        number_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:km/h|kph|kmh)", lowered)
        if number_match:
            return float(number_match.group(1)) / 3.6
    return None


def _looks_like_sensor(value: str) -> bool:
    lowered = value.lower()
    if "cmos" in lowered or "ccd" in lowered:
        return True
    if "effective pixel" in lowered:
        return True
    if re.search(r"[0-9]+/[0-9.]+\s*(?:inch|in)", lowered):
        return True
    if re.search(r"[0-9.]+\s*inch", lowered):
        return True
    return False


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


def _parse_key_values(lines: Iterable[str]) -> List[Tuple[str, str, str]]:
    kv: List[Tuple[str, str, str]] = []
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
                kv.append((key, value, line.strip()))
                break
    return kv


def _collapse_tables(tables: List[Dict[str, str]]) -> List[Tuple[str, str, str]]:
    entries: List[Tuple[str, str, str]] = []
    for row in tables:
        if len(row) == 2:
            first_key, second_key = list(row.keys())
            raw_key_candidate = row[first_key]
            raw_value_candidate = row[second_key]
            entries.append((raw_key_candidate, raw_value_candidate, f"{raw_key_candidate}: {raw_value_candidate}"))
            continue
        for key, value in row.items():
            entries.append((key, value, f"{key}: {value}"))
    return entries


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
        if any(keyword in lowered for keyword in SECTION_KEYWORDS):
            spec_indices.append(idx)
    if not spec_indices:
        for idx, line in enumerate(lines):
            if "#spec" in line.lower() or "anchor=\"spec" in line.lower():
                spec_indices.append(idx)
                break
    if not spec_indices:
        return _filter_lines(lines), {"spec_section_found": False, "section_title": None}
    spec_indices_sorted = sorted(spec_indices)
    if 0 in spec_indices_sorted and len(spec_indices_sorted) > 1:
        start = spec_indices_sorted[1]
    else:
        start = spec_indices_sorted[0]
    next_headings = [idx for idx, _ in sections if idx > start]
    end = min(next_headings) if next_headings else len(lines)
    return _filter_lines(lines[start:end]), {
        "spec_section_found": True,
        "section_title": lines[start].lstrip("# ").strip() if start < len(lines) else None,
    }


def _collect_raw_specs(entries: List[Tuple[str, str, str]]) -> Dict[str, Dict[str, str]]:
    raw_specs: Dict[str, Dict[str, str]] = {}
    for raw_key, raw_val, evidence in entries:
        normalized_key = _normalize_raw_key(raw_key or "")
        if not normalized_key:
            continue
        raw_specs.setdefault(
            normalized_key,
            {
                "value": raw_val,
                "evidence": evidence,
                "raw_key": raw_key,
            },
        )
    return raw_specs


def _canonical_field_from_key(raw_key: str) -> Optional[str]:
    key = raw_key.lower()
    key = re.sub(r"\s+", " ", key)
    if any(token in key for token in ["flight time", "endurance", "hover time", "maximum flight time"]):
        return "max_flight_time"
    if any(token in key for token in ["max speed", "top speed", "speed"]):
        return "max_speed"
    if any(token in key for token in ["sensor", "camera sensor", "sensor type", "image sensor", "imager", "effective pixel"]):
        return "sensor_type"
    if any(token in key for token in ["payload", "payload capacity", "max payload", "payload weight", "maximum payload"]):
        return "payload_capacity"
    if "gimbal" in key:
        return "gimbal"
    if any(token in key for token in ["video tx power", "vtx power", "video transmitter power"]):
        return "video_tx_power_mw"
    if any(token in key for token in ["hd link", "digital link"]):
        return "supports_hd_link"
    if any(token in key for token in ["ingress protection", "ip rating"]):
        return "ingress_protection"
    if "thermal" in key:
        return "thermal_capability"
    if "operating temperature" in key:
        return "operating_temp_c"
    return None


def _map_fields(raw_specs: Dict[str, Dict[str, str]]) -> Tuple[Dict[str, object], Dict[str, str], Dict[str, str]]:
    fields: Dict[str, object] = {}
    unmapped: Dict[str, str] = {}
    mapped_from: Dict[str, str] = {}

    for normalized_key, entry in raw_specs.items():
        raw_key = entry.get("raw_key") or normalized_key
        value = entry.get("value")
        if value is None:
            continue
        normalized_value = _normalize_placeholder(value)
        if normalized_value is None:
            continue
        identity_key = _normalize_key(raw_key or "")
        if identity_key in {"brand", "model"}:
            fields.setdefault(identity_key, str(normalized_value).strip())
            mapped_from[identity_key] = normalized_key
            continue
        canonical_key = _canonical_field_from_key(raw_key)
        text_value = str(normalized_value)
        if canonical_key == "max_flight_time":
            minutes = _parse_minutes(text_value)
            if minutes is not None:
                fields["max_flight_time"] = minutes
                mapped_from["max_flight_time"] = normalized_key
            else:
                unmapped[raw_key] = value
        elif canonical_key == "max_speed":
            speed_ms = _parse_speed(text_value)
            if speed_ms is not None:
                fields["max_speed"] = speed_ms
                mapped_from["max_speed"] = normalized_key
            else:
                unmapped[raw_key] = value
        elif canonical_key == "sensor_type":
            if _looks_like_sensor(text_value):
                fields["sensor_type"] = text_value.strip()
                mapped_from["sensor_type"] = normalized_key
            else:
                unmapped[raw_key] = value
        elif canonical_key == "payload_capacity":
            weight = _coerce_number(text_value)
            if weight is not None:
                fields["payload_capacity"] = weight
                mapped_from["payload_capacity"] = normalized_key
            else:
                unmapped[raw_key] = value
        elif canonical_key == "gimbal":
            bool_val = _coerce_bool(text_value)
            if bool_val is not None:
                fields["gimbal"] = bool_val
                mapped_from["gimbal"] = normalized_key
            else:
                unmapped[raw_key] = value
        elif canonical_key == "video_tx_power_mw":
            num = _coerce_number(text_value)
            if num is not None:
                fields["video_tx_power_mw"] = int(num)
                mapped_from["video_tx_power_mw"] = normalized_key
            else:
                unmapped[raw_key] = value
        elif canonical_key == "supports_hd_link":
            bool_val = _coerce_bool(text_value)
            if bool_val is not None:
                fields["supports_hd_link"] = bool_val
                mapped_from["supports_hd_link"] = normalized_key
            else:
                unmapped[raw_key] = value
        elif canonical_key == "ingress_protection":
            fields["ingress_protection"] = text_value.strip()
            mapped_from["ingress_protection"] = normalized_key
        elif canonical_key == "thermal_capability":
            bool_val = _coerce_bool(text_value)
            if bool_val is not None:
                fields["thermal_capability"] = bool_val
                mapped_from["thermal_capability"] = normalized_key
            else:
                unmapped[raw_key] = value
        elif canonical_key == "operating_temp_c":
            fields["operating_temp_c"] = text_value.strip()
            mapped_from["operating_temp_c"] = normalized_key
        else:
            unmapped[raw_key] = value

    return fields, unmapped, mapped_from


def _clean_model_candidate(candidate: str, brand_prefix: str = "") -> Optional[str]:
    cleaned = candidate.strip().strip("|").strip("-")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if re.fullmatch(r"#+", cleaned):
        return None
    if "customer service" in lowered:
        return None
    if re.search(r"(https?:|//)", cleaned):
        return None
    if brand_prefix and cleaned.lower().startswith(brand_prefix.lower()):
        cleaned = cleaned[len(brand_prefix) :].strip(" -|")
    cleaned = re.sub(r"\s*-\s*specs?\s*\|?\s*dji", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\|\s*dji", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    return cleaned or None


def _derive_identity(fields: Dict[str, object], markdown: str, url: str) -> None:
    url_host = urlsplit(url).netloc.lower()
    if "dji.com" in url_host:
        fields["brand"] = "DJI"
    heading_match = re.search(r"^#\s*(.+)$", markdown, flags=re.MULTILINE)
    page_title = heading_match.group(1).strip() if heading_match else None
    h1_match = re.search(r"^#+\s*(.+)$", markdown, flags=re.MULTILINE)
    heading_candidate = h1_match.group(1).strip() if h1_match else None
    path_parts = [part for part in urlsplit(url).path.split("/") if part]
    slug = path_parts[-1] if path_parts else ""
    normalized_slug = slug.replace("-", " ").replace("_", " ").strip()
    slug_candidate = " ".join(word.upper() if word.isupper() else word.capitalize() for word in normalized_slug.split())

    brand_prefix = str(fields.get("brand", "") or "")
    for candidate in (page_title, heading_candidate, slug_candidate):
        model_candidate = _clean_model_candidate(candidate or "", brand_prefix=brand_prefix)
        if model_candidate:
            fields.setdefault("model", model_candidate)
            break
    fields["category"] = _maybe_category(fields, markdown)
    fields.setdefault("link", url)


def _grade_quality(canonical_fields: Dict[str, object], raw_specs: Dict[str, Dict[str, str]], canonical_count: int) -> str:
    brand_valid = bool(canonical_fields.get("brand"))
    model_valid = bool(canonical_fields.get("model"))
    raw_present = bool(raw_specs)
    # Exclude identity/link/category from canonical count for grading.
    canonical_metrics = {
        key: value
        for key, value in canonical_fields.items()
        if key not in {"brand", "model", "category", "link"} and value not in (None, "")
    }
    metric_count = len(canonical_metrics)
    if brand_valid and model_valid and metric_count >= 3:
        return "A"
    if brand_valid and model_valid and metric_count >= 1:
        return "B"
    if model_valid and raw_present:
        return "C"
    if not model_valid and not raw_present:
        return "D"
    return "C"


def deterministic_parser_factory(
    llm_mapper: Optional[Callable[[Dict[str, object], str, str], Dict[str, object]]] = None,
) -> Callable[[str, str], str]:
    def parser(prompt: str, url: str) -> str:
        markdown = _extract_markdown(prompt)
        spec_lines, section_info = _select_spec_section(markdown)
        tables = _parse_table(spec_lines)
        table_entries = _collapse_tables(tables)
        kv_entries = _parse_key_values(spec_lines)
        entries = table_entries + kv_entries
        raw_specs = _collect_raw_specs(entries)
        fields, unmapped, mapped_from = _map_fields(raw_specs)
        _derive_identity(fields, markdown, url)
        if llm_mapper:
            try:
                mapped = llm_mapper(fields, markdown, url)
                if isinstance(mapped, dict):
                    fields.update({k: v for k, v in mapped.items() if v is not None})
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM mapper failed; continuing with deterministic fields", exc_info=exc)
        drone = _to_schema(fields)
        canonical_fields = drone.dict()
        canonical_count = sum(1 for key, value in canonical_fields.items() if value not in (None, ""))
        quality = _grade_quality(canonical_fields, raw_specs, canonical_count)
        metadata = {
            "unmapped_specs": unmapped,
            "mapped_fields": sorted([key for key, value in fields.items() if value is not None]),
            "mapped_from": mapped_from,
            "raw_specs_count": len(raw_specs),
            "canonical_field_count": canonical_count,
            "quality": quality,
            **section_info,
        }
        payload = {
            "canonical": canonical_fields,
            "data": canonical_fields,  # backward compatibility alias
            "raw_specs": raw_specs,
            "metadata": metadata,
        }
        return json.dumps(payload)

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
