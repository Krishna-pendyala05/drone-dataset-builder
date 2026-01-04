from __future__ import annotations

import json
from pathlib import Path

from agents.parser import deterministic_parser_factory
from config.schemas import CameraDrone


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_deterministic_parser_extracts_table_fields() -> None:
    markdown = (FIXTURE_DIR / "camera.md").read_text(encoding="utf-8")
    parser = deterministic_parser_factory()

    result = json.loads(parser(markdown, "https://example.com/camera"))
    drone = CameraDrone(**result["canonical"])

    assert drone.brand == "SkyCam"
    assert drone.model == "Aero X"
    assert drone.max_flight_time == 31
    assert drone.sensor_type == '1" CMOS'
    assert drone.category == "camera"


def test_deterministic_parser_respects_llm_mapper_override() -> None:
    markdown = "- Brand: Foo\n- Model: Bar\n"

    def mapper(fields, markdown_payload, url):  # noqa: ANN001
        assert "foo" in markdown_payload.lower()
        fields["brand"] = "LLMBrand"
        return fields

    parser = deterministic_parser_factory(llm_mapper=mapper)

    result = json.loads(parser(markdown, "https://example.com/override"))

    assert result["canonical"]["brand"] == "LLMBrand"
    assert result["canonical"]["model"] == "Bar"


def test_deterministic_parser_normalizes_empty_numeric_fields() -> None:
    markdown = "- Brand: SkyCam\n- Model: EmptySpeed\n- Max speed: —\n"
    parser = deterministic_parser_factory()

    result = json.loads(parser(markdown, "https://example.com/empty-speed"))
    drone = CameraDrone(**result["canonical"])

    assert drone.max_speed is None


def test_deterministic_parser_ignores_nav_noise_and_urls() -> None:
    markdown = (FIXTURE_DIR / "dji_noise.md").read_text(encoding="utf-8")
    parser = deterministic_parser_factory()

    result = json.loads(parser(markdown, "https://www.dji.com/global/mini-4-pro"))
    data = result["canonical"]

    assert data["brand"] == "DJI"
    assert data["model"] == "Mini 4 Pro"
    assert data["max_flight_time"] == 34
    assert data["max_speed"] == 21
    assert data["sensor_type"] == '1/1.3" CMOS'
    assert all("http" not in str(value) for key, value in data.items() if value and key != "link")


def test_parser_emits_raw_specs_and_quality_grades() -> None:
    markdown = (FIXTURE_DIR / "dji_spec.md").read_text(encoding="utf-8")
    parser = deterministic_parser_factory()

    result = json.loads(parser(markdown, "https://www.dji.com/air-3"))

    raw_specs = result["raw_specs"]
    canonical = result["canonical"]
    metadata = result["metadata"]

    assert raw_specs
    assert "max flight time" in raw_specs
    assert "image sensor" in raw_specs
    assert canonical["model"] == "Air 3"
    assert "Customer Service" not in canonical["model"]
    assert canonical["brand"] == "DJI"
    assert canonical["max_flight_time"] == 46
    assert canonical["max_speed"] == 21
    assert "CMOS" in canonical["sensor_type"]
    assert metadata["quality"] == "A"
    assert metadata["mapped_from"]["max_speed"] == "max speed"


def test_parser_requires_units_for_core_metrics() -> None:
    markdown = (FIXTURE_DIR / "dji_unitless.md").read_text(encoding="utf-8")
    parser = deterministic_parser_factory()

    result = json.loads(parser(markdown, "https://www.dji.com/nova"))
    canonical = result["canonical"]
    raw_specs = result["raw_specs"]

    assert canonical["brand"] == "DJI"
    assert canonical["model"] == "Nova"
    assert canonical["max_flight_time"] is None
    assert canonical["max_speed"] is None
    assert "max speed" in raw_specs
    assert result["metadata"]["quality"] == "B"


def test_parser_extracts_raw_specs_from_shadow_text() -> None:
    spec_text = (FIXTURE_DIR / "dji_shadow.txt").read_text(encoding="utf-8")
    parser = deterministic_parser_factory()

    payload = {
        "markdown": "",
        "full_html": "",
        "pruned_html": "",
        "spec_text": spec_text,
        "title": "DJI Air 3 - Specs | DJI",
    }

    result = json.loads(parser(payload, "https://www.dji.com/air-3"))
    raw_specs = result["raw_specs"]
    canonical = result["canonical"]

    assert raw_specs
    assert canonical["brand"] == "DJI"
    assert canonical["model"] == "Air 3"
    assert canonical["max_flight_time"] == 46
    assert canonical["max_speed"] == 21
    assert canonical["sensor_type"]
    assert result["metadata"]["mapped_from"]["max_flight_time"] == "max flight time"


def test_model_extraction_falls_back_from_support_titles() -> None:
    markdown = "# Support | DJI\n"
    parser = deterministic_parser_factory()

    payload = {
        "markdown": markdown,
        "full_html": "",
        "pruned_html": "",
        "spec_text": "",
        "title": "Support - DJI",
    }

    result = json.loads(parser(payload, "https://www.dji.com/support/mini-4-pro"))
    canonical = result["canonical"]

    assert canonical["model"] == "Mini 4 Pro"
    assert "Support" not in canonical["model"]
