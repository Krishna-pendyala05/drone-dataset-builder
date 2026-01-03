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
    drone = CameraDrone(**result["data"])

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

    assert result["data"]["brand"] == "LLMBrand"
    assert result["data"]["model"] == "Bar"


def test_deterministic_parser_normalizes_empty_numeric_fields() -> None:
    markdown = "- Brand: SkyCam\n- Model: EmptySpeed\n- Max speed: —\n"
    parser = deterministic_parser_factory()

    result = json.loads(parser(markdown, "https://example.com/empty-speed"))
    drone = CameraDrone(**result["data"])

    assert drone.max_speed is None


def test_deterministic_parser_ignores_nav_noise_and_urls() -> None:
    markdown = (FIXTURE_DIR / "dji_noise.md").read_text(encoding="utf-8")
    parser = deterministic_parser_factory()

    result = json.loads(parser(markdown, "https://www.dji.com/global/mini-4-pro"))
    data = result["data"]

    assert data["brand"] == "DJI"
    assert data["model"] == "Mini 4 Pro"
    assert data["max_flight_time"] == 34
    assert data["max_speed"] == 21
    assert data["sensor_type"] == '1/1.3" CMOS'
    assert all("http" not in str(value) for key, value in data.items() if value and key != "link")
