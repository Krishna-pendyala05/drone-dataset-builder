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
    drone = CameraDrone(**result)

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

    assert result["brand"] == "LLMBrand"
    assert result["model"] == "Bar"


def test_deterministic_parser_normalizes_empty_numeric_fields() -> None:
    markdown = "- Brand: SkyCam\n- Model: EmptySpeed\n- Max speed: —\n"
    parser = deterministic_parser_factory()

    result = json.loads(parser(markdown, "https://example.com/empty-speed"))
    drone = CameraDrone(**result)

    assert drone.max_speed is None
