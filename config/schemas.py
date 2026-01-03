from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, validator


class DroneBase(BaseModel):
    """Normalized drone specification shared across categories."""

    brand: Optional[str] = Field(None, description="Manufacturer or OEM name.")
    model: Optional[str] = Field(None, description="Model or product identifier.")
    category: Literal["camera", "fpv", "enterprise"] = Field(
        "camera", description="Primary drone category."
    )
    max_flight_time: Optional[float] = Field(
        None,
        description="Maximum flight or endurance time in minutes. Map aliases like 'Endurance'.",
    )
    sensor_type: Optional[str] = Field(
        None,
        description="Primary sensor (e.g., 1\" CMOS, thermal, LiDAR).",
    )
    payload_capacity: Optional[float] = Field(
        None, description="Payload capacity in kilograms."
    )
    max_speed: Optional[float] = Field(
        None, description="Maximum speed in meters per second."
    )
    link: Optional[str] = Field(
        None, description="Canonical URL used for the extraction run."
    )
    notes: Optional[str] = Field(
        None, description="Free-form notes captured from the page."
    )

    @validator("brand", "model", pre=True)
    def _strip_text(cls, value: Optional[str]) -> Optional[str]:
        if isinstance(value, str):
            return value.strip() or None
        return value

    class Config:
        anystr_strip_whitespace = True
        validate_assignment = True


class CameraDrone(DroneBase):
    category: Literal["camera"] = "camera"
    camera_resolution: Optional[str] = Field(
        None, description="Primary capture resolution (e.g., 5.1K, 4K)."
    )
    gimbal: Optional[bool] = Field(
        None, description="Whether the platform ships with a stabilized gimbal."
    )


class FPVDrone(DroneBase):
    category: Literal["fpv"] = "fpv"
    video_tx_power_mw: Optional[int] = Field(
        None, description="Maximum video transmitter output in milliwatts."
    )
    supports_hd_link: Optional[bool] = Field(
        None, description="Whether the FPV system supports an HD digital link."
    )


class EnterpriseDrone(DroneBase):
    category: Literal["enterprise"] = "enterprise"
    ingress_protection: Optional[str] = Field(
        None, description="Ingress protection rating (e.g., IP43)."
    )
    thermal_capability: Optional[bool] = Field(
        None, description="Whether a thermal sensor is integrated or supported."
    )
    operating_temp_c: Optional[str] = Field(
        None, description="Operating temperature range in °C."
    )

