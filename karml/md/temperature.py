"""Backend-independent temperature schedules."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TemperatureStage", "TemperatureSchedule", "parse_temperature_schedule"]


@dataclass(frozen=True)
class TemperatureStage:
    start_K: float
    stop_K: float
    fraction: float

    def __post_init__(self) -> None:
        if self.start_K <= 0 or self.stop_K <= 0:
            raise ValueError("temperature schedule values must be positive")
        if self.fraction <= 0:
            raise ValueError("temperature stage fraction must be positive")


@dataclass(frozen=True)
class TemperatureSchedule:
    stages: tuple[TemperatureStage, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported temperature schedule schema version")
        if not self.stages:
            raise ValueError("temperature schedule needs at least one stage")
        if abs(sum(s.fraction for s in self.stages) - 1.0) > 1e-9:
            raise ValueError("temperature stage fractions must sum to 1")

    def temperature_at(self, step: int, total_steps: int) -> float:
        if total_steps < 0 or step < 0:
            raise ValueError("step counts must be non-negative")
        progress = 1.0 if total_steps == 0 else min(step / total_steps, 1.0)
        lower = 0.0
        for stage in self.stages:
            upper = lower + stage.fraction
            if progress <= upper or stage is self.stages[-1]:
                local = min(max((progress - lower) / stage.fraction, 0.0), 1.0)
                return stage.start_K + local * (stage.stop_K - stage.start_K)
            lower = upper
        raise AssertionError("unreachable")


def parse_temperature_schedule(text: str) -> TemperatureSchedule:
    """Parse ``300`` or ``200->300:0.25,300:0.75``."""

    parts = [part.strip() for part in text.split(",") if part.strip()]
    explicit = any(":" in part for part in parts)
    stages = []
    for part in parts:
        value, sep, frac = part.partition(":")
        if "->" in value:
            start, stop = (float(x.strip()) for x in value.split("->", 1))
        else:
            start = stop = float(value)
        fraction = float(frac) if sep else (1.0 / len(parts) if len(parts) > 1 else 1.0)
        stages.append(TemperatureStage(start, stop, fraction))
    if explicit and any(":" not in part for part in parts):
        raise ValueError("either all temperature stages specify fractions or none do")
    return TemperatureSchedule(tuple(stages))
