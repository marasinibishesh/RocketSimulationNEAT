"""Episode telemetry capture and CSV export."""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TelemetryRow:
    time_s: float
    x_m: float
    y_m: float
    altitude_m: float
    vx_mps: float
    vy_mps: float
    vz_mps: float
    throttle: float
    gimbal_pitch: float
    gimbal_yaw: float
    tilt_deg: float
    fuel_kg: float
    reward: float


class TelemetryLog:
    def __init__(self) -> None:
        self.rows: list[TelemetryRow] = []

    def append(self, row: TelemetryRow) -> None:
        self.rows.append(row)

    def write_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            path.write_text("", encoding="utf-8")
            return path
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(asdict(self.rows[0]).keys()))
            writer.writeheader()
            for row in self.rows:
                writer.writerow(asdict(row))
        return path
