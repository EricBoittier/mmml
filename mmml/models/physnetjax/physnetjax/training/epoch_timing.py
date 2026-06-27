"""Epoch timing breakdown for PhysNetJax training."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpochTiming:
    """Wall-clock seconds for one training epoch."""

    batch_prep_s: float = 0.0
    train_s: float = 0.0
    valid_s: float = 0.0
    checkpoint_s: float = 0.0
    other_s: float = 0.0

    @property
    def total_s(self) -> float:
        return (
            self.batch_prep_s
            + self.train_s
            + self.valid_s
            + self.checkpoint_s
            + self.other_s
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "batch_prep_s": self.batch_prep_s,
            "train_s": self.train_s,
            "valid_s": self.valid_s,
            "checkpoint_s": self.checkpoint_s,
            "other_s": self.other_s,
            "total_s": self.total_s,
        }


@dataclass
class EpochTimingSummary:
    """Rolling epoch timing stats."""

    epochs: int = 0
    batch_prep_s: float = 0.0
    train_s: float = 0.0
    valid_s: float = 0.0
    checkpoint_s: float = 0.0
    other_s: float = 0.0

    def record(self, timing: EpochTiming) -> None:
        self.epochs += 1
        self.batch_prep_s += timing.batch_prep_s
        self.train_s += timing.train_s
        self.valid_s += timing.valid_s
        self.checkpoint_s += timing.checkpoint_s
        self.other_s += timing.other_s

    def means(self) -> dict[str, float]:
        if self.epochs <= 0:
            return {}
        n = float(self.epochs)
        return {
            "batch_prep_s": self.batch_prep_s / n,
            "train_s": self.train_s / n,
            "valid_s": self.valid_s / n,
            "checkpoint_s": self.checkpoint_s / n,
            "other_s": self.other_s / n,
            "total_s": (self.batch_prep_s + self.train_s + self.valid_s + self.checkpoint_s + self.other_s) / n,
        }

    def format_means(self) -> str:
        m = self.means()
        if not m:
            return ""
        return (
            f"avg epoch {m['total_s']:.2f}s "
            f"(batch_prep={m['batch_prep_s']:.2f}s, "
            f"train={m['train_s']:.2f}s, valid={m['valid_s']:.2f}s, "
            f"ckpt={m['checkpoint_s']:.2f}s)"
        )
