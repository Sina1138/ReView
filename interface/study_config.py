"""Study configuration for ReView evaluation conditions."""

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class StudyConfig:
    """Configuration for a study condition."""
    condition: str            # "full" or "no_highlight"
    logging_enabled: bool     # True for study variants
    log_dir: Path             # directory for JSONL interaction logs

    @property
    def highlights_enabled(self) -> bool:
        return self.condition == "full"


def default_config() -> StudyConfig:
    """Default config: full features, no logging (backward-compat Demo.py)."""
    return StudyConfig(
        condition="full",
        logging_enabled=False,
        log_dir=BASE_DIR / "study" / "interaction_logs",
    )


def full_study_config() -> StudyConfig:
    """Full condition with logging enabled."""
    return StudyConfig(
        condition="full",
        logging_enabled=True,
        log_dir=BASE_DIR / "study" / "interaction_logs",
    )


def no_highlight_study_config() -> StudyConfig:
    """No-highlight condition with logging enabled."""
    return StudyConfig(
        condition="no_highlight",
        logging_enabled=True,
        log_dir=BASE_DIR / "study" / "interaction_logs",
    )
