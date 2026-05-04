from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Uniform interface for optional report-generation providers."""

    name = "base"

    @abstractmethod
    def generate_mission_report(self, mission_result: dict[str, Any]) -> dict[str, Any]:
        """Return a structured mission report draft for an existing result payload."""
