import json
import os
from typing import Any

import requests

from .base import LLMProvider
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .schemas import REPORT_JSON_SCHEMA, coerce_report


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout_seconds: int = 45):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.5")
        self.timeout_seconds = timeout_seconds

    def generate_mission_report(self, mission_result: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        payload = {
            "model": self.model,
            "instructions": SYSTEM_PROMPT,
            "input": build_user_prompt(mission_result),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": REPORT_JSON_SCHEMA["name"],
                    "strict": REPORT_JSON_SCHEMA["strict"],
                    "schema": REPORT_JSON_SCHEMA["schema"],
                }
            },
        }
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return coerce_report(json.loads(_extract_output_text(data)))


def _extract_output_text(response_payload: dict[str, Any]) -> str:
    if isinstance(response_payload.get("output_text"), str):
        return response_payload["output_text"]
    for item in response_payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                return text
    raise ValueError("OpenAI Responses API returned no text output.")
