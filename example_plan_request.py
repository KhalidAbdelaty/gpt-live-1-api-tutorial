"""Standalone example: call the same backend GPT-Live-1 delegates to.

This hits the real Responses API with the exact model, instructions, and
tools configured in server.py's delegation.responses block, so you can see
a grounded learning plan without holding a microphone conversation. Useful
for checking the backend prompt and web_search domain filter on their own.

Run:
    python example_plan_request.py
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from server import BACKEND_INSTRUCTIONS, BACKEND_MODEL, SAVE_LEARNING_PLAN_TOOL

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = OpenAI()

TOOLS = [
    {"type": "web_search", "filters": {"allowed_domains": ["datacamp.com", "www.datacamp.com"]}},
    SAVE_LEARNING_PLAN_TOOL,
]


def main() -> None:
    goal = client.responses.create(
        model=BACKEND_MODEL,
        instructions=BACKEND_INSTRUCTIONS,
        tools=TOOLS,
        tool_choice="auto",
        input="I want to become a data engineer, but I only have five hours per week. I know a little Python and no SQL.",
    )
    print("=== initial plan ===")
    print(goal.output_text)

    constraint = client.responses.create(
        model=BACKEND_MODEL,
        instructions=BACKEND_INSTRUCTIONS,
        previous_response_id=goal.id,
        tools=TOOLS,
        tool_choice="auto",
        input="Actually, prioritize hands-on projects, and skip beginner Python since I already know it.",
    )
    print("\n=== revised plan after a new constraint ===")
    print(constraint.output_text)

    save = client.responses.create(
        model=BACKEND_MODEL,
        instructions=BACKEND_INSTRUCTIONS,
        previous_response_id=constraint.id,
        tools=TOOLS,
        tool_choice="auto",
        input="This looks good, please save this plan.",
    )
    print("\n=== function call after a save request ===")
    for item in save.output:
        if item.type == "function_call":
            print(item.name, json.dumps(json.loads(item.arguments), indent=2))


if __name__ == "__main__":
    main()
