"""
One-shot CLI runner for the Trip Concierge Agent.

    python run_agent.py
    python run_agent.py --origin Paris --budget 500 --nights 3
    python run_agent.py --goal "Plan a 3-day trip to Porto under €600 ..."
    python run_agent.py --step-limit 1     # deliberately trip the step limit

What this script demonstrates, end to end:
  - the agent deciding its own tool calls for a multi-step goal (printed
    live as they happen, below)
  - a hard step limit (RunConfig.max_llm_calls) so a stuck or adversarial
    run can't loop forever or rack up unbounded API cost
  - graceful handling of: the step limit being hit, a tool returning an
    error, and the model's final text not matching the expected JSON shape
  - a final, schema-validated, machine-parseable result on stdout (and, on
    success, written to trip_plan_result.json)

Requires a Gemini API key (Google AI Studio or Vertex AI) — see
README.md -> "Setup" and .env.example. Without one, model calls will fail
with an authentication error, which this script also reports gracefully
rather than crashing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()  # picks up GOOGLE_API_KEY etc. from a local .env file
except ImportError:
    pass  # python-dotenv is optional; env vars can also be exported directly

from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import ValidationError

from trip_concierge.agent import root_agent
from trip_concierge.schema import TripPlanResult

APP_NAME = "trip_concierge_app"
USER_ID = "local_user"
SESSION_ID = "local_session"

# --- Reliability: the step limit -------------------------------------------
# A realistic run for this goal is: think -> search_flights -> think ->
# search_hotels -> think -> calculate (hotel total) -> think -> calculate
# (grand total) -> final answer. That's 5 LLM calls. STEP_LIMIT gives some
# headroom above that (e.g. for a self-correction) without allowing genuine
# runaway tool-calling loops or unbounded spend. ADK's own default
# (max_llm_calls=500) is far too generous to count as a real bound for an
# assessment like this one, so we override it explicitly.
DEFAULT_STEP_LIMIT = 8

DEFAULT_GOAL_TEMPLATE = (
    "I'm flying out of {origin}. Plan a {nights_plus_one}-day trip "
    "({nights} nights) to Porto under €{budget} total, covering round-trip "
    "flights and the hotel, and give me a full cost breakdown."
)


def build_goal(origin: str, budget: float, nights: int) -> str:
    return DEFAULT_GOAL_TEMPLATE.format(
        origin=origin, nights_plus_one=nights + 1, nights=nights, budget=budget
    )


def _print_event(event) -> None:
    """Renders one ADK event as a readable transcript line. This is what
    makes the agent's own tool choices visible, rather than just printing
    the final answer."""
    if not event.content or not event.content.parts:
        return
    for part in event.content.parts:
        if part.function_call:
            args = json.dumps(part.function_call.args)
            print(f"  -> [tool call]   {part.function_call.name}({args})")
        elif part.function_response:
            resp = json.dumps(part.function_response.response)
            print(f"  <- [tool result] {part.function_response.name} -> {resp}")
        elif part.text and part.text.strip():
            label = "[final answer]" if event.is_final_response() else "[model]"
            print(f"  {label} {part.text.strip()}")


def _extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from the model's final text,
    tolerating accidental markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


async def run(goal: str, step_limit: int) -> dict:
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    runner = Runner(
        agent=root_agent, app_name=APP_NAME, session_service=session_service
    )
    run_config = RunConfig(max_llm_calls=step_limit)
    content = types.Content(role="user", parts=[types.Part(text=goal)])

    print(f"--- Goal ---\n{goal}\n")
    print(f"--- Run (step limit = {step_limit} LLM calls) ---")

    final_text = None
    try:
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=content,
            run_config=run_config,
        ):
            _print_event(event)
            if event.is_final_response() and event.content and event.content.parts:
                text_parts = [p.text for p in event.content.parts if p.text]
                if text_parts:
                    final_text = "".join(text_parts)
    except LlmCallsLimitExceededError as exc:
        # --- Reliability: graceful handling of the step limit -------------
        return {
            "status": "error",
            "error_message": (
                f"Step limit ({step_limit} LLM calls) reached before the "
                f"agent finished planning the trip. Try again with a "
                f"higher --step-limit, or check the transcript above for a "
                f"loop. ({exc})"
            ),
        }
    except Exception as exc:  # e.g. missing/invalid API key, network error
        return {
            "status": "error",
            "error_message": f"The agent run failed before completing: {exc}",
        }

    if final_text is None:
        return {
            "status": "error",
            "error_message": "The agent finished without producing a final text response.",
        }

    # --- Reliability: graceful handling of a malformed final answer -------
    try:
        parsed = _extract_json(final_text)
        validated = TripPlanResult.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError) as exc:
        return {
            "status": "error",
            "error_message": f"Agent's final answer did not match the expected JSON shape: {exc}",
            "raw_final_text": final_text,
        }

    return {"status": "success", "result": validated.model_dump()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Trip Concierge Agent once.")
    parser.add_argument("--origin", default="London", help="Departure city (default: London)")
    parser.add_argument("--budget", type=float, default=600, help="Total budget in EUR (default: 600)")
    parser.add_argument("--nights", type=int, default=2, help="Nights in Porto (default: 2, i.e. a 3-day trip)")
    parser.add_argument("--goal", default=None, help="Override the goal text entirely")
    parser.add_argument("--step-limit", type=int, default=DEFAULT_STEP_LIMIT, help="Max LLM calls allowed for the run")
    args = parser.parse_args()

    goal = args.goal or build_goal(args.origin, args.budget, args.nights)
    outcome = asyncio.run(run(goal, args.step_limit))

    print("\n--- Structured result ---")
    print(json.dumps(outcome, indent=2))

    if outcome["status"] == "success":
        with open("trip_plan_result.json", "w", encoding="utf-8") as f:
            json.dump(outcome["result"], f, indent=2)
        print("\nSaved to trip_plan_result.json")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
