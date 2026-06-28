"""
Tool implementations for the Trip Concierge Agent.

Design rule used throughout: tools never raise exceptions for *expected*
failure modes (unknown city, no matching route, bad expression, etc). They
catch those cases and return a plain dict with status="error" plus a
human-readable error_message instead. That gives the LLM something to react
to ("no flights found, try a different origin") instead of crashing the run.
See README.md -> "Reliability" for why this matters.

calculate() is also the home of this project's safety mitigation: it never
uses eval()/exec(). See the comment block above _safe_eval for details, and
README.md -> "Safety" for the threat it defends against.
"""

from __future__ import annotations

import ast
import json
import operator
import pathlib
from typing import Optional

_DATA_DIR = pathlib.Path(__file__).parent / "data"

with open(_DATA_DIR / "flights.json", encoding="utf-8") as f:
    _FLIGHTS = json.load(f)

with open(_DATA_DIR / "hotels.json", encoding="utf-8") as f:
    _HOTELS = json.load(f)

_ALIASES = _FLIGHTS["aliases"]


def _resolve_city(name: str) -> Optional[str]:
    """Normalises a free-text city/airport name to an IATA code via the
    fixed alias table in flights.json. Returns None if it isn't recognised
    — callers must treat that as a validation failure, not guess a code.
    """
    if not isinstance(name, str) or not name.strip():
        return None
    key = name.strip().upper().replace(" ", "")
    return _ALIASES.get(key)


def search_flights(origin: str, destination: str, trip_type: str = "round_trip") -> dict:
    """Looks up available flights between two cities.

    Args:
        origin: Departure city or airport name (e.g. "London", "Paris", "LHR").
        destination: Arrival city or airport name (e.g. "Porto", "OPO").
        trip_type: "one_way" or "round_trip" (default "round_trip"). Round-trip
            price is estimated as 2x the one-way fare, since the mock dataset
            only models inbound legs to Porto (no separate return-leg pricing).

    Returns:
        On success: {"status": "success", "route": "LHR-OPO", "options": [...]}
        where each option has airline, flight_number, duration,
        one_way_price_eur, trip_type, estimated_total_price_eur.
        On failure: {"status": "error", "error_message": "..."}.
    """
    if trip_type not in ("one_way", "round_trip"):
        return {
            "status": "error",
            "error_message": f"Unknown trip_type '{trip_type}'. Use 'one_way' or 'round_trip'.",
        }

    origin_code = _resolve_city(origin)
    dest_code = _resolve_city(destination)

    if origin_code is None:
        return {
            "status": "error",
            "error_message": f"Unknown origin city '{origin}'. Known cities: {sorted(_ALIASES)}",
        }
    if dest_code is None:
        return {
            "status": "error",
            "error_message": f"Unknown destination city '{destination}'. Known cities: {sorted(_ALIASES)}",
        }

    route_key = f"{origin_code}-{dest_code}"
    options = _FLIGHTS["routes"].get(route_key)
    if not options:
        return {
            "status": "error",
            "error_message": f"No flights found for route {route_key} in the mock dataset.",
        }

    multiplier = 2 if trip_type == "round_trip" else 1
    enriched = [
        {
            "airline": opt["airline"],
            "flight_number": opt["flight_number"],
            "duration": opt["duration"],
            "one_way_price_eur": opt["price"],
            "trip_type": trip_type,
            "estimated_total_price_eur": opt["price"] * multiplier,
        }
        for opt in options
    ]
    return {"status": "success", "route": route_key, "options": enriched}


def search_hotels(destination: str, max_price_per_night: Optional[float] = None) -> dict:
    """Looks up hotels available in a destination city.

    Args:
        destination: City to search (e.g. "Porto"). Case-insensitive.
        max_price_per_night: Optional filter (EUR) — only hotels at or below
            this nightly price are returned. Omit to see every option.

    Returns:
        On success: {"status": "success", "destination": "porto",
        "hotels": [...]}, sorted cheapest-first, each with name,
        price_per_night, rating, category, amenities, location.
        On failure: {"status": "error", "error_message": "..."}.
    """
    if not isinstance(destination, str) or not destination.strip():
        return {"status": "error", "error_message": "destination must be a non-empty string."}

    key = destination.strip().lower()
    hotels = _HOTELS["destinations"].get(key)
    if not hotels:
        return {
            "status": "error",
            "error_message": f"No hotel data for destination '{destination}'. "
            f"Known destinations: {sorted(_HOTELS['destinations'])}",
        }

    if max_price_per_night is not None:
        if not isinstance(max_price_per_night, (int, float)) or max_price_per_night < 0:
            return {
                "status": "error",
                "error_message": "max_price_per_night must be a non-negative number.",
            }
        hotels = [h for h in hotels if h["price_per_night"] <= max_price_per_night]
        if not hotels:
            return {
                "status": "error",
                "error_message": f"No hotels in '{destination}' at or under "
                f"€{max_price_per_night}/night.",
            }

    sorted_hotels = sorted(hotels, key=lambda h: h["price_per_night"])
    return {"status": "success", "destination": key, "hotels": sorted_hotels}


# ---------------------------------------------------------------------------
# SAFETY MITIGATION
#
# A naive "calculator tool" is usually implemented as `eval(expression)`.
# That is a code-injection hole: anything that can influence the
# `expression` argument — a misbehaving model, or text smuggled in through a
# tool RESULT that later gets echoed back as a tool ARGUMENT — can run
# arbitrary Python ("__import__('os').system(...)", reading env vars/secrets,
# etc), not just arithmetic.
#
# calculate() never calls eval()/exec(). It parses the expression into an
# AST and walks it with _safe_eval, which only recurses into a fixed
# whitelist of node types: numeric constants, the basic binary/unary math
# operators, and (implicitly) parentheses. Anything else — names, function
# calls, attribute access, subscripts, comprehensions, string literals,
# imports — hits the `else: raise ValueError` branch and the tool returns a
# clean status="error" instead of executing anything.
#
# See README.md -> "Safety mitigation" for the full justification and the
# unit tests in tests/test_tools.py for proof-of-rejection.
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"disallowed constant: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"disallowed expression element: {type(node).__name__}")


def calculate(expression: str) -> dict:
    """Safely evaluates a basic arithmetic expression. This is the ONLY
    tool allowed to do arithmetic — never compute totals by hand.

    Supports +, -, *, /, %, ** and parentheses over plain numbers, e.g.
    "149 + 95 * 2" or "(149 + 79) / 2". Nothing else is accepted: no
    variables, function calls, or text of any kind.

    Args:
        expression: An arithmetic expression as a string.

    Returns:
        On success: {"status": "success", "expression": "...", "result": 343}.
        On failure (division by zero, or anything that isn't plain
        arithmetic): {"status": "error", "error_message": "..."}.
    """
    if not isinstance(expression, str) or not expression.strip():
        return {"status": "error", "error_message": "expression must be a non-empty string."}
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _safe_eval(parsed)
    except ZeroDivisionError:
        return {"status": "error", "error_message": "Division by zero."}
    except (SyntaxError, ValueError) as exc:
        return {
            "status": "error",
            "error_message": f"Invalid or unsafe expression: {exc}",
        }
    return {"status": "success", "expression": expression, "result": result}
