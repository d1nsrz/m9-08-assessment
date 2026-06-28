"""
Unit tests for the tool layer — no LLM, no network, no API key required.

Run with:  pytest tests/ -v

These exist for two reasons:
1. Prove the tools behave correctly (happy path + graceful error path) in
   isolation, which is most of what the "reliability" requirement is about.
2. Prove the safety mitigation in calculate() actually rejects the kind of
   input it claims to defend against, rather than just asserting it in the
   README.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from trip_concierge.tools import calculate, search_flights, search_hotels


# --------------------------------------------------------------------------
# search_flights
# --------------------------------------------------------------------------

def test_search_flights_happy_path_round_trip():
    result = search_flights("London", "Porto", "round_trip")
    assert result["status"] == "success"
    assert result["route"] == "LHR-OPO"
    cheapest = min(result["options"], key=lambda o: o["estimated_total_price_eur"])
    assert cheapest["airline"] == "Ryanair"
    assert cheapest["estimated_total_price_eur"] == 79 * 2


def test_search_flights_resolves_aliases_and_is_case_insensitive():
    a = search_flights("london heathrow", "PORTO")
    b = search_flights("LONDON", "porto")
    assert a["status"] == "success"
    assert b["status"] == "success"
    assert a["route"] == b["route"] == "LHR-OPO"


def test_search_flights_one_way_is_not_doubled():
    result = search_flights("Madrid", "Porto", "one_way")
    assert result["status"] == "success"
    opt = result["options"][0]
    assert opt["estimated_total_price_eur"] == opt["one_way_price_eur"]


def test_search_flights_unknown_origin_fails_gracefully():
    result = search_flights("Atlantis", "Porto")
    assert result["status"] == "error"
    assert "Atlantis" in result["error_message"]


def test_search_flights_unknown_route_fails_gracefully():
    # Both cities are real, but there's no Berlin -> Porto route in the data.
    result = search_flights("Berlin", "Porto")
    assert result["status"] == "error"
    assert "BER-OPO" in result["error_message"]


def test_search_flights_rejects_bad_trip_type():
    result = search_flights("London", "Porto", trip_type="business_class_only")
    assert result["status"] == "error"


# --------------------------------------------------------------------------
# search_hotels
# --------------------------------------------------------------------------

def test_search_hotels_happy_path_sorted_cheapest_first():
    result = search_hotels("Porto")
    assert result["status"] == "success"
    prices = [h["price_per_night"] for h in result["hotels"]]
    assert prices == sorted(prices)
    assert prices[0] == 28  # YES Porto Hostel


def test_search_hotels_budget_filter():
    result = search_hotels("Porto", max_price_per_night=90)
    assert result["status"] == "success"
    assert all(h["price_per_night"] <= 90 for h in result["hotels"])


def test_search_hotels_budget_filter_with_no_matches_fails_gracefully():
    result = search_hotels("Porto", max_price_per_night=1)
    assert result["status"] == "error"


def test_search_hotels_unknown_destination_fails_gracefully():
    result = search_hotels("Narnia")
    assert result["status"] == "error"
    assert "Narnia" in result["error_message"]


# --------------------------------------------------------------------------
# calculate — correctness
# --------------------------------------------------------------------------

def test_calculate_basic_arithmetic():
    assert calculate("149 + 95 * 2")["result"] == 149 + 95 * 2


def test_calculate_parentheses_and_division():
    assert calculate("(149 + 79) / 2")["result"] == (149 + 79) / 2


def test_calculate_negative_numbers():
    assert calculate("-5 + 10")["result"] == 5


def test_calculate_division_by_zero_fails_gracefully():
    result = calculate("1 / 0")
    assert result["status"] == "error"
    assert "zero" in result["error_message"].lower()


def test_calculate_empty_string_fails_gracefully():
    result = calculate("")
    assert result["status"] == "error"


# --------------------------------------------------------------------------
# calculate — SAFETY: must reject anything that isn't plain arithmetic.
# This is the core proof for the README's safety mitigation section.
# --------------------------------------------------------------------------

DANGEROUS_INPUTS = [
    "__import__('os').system('echo pwned')",
    "open('/etc/passwd').read()",
    "[x for x in ().__class__.__bases__[0].__subclasses__()]",
    "exec('import os')",
    "1 if True else 0",
    "(lambda: 1)()",
    "os.system('echo pwned')",
    "'a' + 'b'",
]


def test_calculate_rejects_code_injection_attempts():
    for payload in DANGEROUS_INPUTS:
        result = calculate(payload)
        assert result["status"] == "error", f"DID NOT REJECT: {payload!r}"
        assert "result" not in result


def test_calculate_does_not_use_eval_or_exec():
    # Static guard so nobody "fixes" a bug later by swapping _safe_eval for
    # eval()/exec(). Walks the real AST of tools.py (not the docstrings, so
    # the word "eval" appearing in a comment doesn't trigger a false alarm)
    # looking for an actual call to the builtin eval() or exec().
    import ast
    import inspect

    from trip_concierge import tools

    source = inspect.getsource(tools)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("eval", "exec"), (
                f"calculate() must never call builtin {node.func.id}()"
            )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
