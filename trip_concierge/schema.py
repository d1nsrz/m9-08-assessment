"""
Pydantic schema for the agent's final answer.

This is deliberately NOT wired up as ADK's `output_schema=` on the agent
itself. ADK's docs note that an LlmAgent with output_schema set is expected
to answer directly with no tools and no sub-agents — combining tools +
output_schema on one agent isn't reliably supported across models. Since
this agent's whole point is calling three tools, we instead:

  1. Instruct the model (in agent.py) to emit ONE raw JSON object as its
     final text response, matching this schema.
  2. Parse + validate that text against TripPlanResult here, in run_agent.py,
     after the run finishes.

That keeps tool-calling and structured output decoupled, and gives us a
clean validation failure to handle gracefully if the model ever drifts from
the requested shape (see run_agent.py's exception handling).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class FlightChoice(BaseModel):
    airline: str
    flight_number: str
    price_eur: float


class HotelChoice(BaseModel):
    name: str
    price_per_night_eur: float
    rating: float


class CostBreakdown(BaseModel):
    flight_total_eur: float
    hotel_total_eur: float
    grand_total_eur: float


class TripPlanResult(BaseModel):
    feasible: bool
    destination: str
    nights: int
    flight: Optional[FlightChoice] = None
    hotel: Optional[HotelChoice] = None
    cost_breakdown: Optional[CostBreakdown] = None
    budget_eur: float
    notes: str
