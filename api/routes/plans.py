from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.common.config import CONFIG_DIR, load_yaml

router = APIRouter()


class PricingModel(BaseModel):
    mode: str
    base_price_weekly_inr: float | None = None
    note: str | None = None
    contact_sales: bool = False


class CoverageModel(BaseModel):
    events: list[str]
    zones_allowed: int | str
    compensation_cap_tier: str


class ClaimsModel(BaseModel):
    review_queue: str
    appeal_support: str
    green_instant_payout: bool


class PlanModel(BaseModel):
    plan_id: str
    display_name: str
    pricing: PricingModel
    coverage: CoverageModel
    claims: ClaimsModel


@lru_cache(maxsize=1)
def _load_plans() -> dict[str, Any]:
    payload = load_yaml(CONFIG_DIR / "pricing" / "plans.yaml")
    plans = payload.get("plans")
    if not isinstance(plans, dict):
        raise ValueError("Invalid plans.yaml: missing top-level 'plans' map")
    return plans


def _normalize_plan(plan_id: str, payload: dict[str, Any]) -> PlanModel:
    return PlanModel(plan_id=plan_id, **payload)


@router.get("/", response_model=list[PlanModel])
def list_plans() -> list[PlanModel]:
    plans = _load_plans()
    return [_normalize_plan(plan_id, payload) for plan_id, payload in plans.items()]


@router.get("/compare/table")
def compare_plans() -> list[dict[str, Any]]:
    plans = _load_plans()
    rows: list[dict[str, Any]] = []

    for plan_id, payload in plans.items():
        pricing = payload.get("pricing", {})
        coverage = payload.get("coverage", {})
        claims = payload.get("claims", {})

        if pricing.get("contact_sales"):
            price_label = "Contact us"
        elif pricing.get("base_price_weekly_inr") is not None:
            price_label = f"From INR {pricing['base_price_weekly_inr']}/week"
        else:
            price_label = "Dynamic"

        rows.append(
            {
                "plan_id": plan_id,
                "display_name": payload.get("display_name"),
                "price_label": price_label,
                "coverage_events": coverage.get("events", []),
                "zones_allowed": coverage.get("zones_allowed"),
                "compensation_cap_tier": coverage.get("compensation_cap_tier"),
                "review_queue": claims.get("review_queue"),
                "appeal_support": claims.get("appeal_support"),
                "green_instant_payout": claims.get("green_instant_payout"),
            }
        )

    return rows


@router.get("/{plan_id}", response_model=PlanModel)
def get_plan(plan_id: str) -> PlanModel:
    plans = _load_plans()
    normalized_plan_id = plan_id.lower().strip()

    if normalized_plan_id not in plans:
        raise HTTPException(status_code=404, detail=f"Unknown plan: {plan_id}")

    return _normalize_plan(normalized_plan_id, plans[normalized_plan_id])
