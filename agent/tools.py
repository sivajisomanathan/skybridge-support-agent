"""
Mocked booking database and deterministic policy calculator. Unchanged logic
from Phase 5-7, moved into a shared module.
"""
from typing import Optional

BOOKING_DB = {
    "AB1234": {"fare_type": "standard", "delay_hours": 6, "exit_row_eligible": True},
    "CD5678": {"fare_type": "flex",     "delay_hours": 0, "exit_row_eligible": True},
    "EF9012": {"fare_type": "basic",    "delay_hours": 0, "exit_row_eligible": False},
    "GH3456": {"fare_type": "standard", "delay_hours": 3, "exit_row_eligible": True},
}


def booking_lookup(pnr: Optional[str]) -> dict:
    if not pnr:
        return {"booking_found": False, "pnr": pnr, "reason": "no_pnr_provided"}
    record = BOOKING_DB.get(pnr.upper())
    if record is None:
        return {"booking_found": False, "pnr": pnr, "reason": "pnr_not_found"}
    return {"booking_found": True, "pnr": pnr.upper(), **record}


def policy_calculator(calc_type: str, delay_hours: Optional[int] = None,
                       fare_type: Optional[str] = None) -> dict:
    if calc_type == "delay_compensation":
        if delay_hours is None:
            return {"error": "missing delay_hours"}
        if delay_hours < 2:
            tier = {"voucher_percent": 0, "meal_voucher": False, "hotel": False}
        elif delay_hours < 4:
            tier = {"voucher_percent": 25, "meal_voucher": False, "hotel": False}
        elif delay_hours < 6:
            tier = {"voucher_percent": 50, "meal_voucher": True, "hotel": False}
        else:
            tier = {"voucher_percent": 100, "meal_voucher": True, "hotel": True}
        return {"calc_type": calc_type, "delay_hours": delay_hours, **tier}

    if calc_type == "refund_fee":
        if not fare_type:
            return {"error": "missing fare_type"}
        fare_type = fare_type.lower()
        if fare_type == "flex":
            result = {"refundable": True, "fee": 0, "credit_only": False}
        elif fare_type == "standard":
            result = {"refundable": True, "fee": 75, "credit_only": False}
        elif fare_type == "basic":
            result = {"refundable": False, "fee": 75, "credit_only": True, "credit_validity_months": 12}
        else:
            return {"error": f"unknown fare_type: {fare_type}"}
        return {"calc_type": calc_type, "fare_type": fare_type, **result}

    return {"error": f"unknown calc_type: {calc_type}"}
