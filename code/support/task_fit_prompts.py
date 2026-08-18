from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "07_agents" / "prompts.py"


def load_base():
    spec = importlib.util.spec_from_file_location("symbiotrade_v3_base_prompts", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base()
PROMPT_VERSION = "symbiotrade-task-fit-valuation-v3"

_OLD_BUYER_SCHEMA = """Return exactly one JSON object with keys: action, offer_usd, message.
Allowed actions: ask, assess, counter, accept, walk_away.
Use null for offer_usd unless action is counter. Never include additional keys."""
_NEW_BUYER_SCHEMA = """Evidence-to-value update:
- Your private context contains one pre-specified task requirement. Compare only the seller's grounded answer with that requirement.
- On action=assess, return value_update as a continuous number in [-1,1]: positive only when the answer improves expected task value, negative only when it reduces task value or adds supported cost/risk, and zero when immaterial.
- A confirmed property can be favorable or unfavorable depending on task fit. Unresolved evidence is uncertainty, not proof of a defect.
- Use the full continuous scale when justified; do not restrict scores to a small fixed menu. Do not disclose the score or any private price boundary in the public message.
- On every action other than assess, value_update must be null.

Return exactly one JSON object with keys: action, offer_usd, message, value_update.
Allowed actions: ask, assess, counter, accept, walk_away.
Use null for offer_usd unless action is counter. Never include additional keys."""

if _OLD_BUYER_SCHEMA not in BASE.BUYER_SYSTEM_PROMPT:
    raise RuntimeError("Expected buyer schema was not found in the frozen prompt")

BUYER_SYSTEM_PROMPT = BASE.BUYER_SYSTEM_PROMPT.replace(
    _OLD_BUYER_SCHEMA, _NEW_BUYER_SCHEMA
)
SELLER_SYSTEM_PROMPT = BASE.SELLER_SYSTEM_PROMPT


def buyer_user_prompt(**kwargs: Any) -> str:
    return BASE.buyer_user_prompt(**kwargs)


def seller_user_prompt(**kwargs: Any) -> str:
    return BASE.seller_user_prompt(**kwargs)
