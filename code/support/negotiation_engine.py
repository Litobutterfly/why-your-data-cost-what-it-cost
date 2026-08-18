from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import local
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter


ROOT = Path(__file__).resolve().parents[2]
SELECTION_PATH = ROOT / "results" / "work" / "negotiation" / "m0_full_train_selection.csv"
QUALITY_LEDGER_PATH = ROOT / "results" / "work" / "quality_conditions" / "seller_quality_ledgers.jsonl"
PROMPT_PATH = Path(__file__).resolve().parent / "prompts.py"
OUTPUT_DIR = ROOT / "results" / "work" / "negotiation"
ENDPOINT = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
RUN_VERSION = "m0-bilateral-negotiation-v11-predecision-buyer-assessment"
DEFAULT_MAX_MESSAGES = 16
DEFAULT_M1_PRICE_GATE_TOLERANCE = 0.45
_HTTP_LOCAL = local()


class NonRetryableAPIError(RuntimeError):
    pass


def evaluate_m1_real_price_gate(
    candidate_price_usd: float,
    real_transaction_price_usd: float,
    relative_error_tolerance: float = DEFAULT_M1_PRICE_GATE_TOLERANCE,
) -> dict[str, Any]:
    candidate = float(candidate_price_usd)
    real = float(real_transaction_price_usd)
    tolerance = float(relative_error_tolerance)
    if not candidate > 0 or not real > 0:
        raise ValueError("M1 price gate requires positive candidate and real transaction prices")
    if not 0 <= tolerance < 1:
        raise ValueError("M1 price gate tolerance must be in [0, 1)")
    lower = real * (1 - tolerance)
    upper = real * (1 + tolerance)
    absolute_relative_error = abs(candidate - real) / real
    return {
        "enabled": True,
        "relative_error_tolerance": tolerance,
        "real_transaction_price_usd": real,
        "candidate_negotiated_price_usd": candidate,
        "lower_allowed_price_usd": lower,
        "upper_allowed_price_usd": upper,
        "absolute_relative_error": absolute_relative_error,
        "absolute_log10_error": abs(float(np.log10(candidate / real))),
        "passed": bool(absolute_relative_error <= tolerance),
    }


def http_session() -> requests.Session:
    """Reuse connections within each product worker without sharing sessions across threads."""
    session = getattr(_HTTP_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=0)
        session.mount("https://", adapter)
        session.headers.update({"Connection": "keep-alive"})
        _HTTP_LOCAL.session = session
    return session


def load_prompts():
    spec = importlib.util.spec_from_file_location("symbiotrade_prompts", PROMPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load prompts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ledgers() -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for line in QUALITY_LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            result[int(item["product_id"])] = item
    return result


def agent_safe_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Build a public listing without interpreting opaque catalog units."""
    packet = copy.deepcopy(packet)
    source_generated_listing = str(packet.get("platform_listing", "")).strip()
    original_provenance_note = str(packet.get("listing_provenance_note", "")).strip()
    channels = packet.get("delivery_channels_listed") or []
    updates = packet.get("update_options_listed") or []
    formats = packet.get("formats_listed") or []
    facts = []
    if source_generated_listing:
        facts.append("Source-grounded catalog description: " + source_generated_listing)
    facts.append("This financial data product has a platform record with structured coverage and access fields.")
    facts.append(
        "Listed delivery channels: " + ", ".join(channels) + "."
        if channels else "No standard delivery channel is listed in the supplied platform fields."
    )
    facts.append(
        "Listed update options: " + ", ".join(updates) + "."
        if updates else "No update schedule is listed in the supplied platform fields."
    )
    facts.append(
        "Listed formats: " + ", ".join(formats) + "."
        if formats else "Neither CSV nor JSON is listed in the supplied platform fields."
    )
    if packet.get("company_identifiers_indicated"):
        facts.append("The platform fields indicate company identifiers.")
    if packet.get("individual_identifiers_indicated"):
        facts.append("The platform fields indicate individual identifiers.")
    facts.append(
        f"The country-coverage field lists {int(packet.get('countries_covered', 0))} countries; "
        "practical sufficiency depends on the buyer's intended use."
    )
    packet["platform_listing"] = " ".join(facts)
    packet["source_generated_listing"] = source_generated_listing
    packet["listing_provenance_note"] = (
        "The catalog description was generated only from supplied source-record fields and description-term "
        "signals; it is not independent quality evidence. "
        + (original_provenance_note + " " if original_provenance_note else "")
        + "Opaque source values remain catalog-relative signals with undocumented units."
    )
    packet["inference_constraints"] = [
        "Company or individual identifier flags do not establish row granularity, schema contents, entity status, or identifier standard.",
        "Update options do not establish retention, snapshot access, update time, or full-versus-incremental refresh.",
        "Opaque catalog-relative signals cannot be converted into days, records, money, percentages, or other units.",
        "Catalog-description wording may motivate due diligence but does not establish an unlisted guarantee, defect, or certification.",
    ]
    return packet


def safe_base_ledger(packet: dict[str, Any]) -> dict[str, Any]:
    supplied = [
        {"evidence_id": "record:delivery", "topic": "listed delivery channels", "status": "supplied", "fact": packet.get("delivery_channels_listed", [])},
        {"evidence_id": "record:update", "topic": "listed update options", "status": "supplied", "fact": packet.get("update_options_listed", [])},
        {"evidence_id": "record:format", "topic": "listed formats", "status": "supplied", "fact": packet.get("formats_listed", [])},
        {"evidence_id": "record:coverage", "topic": "coverage fields", "status": "supplied", "fact": {"countries_covered": packet.get("countries_covered"), "gdp_coverage_share_field": packet.get("gdp_coverage_share_field")}},
        {"evidence_id": "record:identifiers", "topic": "identifier flags", "status": "supplied", "fact": {"companies": packet.get("company_identifiers_indicated"), "individuals": packet.get("individual_identifiers_indicated")}},
    ]
    missing_topics = [
        "data provenance documentation", "collection authorization documentation",
        "license or permitted-use text", "privacy or regulatory compliance evidence",
        "external certification or audit", "measured accuracy or error rate",
        "sample records or schema documentation", "service-level guarantee",
        "customer or adoption counts",
    ]
    missing = [
        {"evidence_id": f"missing:{index}", "topic": topic,
         "status": "not supplied in the source record", "fact": None}
        for index, topic in enumerate(missing_topics, start=1)
    ]
    return {
        "ledger_version": "sanitized-source-bounded-ledger-v2",
        "items": supplied + missing,
        "interpretation_constraint": (
            "Only the supplied structured facts may be asserted. An unavailable record is not evidence of failure, "
            "and no opaque source value has a documented physical unit."
        ),
    }


def merge_ledgers(base: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
    return {
        "ledger_version": f"{base.get('ledger_version')}+{quality.get('ledger_version')}",
        "source_record_items": base.get("items", []),
        "quality_observation_items": quality.get("items", []),
        "seller_behavior_contract": quality.get("seller_behavior_contract", {}),
        "interpretation_constraints": [
            base.get("interpretation_constraint", ""), quality.get("interpretation_constraint", "")
        ],
    }


def parse_action(raw: str) -> dict[str, Any]:
    text = raw.strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found")
    value = json.loads(match.group(0))
    if set(value) != {"action", "offer_usd", "message"}:
        raise ValueError("Response must contain exactly action, offer_usd, message")
    if not isinstance(value["message"], str) or not value["message"].strip():
        raise ValueError("message must be a nonempty string")
    if value["offer_usd"] is not None:
        value["offer_usd"] = float(value["offer_usd"])
    return value


def validate_action(
    actor: str,
    action: dict[str, Any],
    *,
    ceiling: float,
    floor: float,
    last_buyer_offer: float | None,
    last_seller_offer: float | None,
    required_action: str | None,
    required_actions: set[str] | None = None,
) -> str | None:
    allowed = {"buyer": {"ask", "assess", "counter", "accept", "walk_away"},
               "seller": {"answer", "counter", "accept", "walk_away"}}[actor]
    kind = action["action"]
    offer = action["offer_usd"]
    if kind not in allowed:
        return f"invalid action {kind} for {actor}"
    if required_action is not None and kind != required_action:
        return f"this protocol phase requires action={required_action}"
    if required_actions is not None and kind not in required_actions:
        return f"this protocol phase requires one of {sorted(required_actions)}"
    public_text = action["message"].lower()
    experiment_leakage_terms = [
        "controlled observation", "controlled treatment", "synthetic condition",
        "synthetic observation", "experimental status", "condition id",
        "real product", "simulation", "prompt rule", "hidden variable", "m0", "m1",
    ]
    leaked = [term for term in experiment_leakage_terms if term in public_text]
    if leaked:
        return f"public message leaks experiment metadata: {leaked}"
    private_bound_leakage_terms = [
        "lowest i can go", "lowest price i can", "minimum i can accept",
        "minimum acceptable", "my floor", "below my floor", "at my floor",
        "my ceiling", "my maximum", "maximum i can pay", "maximum we can pay",
        "highest i can go", "highest we can go", "at my ceiling",
    ]
    bound_leaks = [term for term in private_bound_leakage_terms if term in public_text]
    if bound_leaks:
        return f"public message implies a private price boundary: {bound_leaks}"
    if (kind == "counter") != (offer is not None):
        return "offer_usd must be numeric only for counter"
    proposed_amount = re.search(
        r"\b(?:offer|counteroffer|counter|pay|propose)\b[^$\d]{0,16}\$?\s*([0-9][0-9,]*(?:\.\d+)?)",
        action["message"], flags=re.IGNORECASE,
    )
    first_person_proposal = re.search(
        r"\b(?:i|we)\s+(?:can\s+|could\s+|will\s+|would\s+)?"
        r"(?:offer|counteroffer|counter|pay|propose)\b[^$\d]{0,20}"
        r"\$?\s*([0-9][0-9,]*(?:\.\d+)?)",
        action["message"], flags=re.IGNORECASE,
    )
    if proposed_amount and kind == "accept":
        accepted_amount = float(proposed_amount.group(1).replace(",", ""))
        outstanding = last_seller_offer if actor == "buyer" else last_buyer_offer
        if outstanding is None or abs(accepted_amount - float(outstanding)) > max(0.01, 0.001 * float(outstanding)):
            return "accepted amount does not match the outstanding counteroffer"
    elif first_person_proposal and kind != "counter":
        return "a proposed numeric price in the message requires action=counter"
    if kind == "counter" and (first_person_proposal or proposed_amount):
        proposal_match = first_person_proposal or proposed_amount
        stated = float(proposal_match.group(1).replace(",", ""))
        if abs(stated - float(offer)) > max(0.01, 0.001 * float(offer)):
            return "the proposed price in message does not match offer_usd"
    if kind == "counter" and offer <= 0:
        return "counteroffer must be positive"
    if actor == "buyer" and kind == "counter":
        if offer > ceiling:
            return "buyer counter exceeds private ceiling"
        if last_buyer_offer is not None and offer < last_buyer_offer:
            return "buyer counter decreased"
    if actor == "seller" and kind == "counter":
        if offer < floor:
            return "seller counter is below private floor"
        if last_seller_offer is not None and offer > last_seller_offer:
            return "seller counter increased"
    if actor == "buyer" and kind == "accept":
        if last_seller_offer is None or last_seller_offer > ceiling:
            return "buyer cannot accept the outstanding seller offer"
    if actor == "seller" and kind == "accept":
        if last_buyer_offer is None or last_buyer_offer < floor:
            return "seller cannot accept the outstanding buyer offer"
    return None


def call_agent(
    *,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    actor: str,
    state: dict[str, Any],
    timeout: float,
    retries: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    validation_feedback = ""
    for attempt in range(1, retries + 2):
        effective_user = user_prompt + validation_feedback
        body = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": effective_user},
            ],
            "temperature": 0.0,
            "max_tokens": 240,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        started = time.time()
        raw = ""
        try:
            response = http_session().post(
                ENDPOINT,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body,
                timeout=timeout,
            )
            raw = response.text
            if response.status_code in {400, 401, 402, 403, 404, 405, 422}:
                raise NonRetryableAPIError(f"HTTP {response.status_code}: {raw[:400]}")
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {raw[:400]}")
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            action = parse_action(content)
            error = validate_action(actor, action, **state)
            attempts.append({
                "attempt": attempt, "status": "ok" if error is None else "invalid",
                "validation_error": error or "", "raw_response": raw,
                "elapsed_seconds": round(time.time() - started, 3), "usage": payload.get("usage", {}),
            })
            if error is None:
                return action, attempts
            correction = (
                " Do not mention any dollar amount or propose a price in the public message; "
                "answer only the evidence question."
                if "numeric price" in error and "counter" not in state.get("required_actions", [])
                else ""
            )
            validation_feedback = (
                "\n\nYour previous response was invalid: " + error
                + ". Return a corrected JSON action that obeys your private price bounds and monotonicity rules. "
                + "Do not mention or reveal any private boundary in the public message."
                + correction
            )
        except Exception as exc:
            attempts.append({
                "attempt": attempt, "status": "error", "validation_error": repr(exc),
                "raw_response": raw, "elapsed_seconds": round(time.time() - started, 3), "usage": {},
            })
            if isinstance(exc, NonRetryableAPIError):
                raise RuntimeError(
                    f"{actor} API request failed without retry: {exc}"
                ) from exc
            validation_feedback = "\n\nReturn only a valid JSON object in the required schema."
            if attempt <= retries:
                time.sleep(min(8.0, 2 ** (attempt - 1)))
    compact_errors = [
        {"attempt": item["attempt"], "status": item["status"],
         "validation_error": item["validation_error"], "raw_response": item["raw_response"][:1200]}
        for item in attempts
    ]
    required_actions = state.get("required_actions")
    if required_actions is not None and set(required_actions).issubset(
        {"accept", "counter", "walk_away"}
    ):
        last_buyer_offer = state.get("last_buyer_offer")
        last_seller_offer = state.get("last_seller_offer")
        if actor == "buyer":
            feasible = (
                last_seller_offer is not None
                and float(last_seller_offer) <= float(state["ceiling"])
            )
            fallback = {
                "action": "accept" if feasible else "walk_away",
                "offer_usd": None,
                "message": (
                    f"I accept the outstanding offer of ${float(last_seller_offer):.2f}."
                    if feasible
                    else "I cannot justify the outstanding terms for the intended use, so I will step away."
                ),
            }
        else:
            feasible = (
                last_buyer_offer is not None
                and float(last_buyer_offer) >= float(state["floor"])
            )
            fallback = {
                "action": "accept" if feasible else "walk_away",
                "offer_usd": None,
                "message": (
                    f"I accept the outstanding offer of ${float(last_buyer_offer):.2f}."
                    if feasible
                    else "The outstanding terms do not support an agreement, so I will step away."
                ),
            }
        fallback_error = validate_action(actor, fallback, **state)
        if fallback_error is None:
            attempts.append(
                {
                    "attempt": retries + 2,
                    "status": "deterministic_finality_fallback",
                    "validation_error": "",
                    "raw_response": "",
                    "elapsed_seconds": 0.0,
                    "usage": {},
                    "prior_model_errors": compact_errors,
                }
            )
            return fallback, attempts
    raise RuntimeError(
        f"{actor} failed after {retries + 1} attempts: "
        + json.dumps(compact_errors, ensure_ascii=False)
    )


def scenario(
    row: pd.Series,
    *,
    quality_ledger: dict[str, Any],
    prompts: Any,
    api_key: str,
    max_messages: int,
    timeout: float,
    retries: int,
    patience_assignment: pd.Series,
    real_transaction_price_usd: float | None = None,
    m1_price_gate_tolerance: float = DEFAULT_M1_PRICE_GATE_TOLERANCE,
) -> dict[str, Any]:
    product_id = int(row["product_id"])
    packet = agent_safe_packet(json.loads(row["public_product_packet_json"]))
    buyer_context = json.loads(row["buyer_private_context_json"])
    seller_context = json.loads(row["seller_private_context_json"])
    base_ledger = safe_base_ledger(packet)
    controlled_items = [
        item for item in quality_ledger.get("items", [])
        if item.get("evidence_class") == "controlled synthetic quality observation"
    ]
    expected_evidence_id = f"controlled:{row['controlled_condition_id']}"
    if len(controlled_items) != 1 or controlled_items[0].get("evidence_id") != expected_evidence_id:
        raise RuntimeError(
            f"Quality ledger mismatch for product {product_id}: expected {expected_evidence_id}"
        )
    ledger = merge_ledgers(base_ledger, quality_ledger)
    disclosure_ledger = {
        "ledger_version": quality_ledger.get("ledger_version"),
        "quality_observation_items": controlled_items,
        "seller_behavior_contract": quality_ledger.get("seller_behavior_contract", {}),
        "interpretation_constraints": [quality_ledger.get("interpretation_constraint", "")],
    }
    reference = float(row["m0_platform_reference_usd"])
    ceiling = float(row["buyer_ceiling_usd"])
    floor = float(row["seller_floor_usd"])
    history: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    last_buyer_offer: float | None = None
    last_seller_offer: float | None = None
    outcome = "undecided"
    transaction_price: float | None = None
    m1_price_gate_private: dict[str, Any] = {"enabled": False}
    patience = {
        "buyer": int(patience_assignment["buyer_initial_patience"]),
        "seller": int(patience_assignment["seller_initial_patience"]),
    }
    initial_patience = dict(patience)
    action_counts = {"buyer": 0, "seller": 0}
    pending_final_decision: str | None = None
    disclosure_cost_pending = False
    patience_trace: list[dict[str, Any]] = []
    description_grounded_protocol = bool(packet.get("source_description_term_signals"))

    for index in range(max_messages):
        actor = "seller" if index % 2 == 0 else "buyer"
        other = "buyer" if actor == "seller" else "seller"
        if pending_final_decision == actor:
            required_actions = {"accept", "walk_away"}
        elif index == max_messages - 1:
            required_actions = {"accept", "walk_away"}
        elif patience[actor] <= 1:
            required_actions = {"accept", "counter", "walk_away"}
        else:
            required_actions = None
        if patience[actor] <= 0:
            required_actions = {"accept", "walk_away"}
        patience_state = {
            "remaining_action_opportunities": patience[actor],
            "initial_action_opportunities": initial_patience[actor],
            "actions_taken": action_counts[actor],
            "protocol_note": (
                "Make a final decision or final counteroffer when only one opportunity remains. "
                "Do not disclose this numeric state."
            ),
        }
        if actor == "seller":
            active_ledger = (
                ledger if description_grounded_protocol else disclosure_ledger
            ) if index == 2 else ledger
            user_prompt = prompts.seller_user_prompt(
                public_product_packet=packet,
                platform_reference_price_usd=reference,
                private_context=seller_context,
                private_floor_usd=floor,
                evidence_ledger=active_ledger,
                dialogue_history=history,
                patience_state=patience_state,
            )
            system_prompt = prompts.SELLER_SYSTEM_PROMPT
        else:
            user_prompt = prompts.buyer_user_prompt(
                public_product_packet=packet,
                platform_reference_price_usd=reference,
                private_context=buyer_context,
                private_ceiling_usd=ceiling,
                dialogue_history=history,
                patience_state=patience_state,
            )
            system_prompt = prompts.BUYER_SYSTEM_PROMPT
        if index == 1:
            if description_grounded_protocol:
                user_prompt += (
                    "\n\nProtocol phase: this is the buyer's source-description due-diligence turn. Use action=ask "
                    "and offer_usd=null. Briefly state the intended use. Select the one most decision-relevant "
                    "source-description concept that is not already resolved by the structured delivery, format, "
                    "update, identifier, or country fields, and ask what concrete product content or evidence supports "
                    "that wording for your intended use. A description-term signal records catalog wording, not proof "
                    "of quality or a guarantee. Ask one focused question and do not propose a price."
                )
            else:
                user_prompt += (
                    "\n\nProtocol phase: this is the buyer's neutral due-diligence turn. Use action=ask and offer_usd=null. "
                    "Briefly state the intended use, then ask whether the seller has any recent internal observation or "
                    "evidence limitation that could materially affect that use. Do not name or list possible issue types, "
                    "and do not make a price proposal in this turn."
                )
        elif index == 2:
            if description_grounded_protocol:
                user_prompt += (
                    "\n\nProtocol phase: answer the buyer's source-description question with action=answer and "
                    "offer_usd=null. Distinguish what the source record actually supports from catalog wording that "
                    "has no more specific supplied evidence. Do not invent an interpretation, metric, guarantee, or "
                    "defect. Include a registered quality observation only when it directly answers the selected "
                    "description concept; do not volunteer an unrelated observation. Do not propose a price."
                )
            else:
                user_prompt += (
                    "\n\nProtocol phase: answer the buyer's neutral due-diligence question now with action=answer and "
                    "offer_usd=null. Use the registered quality-observation item in your private ledger. State the concrete "
                    "observation and its evidence boundary in neutral commercial language. Do not state its condition ID, "
                    "experimental status, or a quality verdict. Do not propose a price in this turn."
                )
        elif index == 3:
            source_note = (
                " Explicitly separate a supported product property from unresolved catalog wording, and state how "
                "that distinction changes expected utility, verification effort, integration cost, or risk."
                if description_grounded_protocol
                else ""
            )
            user_prompt += (
                "\n\nProtocol phase: assess what the seller's latest answer actually changes for your stated use. "
                "The update may concern task fit, expected utility, verification or integration effort, time-to-value, "
                "risk, confidence, trust, or objection resolution, and it may be favorable, unfavorable, mixed, or "
                "immaterial. State only consequences supported by the specific answer; do not invent a burden from a "
                "generic no-additional-evidence response."
                + source_note
                + " Use action=assess and offer_usd=null. This is a non-price interpretation turn: do not propose a "
                "price, accept, walk away, or imply a final transaction decision."
            )
        if required_actions is not None:
            user_prompt += (
                "\n\nFinality protocol: this is a finite-patience decision phase. "
                "You must choose one of the permitted actions now. If the outstanding offer is feasible and "
                "acceptable, use accept; otherwise use walk_away. A counteroffer is permitted only when the "
                "protocol explicitly allows it and it must be your final counteroffer. Do not leave the exchange open."
            )
        action, attempts = call_agent(
            api_key=api_key,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            actor=actor,
            state={
                "ceiling": ceiling, "floor": floor,
                "last_buyer_offer": last_buyer_offer, "last_seller_offer": last_seller_offer,
                "required_action": (
                    "ask" if index == 1 else "answer" if index == 2 else "assess" if index == 3 else None
                ),
                "required_actions": required_actions,
            },
            timeout=timeout,
            retries=retries,
        )
        cost = 1
        if actor == "buyer" and disclosure_cost_pending:
            cost += 1
            disclosure_cost_pending = False
        patience_before = patience[actor]
        patience[actor] = max(0, patience[actor] - cost)
        action_counts[actor] += 1
        if actor == "seller" and index == 2 and bool(patience_assignment["controlled_condition_present_audit_only"]):
            disclosure_cost_pending = True
        if action["action"] == "counter":
            if actor == "buyer":
                last_buyer_offer = float(action["offer_usd"])
            else:
                last_seller_offer = float(action["offer_usd"])
        public_event = {
            "message_index": index + 1, "actor": actor, "action": action["action"],
            "offer_usd": action["offer_usd"], "message": action["message"].strip(),
        }
        history.append(public_event)
        calls.append({"message_index": index + 1, "actor": actor, "attempts": attempts})
        patience_trace.append({
            "message_index": index + 1, "actor": actor,
            "patience_before": patience_before, "patience_cost": cost,
            "patience_after": patience[actor],
            "action": action["action"],
        })
        if action["action"] == "walk_away":
            outcome = f"{actor}_walked_away"
            break
        if action["action"] == "accept":
            candidate_price = last_seller_offer if actor == "buyer" else last_buyer_offer
            if candidate_price is None:
                raise RuntimeError("Accept action has no outstanding candidate transaction price")
            if real_transaction_price_usd is not None:
                m1_price_gate_private = evaluate_m1_real_price_gate(
                    candidate_price,
                    real_transaction_price_usd,
                    m1_price_gate_tolerance,
                )
                if not m1_price_gate_private["passed"]:
                    outcome = "m1_price_gate_rejected"
                    transaction_price = None
                    break
            outcome = "agreement"
            transaction_price = candidate_price
            break
        if action["action"] == "counter" and patience[actor] == 0:
            pending_final_decision = other
        if action["action"] in {"ask", "assess", "answer"} and patience[actor] == 0:
            pending_final_decision = other

    if outcome == "undecided":
        raise RuntimeError(
            f"Negotiation for product {product_id} reached the safety ceiling without a final decision"
        )

    return {
        "run_version": RUN_VERSION,
        "prompt_version": prompts.PROMPT_VERSION,
        "release_slot": row["release_slot"],
        "product_id": product_id,
        "controlled_condition_id_audit_only": row["controlled_condition_id"],
        "observable_readiness_level_audit_only": row["observable_readiness_level_selection"],
        "potential_price_overlap_audit_only": bool(row["potential_price_overlap"]),
        "buyer_profile_id_audit_only": row["buyer_profile_id_selection"],
        "seller_profile_id_audit_only": row["seller_profile_id_selection"],
        "m0_platform_reference_usd": reference,
        "buyer_ceiling_usd_private": ceiling,
        "seller_floor_usd_private": floor,
        "outcome": outcome,
        "negotiated_price_usd": transaction_price,
        "m1_real_price_gate_private": m1_price_gate_private,
        "message_count": len(history),
        "initial_patience_private": initial_patience,
        "patience_trace_private": patience_trace,
        "termination_reason": outcome,
        "dialogue": history,
        "api_calls": calls,
        "agent_public_packet": packet,
        "quality_ledger_sha256": hashlib.sha256(
            json.dumps(quality_ledger, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def run(api_key: str, limit: int | None, max_messages: int, timeout: float, retries: int) -> dict[str, Any]:
    if not api_key:
        raise RuntimeError("Missing key: set DEEPSEEK_API_KEY or AI_EDGE_API_KEY")
    if max_messages != DEFAULT_MAX_MESSAGES:
        raise ValueError(
            f"Finite-patience protocol requires the fixed {DEFAULT_MAX_MESSAGES}-message safety ceiling"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(SELECTION_PATH, low_memory=False)
    patience = pd.read_csv(OUTPUT_DIR / "train_frozen_patience_assignments.csv", low_memory=False)
    selected = selected.merge(patience, on="product_id", validate="one_to_one", suffixes=("_selection", "_patience"))
    if limit is not None:
        selected = selected.head(limit).copy()
    ledgers = load_ledgers()
    prompts = load_prompts()
    checkpoint = OUTPUT_DIR / "m0_release_dialogues_v11.jsonl"
    existing: dict[int, dict[str, Any]] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                existing[int(item["product_id"])] = item
    for _, row in selected.iterrows():
        product_id = int(row["product_id"])
        if product_id in existing:
            continue
        item = scenario(
            row, quality_ledger=ledgers[product_id], prompts=prompts, api_key=api_key,
            max_messages=max_messages, timeout=timeout, retries=retries,
            patience_assignment=row,
        )
        with checkpoint.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
        existing[product_id] = item
        print(f"saved product={product_id} outcome={item['outcome']} messages={item['message_count']}", flush=True)

    records = [existing[int(product_id)] for product_id in selected["product_id"] if int(product_id) in existing]
    summary = pd.DataFrame([
        {
            "product_id": item["product_id"], "release_slot": item["release_slot"],
            "condition": item["controlled_condition_id_audit_only"], "outcome": item["outcome"],
            "negotiated_price_usd": item["negotiated_price_usd"], "message_count": item["message_count"],
        }
        for item in records
    ])
    summary.to_csv(OUTPUT_DIR / "m0_release_dialogue_summary_v11.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "run_version": RUN_VERSION, "prompt_version": prompts.PROMPT_VERSION,
        "requested": len(selected), "completed": len(records),
        "agreements": sum(item["outcome"] == "agreement" for item in records),
        "model": MODEL, "endpoint": ENDPOINT, "temperature": 0.0,
        "max_messages": max_messages, "checkpointed": True,
        "message_limit_outcome_allowed": False,
        "real_transaction_price_visible_to_agents": False,
        "buyer_received_quality_ledger": False,
    }
    (OUTPUT_DIR / "m0_release_run_manifest_v11.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-messages", type=int, default=DEFAULT_MAX_MESSAGES)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("AI_EDGE_API_KEY", "")
    print(json.dumps(run(key, args.limit, args.max_messages, args.timeout, args.retries), ensure_ascii=False, indent=2))
