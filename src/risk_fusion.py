"""Multi-signal risk fusion: turns "how suspicious is the voice itself"
(voice authenticity, from RiskEngine/wav2vec2) plus "what do we know about
this interaction" (context) into "how risky is it to trust this interaction"
(interaction risk). These are kept as two distinct numbers on purpose --
a call can score low on voice authenticity alone but still be risky in
context (unknown caller, high-value transfer), which a single voice-only
score can never express. See HANDOFF.md for the architecture this
implements and why.

The context signals and fusion here are simple, hand-picked POLICY
choices for the prototype (config.CONTEXT_UNKNOWN_CALLER_RISK,
config.TRANSACTION_VALUE_RISK, config.DECISION_BANDS) -- not a learned
model. A production system would tune or learn these from labeled fraud
outcomes; described here as the honest, disclosed scope of this prototype.
"""
from config import CONTEXT_UNKNOWN_CALLER_RISK, TRANSACTION_VALUE_RISK, DECISION_BANDS


def compute_context_risk(known_contact: bool, transaction_value: str) -> int:
    """Rule-based context score, 0-100. Not an ML model -- see module docstring."""
    risk = 0
    if not known_contact:
        risk += CONTEXT_UNKNOWN_CALLER_RISK
    risk += TRANSACTION_VALUE_RISK.get(transaction_value, 0)
    return min(100, risk)


def fuse_risk(voice_authenticity: int, context_risk: int) -> int:
    """Context is an additive bump on top of voice authenticity, never a
    dilution -- see module docstring for why."""
    return min(100, voice_authenticity + context_risk)


def decision_for(interaction_risk: int) -> dict:
    """Map an interaction risk to a decision-policy band: {"band", "action"}."""
    for upper, band, action in DECISION_BANDS:
        if interaction_risk < upper:
            return {"band": band, "action": action}
    _, band, action = DECISION_BANDS[-1]
    return {"band": band, "action": action}
