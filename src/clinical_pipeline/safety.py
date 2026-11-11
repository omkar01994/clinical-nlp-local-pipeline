"""Rule-based safety layer applied regardless of what either model outputs."""

from __future__ import annotations

import re

DISCLAIMER = (
    "This information is for general education only and is not a diagnosis or a "
    "substitute for advice from a qualified healthcare professional."
)

EMERGENCY_NOTICE = (
    "⚠️ Some of what you describe can be a medical emergency. If symptoms are severe, "
    "sudden, or getting worse, call your local emergency number (e.g. 911 / 112 / 108) "
    "or go to the nearest emergency department now."
)

_RED_FLAGS = [
    r"chest (pain|pressure|tightness)",
    r"(can'?t|cannot|unable to|trouble|difficulty) breath",
    r"short(ness)? of breath",
    r"face (droop|drooping)",
    r"slurred speech",
    r"(sudden|worst) .*headache",
    r"(numb|weak)(ness)? (on|in) one side",
    r"seizure",
    r"unconscious|passed out|fainted",
    r"cough(ing)? (up )?blood|vomit(ing)? blood",
    r"severe (bleeding|allergic|burn)",
    r"anaphyla",
    r"swollen (throat|tongue)",
    r"overdose",
    r"suicid|kill myself|end my life|self[- ]harm",
]
_RED_FLAG_RE = re.compile("|".join(f"(?:{p})" for p in _RED_FLAGS), re.IGNORECASE)


def detect_red_flags(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in _RED_FLAG_RE.finditer(text)})
