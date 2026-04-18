from __future__ import annotations

from rapidfuzz import fuzz

from kosi_assist.types import Detection


ALIASES: dict[str, list[str]] = {
    "powerbank": [
        "power bank",
        "portable charger",
        "battery pack",
        "charger",
        "cell phone",
        "remote",
        "electronic device",
    ],
    "watch": ["wrist watch", "clock", "wristwatch"],
}


ELECTRONICS_TERMS: list[str] = [
    "power bank",
    "charger",
    "adapter",
    "charging cable",
    "usb cable",
    "battery pack",
    "phone",
    "cell phone",
    "tablet",
    "laptop",
    "computer",
    "monitor",
    "keyboard",
    "mouse",
    "printer",
    "router",
    "wlan router",
    "modem",
    "earbuds",
    "headphones",
    "remote",
    "watch",
    "smartwatch",
    "clock",
]


ELECTRONICS_LABEL_HINTS: list[str] = [
    "electronic device",
    "device",
    "power bank",
    "charger",
    "adapter",
    "cable",
    "usb",
    "phone",
    "cell phone",
    "laptop",
    "computer",
    "monitor",
    "keyboard",
    "mouse",
    "printer",
    "router",
    "modem",
    "remote",
    "watch",
    "clock",
]


def build_query_terms(issue_text: str) -> list[str]:
    terms = [issue_text]
    normalized = issue_text.replace("-", " ")
    for key, aliases in ALIASES.items():
        if key in normalized or key.replace(" ", "") in normalized.replace(" ", ""):
            terms.extend(aliases)
    if is_electronics_issue(issue_text):
        terms.extend(ELECTRONICS_TERMS)
    return terms


def select_best_detection(issue_text: str, detections: list[Detection]) -> Detection:
    if not detections:
        raise ValueError("No objects detected in the provided image.")

    issue_text = issue_text.strip().lower()
    if not issue_text:
        return max(detections, key=lambda d: d.confidence)

    query_terms = build_query_terms(issue_text)
    electronics_issue = is_electronics_issue(issue_text)

    best_detection = detections[0]
    best_score = -1.0

    for det in detections:
        label = det.label.lower()
        text_score = 0.0
        for term in query_terms:
            term_score = max(
                fuzz.partial_ratio(term, label),
                fuzz.token_set_ratio(term, label),
            )
            if label in term or term in label:
                term_score += 8.0
            text_score = max(text_score, term_score)
        confidence_score = det.confidence * 100.0
        score = 0.75 * text_score + 0.25 * confidence_score
        if electronics_issue:
            if is_electronics_label(label):
                score += 18.0
            else:
                score -= 20.0
        if score > best_score:
            best_score = score
            best_detection = det

    return best_detection


def label_match_score(issue_text: str, label: str) -> float:
    issue = issue_text.strip().lower()
    if not issue:
        return 100.0

    terms = build_query_terms(issue)
    label_value = label.strip().lower()
    best = 0.0
    for term in terms:
        term_score = max(
            fuzz.partial_ratio(term, label_value),
            fuzz.token_set_ratio(term, label_value),
        )
        if label_value in term or term in label_value:
            term_score += 8.0
        best = max(best, term_score)

    if is_electronics_issue(issue) and is_electronics_label(label_value):
        best += 10.0
    return best


def is_electronics_issue(issue_text: str) -> bool:
    normalized = issue_text.strip().lower()
    if not normalized:
        return False

    for term in ELECTRONICS_TERMS:
        if term in normalized:
            return True
        if fuzz.partial_ratio(normalized, term) >= 88:
            return True
    return False


def is_electronics_label(label: str) -> bool:
    normalized = label.strip().lower()
    if not normalized:
        return False

    for hint in ELECTRONICS_LABEL_HINTS:
        if hint in normalized or normalized in hint:
            return True
        if fuzz.token_set_ratio(normalized, hint) >= 85:
            return True
    return False
