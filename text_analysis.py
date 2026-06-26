from __future__ import annotations

import re
from dataclasses import dataclass

FILLERS = ["um", "uh", "erm", "hmm", "you know", "like", "basically", "actually", "sort of", "kind of"]


@dataclass
class TextResult:
    sentiment_label: str
    sentiment_score: float
    polarity: float
    word_count: int
    filler_count: int
    filler_rate_per_100_words: float
    lexical_diversity: float
    keyword_coverage: float
    matched_keywords: list[str]
    missing_keywords: list[str]


def analyze_text(text: str, target_keywords: list[str] | None = None) -> TextResult:
    target_keywords = [k.strip().lower() for k in (target_keywords or []) if k.strip()]
    words = re.findall(r"\b[\w'-]+\b", text.lower())
    word_count = len(words)
    lower_text = text.lower()
    filler_count = sum(len(re.findall(rf"\b{re.escape(f)}\b", lower_text)) for f in FILLERS)
    filler_rate = filler_count / word_count * 100 if word_count else 0.0
    diversity = len(set(words)) / word_count * 100 if word_count else 0.0

    try:
        from textblob import TextBlob
        polarity = float(TextBlob(text).sentiment.polarity) if text else 0.0
    except Exception:
        polarity = 0.0
    sentiment_score = (polarity + 1) / 2 * 100
    label = "Positive" if sentiment_score >= 65 else "Negative" if sentiment_score <= 40 else "Neutral"

    matched = [k for k in target_keywords if k in lower_text]
    missing = [k for k in target_keywords if k not in lower_text]
    coverage = len(matched) / len(target_keywords) * 100 if target_keywords else 0.0

    return TextResult(label, round(sentiment_score, 2), round(polarity, 3), word_count, filler_count,
                      round(filler_rate, 2), round(diversity, 2), round(coverage, 2), matched, missing)
