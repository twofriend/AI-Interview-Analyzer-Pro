from __future__ import annotations


def clarity_score(sentiment_score: float, filler_rate: float, lexical_diversity: float,
                  keyword_coverage: float, has_keywords: bool) -> float:
    filler_component = max(0.0, 100 - filler_rate * 8)
    diversity_component = max(0.0, min(100.0, lexical_diversity * 1.5))
    if has_keywords:
        score = sentiment_score * 0.20 + filler_component * 0.35 + diversity_component * 0.25 + keyword_coverage * 0.20
    else:
        score = sentiment_score * 0.25 + filler_component * 0.45 + diversity_component * 0.30
    return round(score, 2)


def overall_practice_score(composure: float, gaze: float, delivery: float, clarity: float,
                           weights: dict[str, int]) -> float:
    total = sum(max(0, int(v)) for v in weights.values()) or 1
    score = (composure * weights["composure"] + gaze * weights["gaze"] +
             delivery * weights["delivery"] + clarity * weights["clarity"]) / total
    return round(max(0.0, min(100.0, score)), 2)


def performance_band(score: float) -> str:
    if score >= 85:
        return "Excellent practice performance"
    if score >= 70:
        return "Good practice performance"
    if score >= 55:
        return "Developing — review the feedback areas"
    return "Needs additional interview practice"
