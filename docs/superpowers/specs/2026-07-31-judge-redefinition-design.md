# Design: Judge Redefinition — Rubric Threshold + Filter Awareness

**Date:** 2026-07-31
**Status:** Approved (pending spec review)

## Purpose

Redefine the context-quality judge to address two concrete pitfalls surfaced after the judge response panel made its raw output visible:

1. **No rubric detail.** The judge outputs one free-text word parsed by substring match against a vague "accurately and completely" instruction — no graduated notion of partial sufficiency.
2. **No awareness of filtered content.** The judge only sees chunks that survived city/price filtering. It has no way to know relevant content was excluded, and the existing retry (widen retrieval depth) can't fix a case where the answer was filtered out, not under-retrieved.

## Current State (baseline, verified against code)

- `app/retrieval/judge.py::judge_context()` sends one Groq chat-completion call, prompt asks for exactly `"context_good"` or `"context_insufficient"`, parsed via `"insufficient" in normalized` then `"good" in normalized`, else `JudgeError`.
- `JudgeVerdict(verdict: Literal["context_good","context_insufficient"], raw_response: str)` — `raw_response` was added in the prior plan (judge-response-panel) and is now rendered in a Query-screen side panel.
- `app/retrieval/filtering.py::filter_chunks(fused_chunks, preferences) -> (kept, excluded_count)` — excluded chunks are dropped from the return value entirely, only a count survives.
- `app/retrieval/router.py`: judge is called against `kept_chunks` only, never sees what filtering removed. Retry (on insufficient) widens `retrieval_top_k`/`display_top_k` via `dataclasses.replace` and re-runs retrieve → filter → judge once; still exactly one retry, then a fixed fallback answer.
- `JudgeAttempt(attempt: int, verdict: str, raw_response: str)` on `QueryResponse.judge_attempts`, rendered in `#judge-panel` on the Query screen.

## Changes

### 1. `app/retrieval/filtering.py`

`filter_chunks` signature changes to return the excluded chunks themselves, not just a count:
```python
def filter_chunks(
    fused_chunks: list[FusedChunk],
    preferences: QueryPreferences,
) -> tuple[list[FusedChunk], list[FusedChunk], int]:
    ...
    return kept, excluded, len(excluded)
```
Filtering logic (`_conflicts`) is unchanged — this is purely a return-shape change so callers can access what got excluded.

### 2. `app/config.py`

New field: `judge_sufficiency_threshold: int = 70` — the score (0-100) at or above which context is considered sufficient.

### 3. `app/retrieval/judge.py`

- `JudgeVerdict` becomes:
  ```python
  class JudgeVerdict(BaseModel):
      score: int              # 0-100: judge's estimate of how much of the query this context answers
      recommend_unfilter: bool  # true if the judge believes excluded chunks would raise the score
      raw_response: str        # full raw response text from Groq, unparsed — shown verbatim in the panel
  ```
- `judge_context()` signature gains an `excluded_chunks: list[FusedChunk]` parameter (alongside the existing kept `chunks` param), so the judge can be shown both.
- Groq call uses JSON mode: `response_format={"type": "json_object"}`. Prompt asks the model to return `{"score": <0-100 int>, "recommend_unfilter": <bool>, "reasoning": <string>}`.
- New system prompt framing, explicit percentage rubric: "Estimate what percentage (0-100) of the question this context could answer. If chunks were excluded by a filter, and you believe they contain information that would raise the score, set recommend_unfilter to true."
- The prompt's user message includes two labeled sections: "Retrieved context" (kept chunks, numbered, same as today) and "Excluded by filter" (the excluded chunks' text, numbered separately) — omitted/empty section text if there are no excluded chunks.
- Parsing: `response.json()["choices"][0]["message"]["content"]` is now expected to be a JSON string; parse it, extract `score`/`recommend_unfilter`/`reasoning`. Missing keys, wrong types, or invalid JSON → `JudgeError` (fail-closed), same failure posture as today — no silent defaults.
- `raw_response` is set to the full raw content string returned by Groq (the JSON blob), not just the reasoning — preserves complete transparency for the panel.
- Empty-`chunks`-and-`excluded_chunks` short-circuit (nothing was retrieved or survived at all): `JudgeVerdict(score=0, recommend_unfilter=False, raw_response="(no chunks retrieved)")`, no Groq call.
- `JudgeError` still exists for network/status/parse failures, unchanged exception type.

### 4. `app/retrieval/router.py`

- `_judge_safely`'s except branch: `JudgeVerdict(score=0, recommend_unfilter=False, raw_response=f"(judge error: {exc})")` — score of 0 guarantees fail-closed (always below any reasonable threshold) without inventing a fake "insufficient" enum value.
- `filter_chunks` call sites updated to the new 3-tuple return; `excluded_chunks` passed into `judge_context`.
- Sufficiency check: `verdict.score >= settings.judge_sufficiency_threshold` replaces the old `verdict.verdict == "context_insufficient"` string check, everywhere it's used (initial check, retry decision, final fallback decision).
- Retry: existing widened-depth `dataclasses.replace` behavior is unchanged. Additionally, if the first attempt's `verdict.recommend_unfilter` was `true`, the retry's `filter_chunks` call is skipped entirely — the retry judges (and, if sufficient, synthesizes) against the full unfiltered retrieved set. If `recommend_unfilter` was `false`, retry behaves exactly as before (widened depth, same filter still applied).
- Still exactly one retry, no change to that constraint.
- `JudgeAttempt` construction gains `score` and `recommend_unfilter`, populated from each attempt's `JudgeVerdict`. A derived `verdict: str` field is still computed (`"sufficient"` or `"insufficient"`, from the same threshold comparison) so the panel heading reads naturally without the frontend re-implementing the threshold logic.

### 5. `app/retrieval/models.py`

`JudgeAttempt` gains two fields:
```python
class JudgeAttempt(BaseModel):
    attempt: int
    verdict: str          # "sufficient" | "insufficient", derived from score vs threshold
    score: int
    recommend_unfilter: bool
    raw_response: str
```

### 6. Frontend (`app/static/js/query.js`)

`renderJudgePanel` heading changes from `"Attempt N: {verdict}"` to `"Attempt N: {score}% {verdict}"`, with `" — unfilter recommended"` appended when `recommend_unfilter` is true. Raw JSON (`raw_response`) still rendered below via `textContent`, unchanged rendering discipline (no `innerHTML` for any judge-derived text).

## Non-Goals

- No change to the judge's underlying model (`settings.judge_model` stays `llama-3.1-8b-instant`) or to the retrieval-depth-widening retry mechanism itself — this redefinition adds a second, independent retry lever (unfiltering), it doesn't replace the first.
- No cap/budget on how many excluded chunks are shown to the judge (mirrors the existing uncapped `kept_chunks` behavior) — if this becomes a real prompt-size problem in practice, that's a separate follow-up.
- No change to the fail-closed philosophy: a judge error still can't accidentally produce "sufficient." Score-of-0-on-error is the mechanism, not a new escape hatch.
- No retry-count change — still exactly one retry, now potentially compounding two independent adjustments (widened depth + dropped filter) in that single retry rather than offering a third attempt to try them separately.
- No UI change to the filter-toggle or filtered-out-count note on the main column — this plan only touches the judge panel.

## Testing Plan

- `tests/retrieval/test_judge.py`: rewrite Groq-response mocks to return JSON bodies (`{"score": N, "recommend_unfilter": bool, "reasoning": "..."}`); cover score-above/below-threshold-adjacent values, `recommend_unfilter` true/false, malformed JSON → `JudgeError`, missing keys → `JudgeError`, empty-chunks-and-excluded-chunks short-circuit, and that excluded-chunk text actually appears in the request body sent to Groq.
- `tests/retrieval/test_filtering.py`: update all existing assertions for the new 3-tuple return (`kept, excluded, count`); add a case asserting `excluded` contains the actual conflicting `FusedChunk` objects, not just a count.
- `tests/retrieval/test_router.py`: update existing mocks to return the new JSON judge shape; add a case where attempt 1 has `recommend_unfilter=true` and assert the retry's retrieved-chunk set is NOT filtered (a chunk that would've been excluded appears in the final `retrieved_chunks`); add a case where `recommend_unfilter=false` and assert filtering IS still applied on retry (regression guard against accidentally always dropping the filter).
- `tests/retrieval/test_models.py`: extend `JudgeAttempt` round-trip test for the two new fields.
- No JS-level automated test (documented, pre-existing exception in this repo) — manual/Playwright verification covers the panel rendering itself.
