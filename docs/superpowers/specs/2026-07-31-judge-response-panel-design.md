# Design: Judge Response Panel

**Date:** 2026-07-31
**Status:** Approved (pending spec review)

## Purpose

Surface the context-quality judge's raw response(s) on the Query screen, in a fixed side panel, so a user can see exactly what the judge said and why — a step toward redefining the judge's behavior later, and consistent with the app's "transparency over polish" ethos.

## Current State (baseline, verified against code)

- `app/retrieval/judge.py::judge_context()` calls Groq, parses the response's `message.content` into a `JudgeVerdict(verdict: Literal["context_good", "context_insufficient"])`, and **discards the raw text** — only the parsed enum survives.
- `app/retrieval/router.py`'s `_judge_safely()` wraps `judge_context()` and converts any `JudgeError` into a synthetic `JudgeVerdict(verdict="context_insufficient")` with no raw text available (the failure reason is lost).
- The router calls the judge up to twice per request: once on the initial retrieval, once more on the widened retry if the first verdict was `context_insufficient`. Neither judge call's output reaches the API response today.
- `QueryResponse` has no field carrying judge output at all.
- Query screen (`app/static/query.html`) is a single-column layout (`body { max-width: 800px }` in `app.css`).

## Changes

### 1. `app/retrieval/judge.py`

`JudgeVerdict` gains a required field: `raw_response: str` — the judge's actual message content string.

- Real Groq call succeeds → `raw_response` = the actual `content` string (same text `judge_context` already parses for the verdict).
- Empty `chunks` short-circuit (no Groq call made) → `raw_response = "(no chunks retrieved)"`.

### 2. `app/retrieval/router.py`

- `_judge_safely()`'s `except JudgeError as exc` branch now returns `JudgeVerdict(verdict="context_insufficient", raw_response=f"(judge error: {exc})")` instead of a bare verdict — failures become visible in the panel instead of silently disappearing.
- The router accumulates a `list[JudgeAttempt]` as it goes: one entry after the first judge call; a second entry appended only if the retry path actually runs. Each entry is `JudgeAttempt(attempt=1|2, verdict=verdict.verdict, raw_response=verdict.raw_response)`.
- This list is passed through unchanged on every response path (happy path and fallback path).

### 3. `app/retrieval/models.py`

New model:
```python
class JudgeAttempt(BaseModel):
    attempt: int
    verdict: str
    raw_response: str
```
`QueryResponse` gains `judge_attempts: list[JudgeAttempt]`.

### 4. Frontend (`app/static/query.html`, `app/static/js/query.js`, `app/static/css/app.css`)

- Layout becomes two-column: a `<div class="page-layout">` flex wrapper holds the existing content (query box, preferences, answer, filter toggle, chunks) as the main column, and a new `<aside id="judge-panel">` as a fixed-width side column (~280px, independently scrollable).
- `body`'s `max-width` widens to accommodate both columns (e.g. `1100px`).
- `renderJudgePanel(judgeAttempts)` (new function in `query.js`): clears the panel, then for each attempt renders a small block — "Attempt N: {verdict}" as a heading, the `raw_response` text below it via `textContent` (never `innerHTML` — same discipline as the rest of the file, since `raw_response` is LLM-generated text and not guaranteed safe to interpolate). Panel is empty before any query and fully replaced (not appended) on each new query.
- Wired into the existing `ask` button handler alongside `renderPreferences`/`renderFilteredNote`.

## Non-Goals

- No change to the judge's actual grading logic/prompt in this task — this is purely making existing behavior visible, a prerequisite step before redefining it.
- No persistence of judge history across queries — panel always reflects only the most recent query.
- No collapse/toggle affordance — panel is always visible per the earlier design discussion, simplest for a first pass.

## Testing Plan

- `tests/retrieval/test_judge.py` (extend): assert `JudgeVerdict.raw_response` is populated correctly on success, on the empty-chunks short-circuit, and that the raw content string matches what the mock Groq response returned.
- `tests/retrieval/test_router.py` (extend): assert `QueryResponse.judge_attempts` has exactly 1 entry on the happy-first-try path, exactly 2 entries (correctly ordered, attempt=1 then attempt=2) on the retry-then-succeed and retry-then-fallback paths, and that a `JudgeError`-forced attempt carries the `"(judge error: ...)"` marker text.
- `tests/frontend/test_query_page.py` (extend): assert `'id="judge-panel"'` is present in the served HTML.
- No JS-level automated test (documented, pre-existing exception in this repo) — manual/Playwright verification covers the panel rendering itself.
