"""Step 5 (half one) — the answer model, behind one function.

    from src import llm
    out = llm.complete("some prompt")        # -> {"text": ..., "cached": ...}

Every provider-specific detail lives in this module, the same way every
Chroma-specific call lives in src/store.py. Swapping Groq for Gemini is a
second branch in _call(), not a change to the pipeline.

Three things here are not decoration, they are consequences of measurements
taken against the live API on 2026-10-17:

1. CACHING. The free tier allows 8,000 tokens per MINUTE (derived from the
   x-ratelimit headers: 60,000ms / 8,000 = the 7.5ms-per-token refill rate
   observed). A RAG prompt is ~2,000 tokens, so roughly three calls a minute.
   Step 7 re-runs the same question set every time retrieval changes
   underneath; uncached that is ~19 minutes of sleeping per run. Cached, a
   re-run is free and instant. The cache is what makes Phase 2 affordable.

2. HEADER-DRIVEN THROTTLING. The API tells us what is left and when it
   refills. Reading that and waiting is strictly better than discovering the
   limit by being refused.

3. A TOKEN BUDGET BIG ENOUGH FOR REASONING. gpt-oss-120b is a reasoning
   model: it spends completion tokens thinking before it writes. Asked for a
   three-word answer with max_tokens=20 it returned an EMPTY STRING, having
   spent all 20 on hidden reasoning. A silent empty answer is the worst kind
   of bug, so _call() raises instead of returning one.
"""

import hashlib
import json
import os
import re
import time

from dotenv import load_dotenv

from src.config import (GEMINI_MODEL, LLM_CACHE as CACHE, MAX_OUTPUT_TOKENS,
                        MAX_RETRIES, MODEL, MODELS, REASONING_EFFORT, ROOT,
                        TEMPERATURE, TPM_SAFETY_MARGIN)

load_dotenv(ROOT / ".env")

_budget = {"remaining_tokens": None, "reset_seconds": 0.0, "stamp": 0.0}


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #

def _cache_key(payload: dict) -> str:
    """Keyed on everything that can change the answer, nothing that cannot."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def _payload(prov: str, model: str, system: str, prompt: str,
             max_tokens: int, temperature: float,
             response_format: dict | None) -> dict:
    """Everything that can change the answer, and nothing that cannot.

    `response_format` is added only when present, which is what keeps every
    answer cached before structured output existed readable instead of
    orphaning all of them.
    """
    payload = {"provider": prov, "model": model, "system": system,
               "prompt": prompt, "max_tokens": max_tokens,
               "temperature": temperature,
               "reasoning_effort": REASONING_EFFORT}
    if response_format is not None:
        payload["response_format"] = response_format
    return payload


def is_cached(prompt: str, system: str = "You are a helpful assistant.",
              model: str | None = None, max_tokens: int = MAX_OUTPUT_TOKENS,
              temperature: float = TEMPERATURE,
              response_format: dict | None = None) -> bool:
    """Would `complete` answer this from disk? Reads nothing but the filename.

    Used by the UI to mark a question as free before anyone clicks it. It
    touches no network and does not read the cached body, so it stays cheap
    enough to run across every question on a settings change.
    """
    prov = provider()
    model = model or (MODEL if prov == "groq" else GEMINI_MODEL)
    key = _cache_key(_payload(prov, model, system, prompt, max_tokens,
                              temperature, response_format))
    return (CACHE / f"{key}.json").exists()


def _cache_read(key: str) -> dict | None:
    f = CACHE / f"{key}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except json.JSONDecodeError:
        return None          # a truncated write is a miss, not a crash


def _cache_write(key: str, record: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / f"{key}.json.tmp"
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    tmp.rename(CACHE / f"{key}.json")      # atomic: no half-written cache files


# --------------------------------------------------------------------------- #
# rate limiting
# --------------------------------------------------------------------------- #

_DUR_RE = re.compile(r"(?:(\d+(?:\.\d+)?)m(?!s))?\s*(?:(\d+(?:\.\d+)?)s)?"
                     r"\s*(?:(\d+(?:\.\d+)?)ms)?$")


def parse_duration(s: str) -> float:
    """Groq reports resets as '1m26.4s', '915ms', '7.5s'. Return seconds."""
    m = _DUR_RE.match((s or "").strip())
    if not m:
        return 0.0
    mins, secs, millis = (float(g) if g else 0.0 for g in m.groups())
    return mins * 60 + secs + millis / 1000


# Groq states the wait inside the error message rather than only in a header,
# e.g. "Rate limit reached ... Please try again in 48.523s". parse_duration is
# anchored, because it parses a header whose whole value is a duration, so it
# returns 0 for a duration embedded in a sentence. This searches instead.
_EMBEDDED_DUR_RE = re.compile(
    r"(?:try again in|retry after|in)\s+"
    r"(?:(\d+(?:\.\d+)?)m(?!s))?\s*(?:(\d+(?:\.\d+)?)s)?", re.I)


def find_duration(s: str) -> float:
    """Pull a wait out of a sentence. Returns seconds, or 0.0 if absent."""
    m = _EMBEDDED_DUR_RE.search(s or "")
    if not m or not any(m.groups()):
        return 0.0
    mins, secs = (float(g) if g else 0.0 for g in m.groups())
    return mins * 60 + secs


# A 429 on a per-minute token bucket is not an error to back off from in
# milliseconds; the bucket refills on a schedule, so the floor is seconds.
RATE_LIMIT_FLOOR_S = 20.0
RATE_LIMIT_CEILING_S = 70.0


def _wait_for_budget(need: int) -> float:
    """Sleep if the bucket cannot cover this call. Returns seconds slept."""
    rem = _budget["remaining_tokens"]
    if rem is None or rem >= need + TPM_SAFETY_MARGIN:
        return 0.0
    elapsed = time.time() - _budget["stamp"]
    sleep = max(0.0, _budget["reset_seconds"] - elapsed)
    if sleep > 0:
        print(f"    [rate limit] {rem} tokens left, need ~{need}; "
              f"waiting {sleep:.1f}s for refill")
        time.sleep(sleep)
    _budget["remaining_tokens"] = None     # unknown again until the next call
    return sleep


def _record_budget(headers) -> None:
    try:
        rem = headers.get("x-ratelimit-remaining-tokens")
        reset = headers.get("x-ratelimit-reset-tokens")
        if rem is not None:
            _budget["remaining_tokens"] = int(float(rem))
            _budget["reset_seconds"] = parse_duration(reset)
            _budget["stamp"] = time.time()
    except (TypeError, ValueError):
        pass          # headers are an optimisation; never fail a call over them


# --------------------------------------------------------------------------- #
# providers
# --------------------------------------------------------------------------- #

def provider() -> str:
    """Chosen by which key is present. Groq wins if both are set."""
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    raise SystemExit(
        "No answer-model key found. Put GROQ_API_KEY=... in .env "
        "(free tier: console.groq.com) or GEMINI_API_KEY=... "
        "(aistudio.google.com)."
    )


def _groq_call(prompt: str, system: str, model: str, max_tokens: int,
               temperature: float,
               response_format: dict | None = None) -> dict:
    """One call. `response_format` is Groq's structured-output parameter.

    With {"type": "json_schema", ...} and strict=True the schema is enforced by
    constrained decoding, so the model CANNOT emit anything else -- which is
    the point. Probed against qwen/qwen3.8-27b on 2026-10-20: supported.
    """
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if MODELS.get(model, {}).get("reasoning"):
        kwargs["reasoning_effort"] = REASONING_EFFORT
    if response_format is not None:
        kwargs["response_format"] = response_format

    raw = client.chat.completions.with_raw_response.create(**kwargs)
    _record_budget(raw.headers)
    completion = raw.parse()
    choice = completion.choices[0]
    text = (choice.message.content or "").strip()

    if not text:
        # The empty-answer trap: all completion tokens went to reasoning.
        raise RuntimeError(
            f"{model} returned an empty answer (finish_reason="
            f"{choice.finish_reason!r}, completion_tokens="
            f"{completion.usage.completion_tokens}). The token budget was "
            f"spent on reasoning. Raise max_tokens or lower REASONING_EFFORT."
        )
    return {
        "text": text,
        "prompt_tokens": completion.usage.prompt_tokens,
        "completion_tokens": completion.usage.completion_tokens,
        "finish_reason": choice.finish_reason,
    }


def _gemini_call(prompt: str, system: str, model: str, max_tokens: int,
                 temperature: float,
                 response_format: dict | None = None) -> dict:
    """Second provider, so the pipeline is not welded to one vendor.

    UNTESTED: written against the google-genai API but never run, because no
    GEMINI_API_KEY was available on 2026-10-17. Treat it as a stub until a key
    exists and this path is exercised.

    `response_format` is accepted and IGNORED: Gemini spells structured output
    differently (response_mime_type plus response_schema). Accepting it keeps
    the two providers call-compatible, but anything depending on the schema
    must stay on Groq until this is written properly.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError(f"{model} returned an empty answer")
    usage = getattr(resp, "usage_metadata", None)
    return {
        "text": text,
        "prompt_tokens": getattr(usage, "prompt_token_count", 0) or 0,
        "completion_tokens": getattr(usage, "candidates_token_count", 0) or 0,
        "finish_reason": "stop",
    }


# --------------------------------------------------------------------------- #
# the one public function
# --------------------------------------------------------------------------- #

def complete(prompt: str, system: str = "You are a helpful assistant.",
             model: str | None = None, max_tokens: int = MAX_OUTPUT_TOKENS,
             temperature: float = TEMPERATURE,
             use_cache: bool = True,
             response_format: dict | None = None) -> dict:
    """Prompt in, answer out. Cached, throttled, and never silently empty.

    Returns {text, cached, latency_s, prompt_tokens, completion_tokens, model}.

    `response_format` asks the provider for structured output. It changes the
    answer, so it belongs in the cache key -- and it is added to that key only
    when present, which is what keeps every answer cached before this existed
    readable instead of orphaning all of them.
    """
    prov = provider()
    model = model or (MODEL if prov == "groq" else GEMINI_MODEL)

    key = _cache_key(_payload(prov, model, system, prompt, max_tokens,
                              temperature, response_format))

    if use_cache:
        hit = _cache_read(key)
        if hit is not None:
            return {**hit["response"], "cached": True, "latency_s": 0.0,
                    "model": model}

    # ~4 characters per token is the usual rule of thumb; this only sizes the
    # throttle wait, so being approximate is fine.
    need = len(prompt) // 4 + len(system) // 4 + max_tokens
    call = _groq_call if prov == "groq" else _gemini_call

    last_err = None
    for attempt in range(MAX_RETRIES):
        _wait_for_budget(need)
        t0 = time.time()
        try:
            out = call(prompt, system, model, max_tokens, temperature,
                       response_format)
            latency = time.time() - t0
            if use_cache:
                _cache_write(key, {
                    "provider": prov, "model": model, "system": system,
                    "prompt": prompt, "response": out,
                    "latency_s": round(latency, 3),
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
            return {**out, "cached": False, "latency_s": latency, "model": model}
        except Exception as e:
            last_err = e
            status = getattr(e, "status_code", None)
            if status not in (429, 500, 502, 503) and "rate" not in str(e).lower():
                raise
            # A per-minute bucket refills in under a minute and is worth
            # retrying. A per-DAY bucket is not: it refills at 200,000/24h, so
            # retrying burns four waits and then fails anyway, which is exactly
            # what it did -- 193 seconds of sleeping to arrive at the same
            # place. Fail immediately and say what is actually wrong.
            if "per day" in str(e).lower() or "tpd" in str(e).lower():
                raise RuntimeError(
                    "Groq daily token limit reached (TPD). Retrying cannot "
                    "clear this; the quota refills over 24 hours. Cached "
                    "answers still work, and retrieval-only evaluation "
                    "(--no-llm) makes no API calls at all.\n"
                    f"  provider message: {e}") from e
            # Groq states the wait in the error BODY ("Please try again in
            # 48.5s"), not always as a retry-after header, so the header-only
            # version fell back to 1, 2, 4, 8 seconds -- 15 seconds in total
            # against a token bucket that needs up to a minute to refill. The
            # eval then died on its last question with quota to spare.
            hdr = (getattr(e, "response", None)
                   and e.response.headers.get("retry-after", "")) or ""
            wait = (parse_duration(str(hdr))
                    or find_duration(str(e))
                    or (RATE_LIMIT_FLOOR_S * (attempt + 1) if status == 429
                        else 2 ** attempt))
            wait = min(wait + 1.0, RATE_LIMIT_CEILING_S)   # +1s of headroom
            print(f"    [retry {attempt+1}/{MAX_RETRIES}] {type(e).__name__}; "
                  f"sleeping {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"giving up after {MAX_RETRIES} attempts: {last_err}")


def cache_stats() -> dict:
    files = list(CACHE.glob("*.json")) if CACHE.exists() else []
    return {"entries": len(files),
            "bytes": sum(f.stat().st_size for f in files)}
