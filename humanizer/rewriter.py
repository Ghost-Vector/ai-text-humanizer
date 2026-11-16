"""Claude-powered rewrite loop: rewrite, re-analyze, and revise until the patterns are gone."""

from __future__ import annotations

import os
import re
from typing import AsyncIterator

import anthropic
import httpx
import ollama

from .analyzer import analyze

# Provider: Ollama when OLLAMA_HOST or OLLAMA_API_KEY is set, otherwise Anthropic.
# OLLAMA_HOST=http://127.0.0.1:11434 runs a model on this machine (free, no key);
# OLLAMA_HOST=https://ollama.com uses Ollama Cloud and needs OLLAMA_API_KEY.
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "").strip()
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com" if OLLAMA_API_KEY else "").strip()
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "kimi-k3")
OLLAMA_IS_CLOUD = "ollama.com" in OLLAMA_HOST
# Local models default to a small context window that would cut long texts off.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
# Hugging Face Inference Providers (OpenAI-compatible router).
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
HF_MODEL = os.environ.get("HF_MODEL", "moonshotai/Kimi-K3")
HF_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = os.environ.get("HUMANIZER_MODEL", "claude-opus-5-5")
# HUMANIZER_PROVIDER = huggingface | ollama | anthropic. Left unset, the first one configured wins.
PROVIDER = os.environ.get("HUMANIZER_PROVIDER", "").strip().lower() or (
    "huggingface" if HF_TOKEN else "ollama" if OLLAMA_HOST else "anthropic"
)
# Models that accept the server-side refusal fallback parameter.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-2", "claude-fable-5-1", "claude-sonnet-5-5"}
TARGET_SCORE = 18
MAX_PASSES = 3

SYSTEM_PROMPT = """You are an editor who rewrites stiff, machine-sounding drafts so they read the way a thoughtful person actually writes. You receive a draft, plus a stylometric report listing the specific patterns that make it read as generated. Your job is a real rewrite: the same content, said the way a person who knows the subject would say it.

What must not change
- Every fact, number, name, date, quotation, citation, URL and code block stays exactly as given. Do not add claims, examples, anecdotes, statistics or sources that are not in the draft, and do not drop any point the draft makes.
- Keep the language of the draft, its point of view (first/third person), and any headings, lists or other formatting it uses.
- Stay close to the original length. Cutting filler is fine; padding is not.

Why drafts read as generated, and what to do instead
- Rhythm. Generated text has sentences of similar length marching in step, and paragraphs of equal size. People don't write like that. Let length follow the thought: a short sentence when the point is simple, a long one when an idea needs room to unfold, and paragraphs that end when the idea ends rather than at a tidy four sentences. Aim for real spread, with several sentences under eight words in any longer piece.
- Structure. Rebuild sentences rather than swapping synonyms. Change what the sentence starts with, merge two, split one, reorder a paragraph so it leads with the concrete thing. Synonym swaps leave the skeleton intact and the skeleton is what gives the draft away.
- Vocabulary. Use the plain word a person would reach for in conversation. Stock words such as delve, crucial, pivotal, landscape, realm, tapestry, leverage, foster, robust, seamless, comprehensive, navigate, underscore, multifaceted and their relatives should go, along with filler such as "it's important to note", "in today's world", "plays a key role" and "when it comes to".
- Connectives. Drop most sentence-opening signposts (However, Moreover, Furthermore, Additionally, In conclusion, Overall). Use and, but, so, still, though, or nothing at all; a well-ordered paragraph rarely needs to announce its logic.
- Templates. Avoid "not only X but also Y", "it's not just X, it's Y", a question immediately answered by the writer, automatic groups of three, and a closing paragraph that restates everything. If the draft ends on a summary that adds nothing, end on the last real point.
- Punctuation. No em dashes; use a comma, a full stop or parentheses.
- Voice. Commit to statements instead of hedging each one. Where the tone allows, use contractions and address the reader directly.

Do not fake humanity with typos, grammar mistakes, slang that doesn't fit, or invented personal stories. Clean, natural prose is the goal.

Output only the rewritten text, with no preface, notes or explanation."""

TONES = {
    "match": "Keep the register of the original draft (formal stays formal, casual stays casual), just make it sound like a person wrote it.",
    "casual": "Write in a relaxed, conversational register, as if explaining this to a friend. Contractions throughout.",
    "professional": "Write in a clear, direct professional register, the way a competent colleague writes an email or report. Contractions are fine where natural.",
    "academic": "Write in an academic register suited to a student essay or paper: precise and measured, no contractions, no direct address, but still plain-spoken and varied in rhythm rather than stilted.",
}

INTENSITIES = {
    "light": "Make a light pass: fix the listed patterns and smooth the rhythm, but keep the original sentence order and most of its wording where it is already natural.",
    "balanced": "Rewrite freely at the sentence level and reorder within paragraphs where it helps, keeping the overall paragraph sequence.",
    "deep": "Rewrite from scratch: take the content as notes and write the piece again in your own structure, reordering and regrouping paragraphs wherever that reads better.",
}


# Small models running on this machine ignore the long editorial brief above and mostly copy
# the draft back. A short numbered rule list with an explicit banned-word list works far better.
IS_LOCAL = PROVIDER == "ollama" and not OLLAMA_IS_CLOUD

LOCAL_SYSTEM_PROMPT = """You rewrite stiff, machine-sounding text so it reads like a real person wrote it. Follow every rule.

1. Keep every fact, number and name exactly. Add nothing new. Keep roughly the same length and the same language.
2. Never use the banned words and phrases listed by the user. Use a plain everyday word instead, or cut the phrase.
3. Mix sentence lengths. At least three sentences must be very short (3 to 6 words). At least one must be long (25 words or more).
4. Never start a sentence with However, Moreover, Furthermore, Additionally, Overall, Ultimately or In conclusion.
5. Never write a list of three things ("A, B, and C"). Name one or two things, or spread them over separate sentences.
6. Make the paragraphs clearly different in length. One paragraph should be a single sentence.
7. {voice}
8. No dashes. No "not only ... but also". No "it's not just X, it's Y".
9. {depth}

Output only the rewritten text."""

LOCAL_VOICE = {
    "match": "Use plain words a person would say out loud, with contractions (it's, don't) unless the text is formal.",
    "casual": "Write casually, like talking to a friend. Use contractions (it's, don't, can't).",
    "professional": "Write like a clear, direct colleague. Plain words. Contractions are fine.",
    "academic": "Write in a formal academic style with plain, precise words. Do not use contractions.",
}
LOCAL_DEPTH = {
    "light": "Fix the problems listed but keep wording that already sounds natural.",
    "balanced": "Change how the sentences are built. Do not just swap words.",
    "deep": "Rebuild every sentence from scratch and reorder ideas inside each paragraph.",
}


def _local_prompt(draft: str, analysis: dict) -> str:
    banned: list[str] = []
    for f in analysis["findings"]:
        if f["id"] in ("vocabulary", "transitions", "formulas"):
            banned += [re.sub(r" ×\d+$", "", e) for e in f["examples"]]
    problems = "\n".join(f"- {f['title']}: {f['detail']}" for f in analysis["findings"]) or "- none"
    return (
        f"Banned words and phrases: {', '.join(dict.fromkeys(banned)) or 'none found'}\n\n"
        f"Problems found in this text:\n{problems}\n\n"
        f"Rewrite this text following every rule:\n\n{draft}"
    )


class HumanizerError(Exception):
    """An error with a message that is safe to show to the user."""


_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        try:
            _client = anthropic.AsyncAnthropic()
        except anthropic.AnthropicError as e:
            raise HumanizerError(
                "No Anthropic credentials found. Put ANTHROPIC_API_KEY=... in a .env file "
                "next to server.py (or set it in your environment) and restart."
            ) from e
    return _client


def _report(analysis: dict) -> str:
    if not analysis["findings"]:
        return "No strong patterns detected; focus on natural rhythm and plain wording."
    lines = []
    for f in analysis["findings"]:
        line = f"- {f['title']} ({f['severity']}): {f['detail']}"
        if f["examples"]:
            line += " Found: " + "; ".join(f["examples"]) + "."
        lines.append(line)
    return "\n".join(lines)


async def _call(messages: list[dict], system: str = SYSTEM_PROMPT) -> str:
    if PROVIDER == "huggingface":
        return await _call_huggingface(messages, system)
    if PROVIDER == "ollama":
        return await _call_ollama(messages, system)
    return await _call_anthropic(messages, system)


async def _call_huggingface(messages: list[dict], system: str) -> str:
    if not HF_TOKEN:
        raise HumanizerError("No Hugging Face token set. Put HF_TOKEN=hf_... in .env and restart.")
    payload = {"model": HF_MODEL, "messages": [{"role": "system", "content": system}, *messages]}
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            r = await client.post(HF_URL, json=payload, headers={"Authorization": f"Bearer {HF_TOKEN}"})
    except httpx.HTTPError as e:
        raise HumanizerError("Could not reach Hugging Face. Check your internet connection.") from e

    if r.status_code != 200:
        try:
            err = r.json().get("error")
            detail = err.get("message") if isinstance(err, dict) else str(err)
        except ValueError:
            detail = r.text[:200]
        if r.status_code == 401:
            raise HumanizerError("The Hugging Face token was rejected. Check HF_TOKEN in .env.")
        if r.status_code == 402:
            raise HumanizerError(
                "Your Hugging Face inference credits are used up. Add credits at "
                "huggingface.co/settings/billing or wait for the monthly reset."
            )
        if r.status_code == 403:
            raise HumanizerError(
                "This Hugging Face token can't call Inference Providers. Create a token with the "
                f"'Make calls to Inference Providers' permission. ({detail})"
            )
        if r.status_code == 429:
            raise HumanizerError("Hugging Face rate limit reached. Wait a minute and try again.")
        raise HumanizerError(f"Hugging Face error ({r.status_code}): {detail}")

    try:
        content = r.json()["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise HumanizerError("Hugging Face returned an unexpected response. Try again.") from e
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    if not text:
        raise HumanizerError("The model returned no text. Try again.")
    return text


async def _call_ollama(messages: list[dict], system: str) -> str:
    headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_IS_CLOUD else {}
    client = ollama.AsyncClient(host=OLLAMA_HOST, headers=headers, timeout=1800)
    parts: list[str] = []
    try:
        stream = await client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "system", "content": system}, *messages],
            stream=True,
            options=None if OLLAMA_IS_CLOUD else {"num_ctx": OLLAMA_NUM_CTX},
            # On CPU, a local model's reasoning phase can run for an hour; skip it.
            think=None if OLLAMA_IS_CLOUD else False,
        )
        async for chunk in stream:
            if chunk.message.content:
                parts.append(chunk.message.content)
    except ollama.ResponseError as e:
        if e.status_code == 401:
            raise HumanizerError("The Ollama API key was rejected. Check OLLAMA_API_KEY in .env.") from e
        if e.status_code == 404:
            raise HumanizerError(f"Ollama model '{OLLAMA_MODEL}' was not found. Check OLLAMA_MODEL in .env.") from e
        if e.status_code in (402, 403):
            raise HumanizerError(f"Your Ollama plan can't use '{OLLAMA_MODEL}': {e.error}") from e
        if e.status_code == 429:
            raise HumanizerError("Ollama usage limit reached. Wait a while and try again.") from e
        raise HumanizerError(f"Ollama API error ({e.status_code}): {e.error}") from e
    except (httpx.HTTPError, ConnectionError) as e:
        if OLLAMA_IS_CLOUD:
            raise HumanizerError("Could not reach the Ollama API. Check your internet connection.") from e
        raise HumanizerError(
            f"Could not reach Ollama at {OLLAMA_HOST}, or the model took too long. "
            "Make sure the Ollama app is running."
        ) from e

    # Some models put reasoning inline; keep only the rewrite itself.
    text = re.sub(r"<think>.*?</think>", "", "".join(parts), flags=re.S).strip()
    if not text:
        raise HumanizerError("The model returned no text. Try again.")
    return text


async def _call_anthropic(messages: list[dict], system: str) -> str:
    kwargs: dict = dict(
        model=MODEL,
        max_tokens=64000,
        system=system,
        output_config={"effort": "high"},
        messages=messages,
    )
    if MODEL in FALLBACK_MODELS:
        kwargs["betas"] = ["server-side-fallback-2026-07-01"]
        kwargs["fallbacks"] = "default"
    try:
        async with _get_client().beta.messages.stream(**kwargs) as stream:
            message = await stream.get_final_message()
    except anthropic.AuthenticationError as e:
        raise HumanizerError("The Anthropic API key was rejected. Check ANTHROPIC_API_KEY in .env.") from e
    except anthropic.PermissionDeniedError as e:
        raise HumanizerError(f"This API key can't use {MODEL}: {e.message}") from e
    except anthropic.NotFoundError as e:
        raise HumanizerError(f"Model '{MODEL}' was not found. Check HUMANIZER_MODEL.") from e
    except anthropic.RateLimitError as e:
        raise HumanizerError("Rate limited by the Anthropic API. Wait a minute and try again.") from e
    except anthropic.BadRequestError as e:
        if "credit balance" in str(e):
            raise HumanizerError(
                "Your Anthropic account has no API credits. Add credits under Plans & Billing "
                "at console.anthropic.com, then try again."
            ) from e
        raise HumanizerError(f"The API rejected the request: {e.message}") from e
    except anthropic.APIStatusError as e:
        raise HumanizerError(f"Anthropic API error ({e.status_code}). Try again shortly.") from e
    except anthropic.APIConnectionError as e:
        raise HumanizerError("Could not reach the Anthropic API. Check your internet connection.") from e
    except (anthropic.AnthropicError, TypeError) as e:
        # A missing key surfaces here as an auth/TypeError when the first request is built.
        raise HumanizerError(
            "Could not authenticate with the Anthropic API. Set ANTHROPIC_API_KEY in .env and restart."
        ) from e

    if message.stop_reason == "refusal":
        raise HumanizerError("The model declined to rewrite this text.")
    text = "".join(b.text for b in message.content if b.type == "text").strip()
    if not text:
        raise HumanizerError("The model returned no text. Try again.")
    if message.stop_reason == "max_tokens":
        raise HumanizerError("The text is too long to rewrite in one go. Split it into sections.")
    return text


async def humanize(text: str, tone: str = "match", intensity: str = "balanced") -> AsyncIterator[dict]:
    """Yield progress events, ending with a `done` event carrying the best rewrite."""
    tone = tone if tone in TONES else "match"
    intensity = intensity if intensity in INTENSITIES else "balanced"
    formal = tone == "academic"

    before = analyze(text, formal=formal)
    yield {"type": "analysis", "before": before}

    instructions = f"Tone: {TONES[tone]}\nDepth: {INTENSITIES[intensity]}"
    first_prompt = (
        f"{instructions}\n\nStylometric report on the draft:\n{_report(before)}\n\n"
        f"<draft>\n{text}\n</draft>"
    )

    best_text, best = "", None
    draft, draft_analysis = text, before
    passes: list[dict] = []

    for n in range(1, MAX_PASSES + 1):
        yield {"type": "pass_start", "n": n}
        system = SYSTEM_PROMPT
        if IS_LOCAL:
            # Each pass edits the latest draft only; given the original too, small models copy it.
            system = LOCAL_SYSTEM_PROMPT.format(voice=LOCAL_VOICE[tone], depth=LOCAL_DEPTH[intensity])
            prompt = _local_prompt(draft, draft_analysis)
        elif n == 1:
            prompt = first_prompt
        else:
            prompt = (
                f"{instructions}\n\nYour previous rewrite is below. A re-check still finds these "
                f"patterns in it:\n{_report(draft_analysis)}\n\nRevise the rewrite to remove them. "
                "Keep what already reads naturally, and check facts against the original draft.\n\n"
                f"<original_draft>\n{text}\n</original_draft>\n\n<previous_rewrite>\n{draft}\n</previous_rewrite>"
            )
        draft = await _call([{"role": "user", "content": prompt}], system)
        draft_analysis = analyze(draft, formal=formal)
        score = draft_analysis["score"]
        passes.append({"n": n, "score": score})
        yield {"type": "pass_done", "n": n, "score": score}

        if best is None or score is None or (best["score"] is not None and score < best["score"]):
            best_text, best = draft, draft_analysis
        if score is None or score <= TARGET_SCORE:
            break

    yield {"type": "done", "output": best_text, "before": before, "after": best, "passes": passes}
