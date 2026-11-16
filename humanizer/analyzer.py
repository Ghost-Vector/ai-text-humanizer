"""Local stylometric analysis: finds the patterns that make text read as machine-written.

Pure Python, no network. The score is a heuristic estimate built from the same kinds of
signals commercial detectors lean on (rhythm uniformity, stock vocabulary, formulaic
structure). It is not a GPTZero or Turnitin result and cannot predict one exactly.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter

WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])[\"'”’)\]]*\s+(?=[\"'“‘(\[]*[A-Z0-9])")
CODE_BLOCK_RE = re.compile(r"```.*?```", re.S)
LINE_PREFIX_RE = re.compile(r"^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s+)", re.M)

# (regex, label shown to the user and to the rewriter)
STOCK_VOCAB: list[tuple[str, str]] = [
    (r"delv(?:e|es|ed|ing)", "delve"),
    (r"tapestr(?:y|ies)", "tapestry"),
    (r"testament to", "a testament to"),
    (r"multifaceted", "multifaceted"),
    (r"pivotal", "pivotal"),
    (r"crucial(?:ly)?", "crucial"),
    (r"realms?", "realm"),
    (r"landscapes?", "landscape"),
    (r"intrica(?:te|cies|cy)", "intricate"),
    (r"seamless(?:ly)?", "seamless"),
    (r"robust", "robust"),
    (r"comprehensive", "comprehensive"),
    (r"leverag(?:e|es|ed|ing)", "leverage"),
    (r"foster(?:s|ed|ing)?", "foster"),
    (r"underscor(?:e|es|ed|ing)", "underscore"),
    (r"harness(?:es|ed|ing)?", "harness"),
    (r"navigat(?:e|es|ed|ing)", "navigate"),
    (r"embark(?:s|ed|ing)?", "embark"),
    (r"elevat(?:e|es|ed|ing)", "elevate"),
    (r"unlock(?:s|ed|ing)?", "unlock"),
    (r"unleash(?:es|ed|ing)?", "unleash"),
    (r"streamlin(?:e|es|ed|ing)", "streamline"),
    (r"bolster(?:s|ed|ing)?", "bolster"),
    (r"showcas(?:e|es|ed|ing)", "showcase"),
    (r"myriad", "myriad"),
    (r"plethora", "plethora"),
    (r"paramount", "paramount"),
    (r"nuanced?", "nuanced"),
    (r"vibrant", "vibrant"),
    (r"holistic(?:ally)?", "holistic"),
    (r"transformative", "transformative"),
    (r"groundbreaking", "groundbreaking"),
    (r"cutting-edge", "cutting-edge"),
    (r"ever-(?:evolving|changing|growing)", "ever-evolving"),
    (r"game-?changers?", "game-changer"),
    (r"synerg(?:y|ies)", "synergy"),
    (r"utiliz(?:e|es|ed|ing)", "utilize"),
    (r"facilitat(?:e|es|ed|ing)", "facilitate"),
    (r"endeavou?rs?", "endeavor"),
    (r"meticulous(?:ly)?", "meticulous"),
    (r"profound(?:ly)?", "profound"),
    (r"encompass(?:es|ed|ing)?", "encompass"),
    (r"resonat(?:e|es|ed|ing)", "resonate"),
    (r"cornerstones?", "cornerstone"),
    (r"interplay", "interplay"),
    (r"treasure trove", "treasure trove"),
    (r"shed(?:s|ding)? light", "shed light"),
    (r"it(?:'s|’s| is) (?:important|worth|crucial|essential) (?:to note|noting)", "it's important to note"),
    (r"worth noting that", "worth noting that"),
    (r"in today(?:'s|’s) [\w-]+(?: [\w-]+)? (?:world|age|era|landscape|society)", "in today's ... world"),
    (r"in conclusion", "in conclusion"),
    (r"in summary", "in summary"),
    (r"to sum up", "to sum up"),
    (r"plays? an? (?:crucial|vital|key|pivotal|significant|important) role", "plays a crucial role"),
    (r"when it comes to", "when it comes to"),
    (r"at its core", "at its core"),
    (r"that being said", "that being said"),
    (r"serves? as an?", "serves as a"),
    (r"stands? as an?", "stands as a"),
    (r"a wide (?:range|variety|array) of", "a wide range of"),
    (r"in order to", "in order to"),
    (r"due to the fact that", "due to the fact that"),
    (r"(?:let(?:'s|’s| us) )?(?:dive|diving) (?:in|into|deeper)", "dive into"),
    (r"i hope this helps", "I hope this helps"),
    (r"as an ai", "as an AI"),
]
STOCK_VOCAB_RE = [(re.compile(rf"\b(?:{p})\b", re.I), label) for p, label in STOCK_VOCAB]

TRANSITION_OPENERS = (
    "however", "moreover", "furthermore", "additionally", "in addition", "consequently",
    "therefore", "thus", "overall", "in conclusion", "ultimately", "nevertheless",
    "nonetheless", "notably", "importantly", "similarly", "in contrast", "as a result",
    "in essence", "essentially", "first and foremost", "firstly", "secondly", "lastly",
)
TRANSITION_RE = re.compile(rf"^(?:{'|'.join(TRANSITION_OPENERS)})\b[,:]?", re.I)

FORMULA_PATTERNS: list[tuple[str, str]] = [
    (r"\bnot only\b.{1,80}?\bbut also\b", "not only ... but also"),
    (r"\b(?:is|are|was|isn't|isn’t|aren't|aren’t)(?: not)? (?:just|merely|simply) .{1,70}?[,;—–-]+ ?(?:it(?:'s|’s| is)|they(?:'re|’re| are)|but)\b", "it's not just X, it's Y"),
    (r"\bwhether you(?:'re|’re| are)\b", "whether you're ..."),
    (r"\bhere(?:'s|’s| is) (?:the thing|why|what)\b", "here's the thing / here's why"),
    (r"\b(?:the|this) (?:result|answer|reason|truth|catch|problem|upside|bottom line)\? ", "question-then-answer reveal"),
    (r"\bby (?:\w+ing)\b.{1,60}?, (?:you|we|they|organizations|businesses|individuals) can\b", "by doing X, you can Y"),
    (r"\bmore than (?:just|merely) an?\b", "more than just a ..."),
]
FORMULA_RE = [(re.compile(p, re.I | re.S), label) for p, label in FORMULA_PATTERNS]

TRICOLON_RE = re.compile(r"\b[\w'’-]+(?: [\w'’-]+)?, [\w'’-]+(?: [\w'’-]+)?,? (?:and|or) [\w'’-]+")
CONTRACTED_RE = re.compile(r"\b\w+(?:n['’]t|['’](?:s|re|ve|ll|d|m))\b", re.I)
UNCONTRACTED_RE = re.compile(
    r"\b(?:do not|does not|did not|is not|are not|was not|were not|cannot|can not|will not|"
    r"would not|should not|could not|have not|has not|had not|it is|that is|there is|"
    r"they are|we are|you are|i am|i have|we have|they have|you will|we will|it will)\b",
    re.I,
)

WEIGHTS = {
    "rhythm": 22,
    "vocabulary": 24,
    "transitions": 12,
    "formulas": 8,
    "tricolons": 8,
    "contractions": 8,
    "em_dashes": 6,
    "paragraphs": 6,
    "openers": 6,
}


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 at lo, 1 at hi, linear in between (lo may be > hi for inverted scales)."""
    if lo == hi:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _clean(text: str) -> str:
    text = CODE_BLOCK_RE.sub(" ", text)
    return LINE_PREFIX_RE.sub("", text)


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for para in re.split(r"\n+", text):
        para = para.strip()
        if para:
            out.extend(s.strip() for s in SENTENCE_SPLIT_RE.split(para) if s.strip())
    return out


def analyze(text: str, formal: bool = False) -> dict:
    """Score `text` 0-100 for machine-writing patterns and list what was found.

    `formal=True` stops penalizing a lack of contractions (academic/professional tone).
    """
    prose = _clean(text)
    words = WORD_RE.findall(prose)
    n_words = len(words)
    sentences = [s for s in split_sentences(prose) if WORD_RE.search(s)]
    n_sent = len(sentences)
    paragraphs = [p for p in re.split(r"\n\s*\n", prose) if len(WORD_RE.findall(p)) >= 8]

    if n_words < 40 or n_sent < 3:
        return {
            "score": None,
            "verdict": "Too short to analyze",
            "reliable": False,
            "words": n_words,
            "sentences": n_sent,
            "signals": {},
            "findings": [],
        }

    per_100 = 100.0 / n_words
    signals: dict[str, float] = {}
    findings: list[dict] = []

    def add(key: str, strength: float, title: str, detail: str, examples: list[str] | None = None):
        signals[key] = round(strength, 3)
        if strength >= 0.25:
            findings.append(
                {
                    "id": key,
                    "title": title,
                    "severity": "high" if strength >= 0.66 else "medium",
                    "detail": detail,
                    "examples": examples or [],
                }
            )

    # Rhythm: humans vary sentence length a lot; models cluster around the mean.
    lengths = [len(WORD_RE.findall(s)) for s in sentences]
    mean_len = statistics.fmean(lengths)
    cv = statistics.pstdev(lengths) / mean_len if mean_len else 0.0
    short_share = sum(1 for n in lengths if n <= 7) / n_sent
    rhythm = 0.75 * _ramp(cv, 0.62, 0.28) + 0.25 * _ramp(short_share, 0.12, 0.0)
    add(
        "rhythm",
        rhythm,
        "Uniform sentence rhythm",
        f"Sentences average {mean_len:.0f} words and barely vary (variation {cv:.2f}; "
        f"human prose usually sits above 0.55). {short_share:.0%} are short (7 words or fewer).",
    )

    # Stock vocabulary.
    hits: Counter[str] = Counter()
    for rx, label in STOCK_VOCAB_RE:
        n = len(rx.findall(prose))
        if n:
            hits[label] += n
    vocab_rate = sum(hits.values()) * per_100
    add(
        "vocabulary",
        _ramp(vocab_rate, 0.15, 1.8),
        "Stock AI vocabulary",
        f"{sum(hits.values())} stock words or phrases ({vocab_rate:.1f} per 100 words).",
        [f"{w} ×{c}" if c > 1 else w for w, c in hits.most_common(14)],
    )

    # Formal transition openers.
    openers = [m.group(0).rstrip(",:").lower() for s in sentences if (m := TRANSITION_RE.match(s))]
    trans_share = len(openers) / n_sent
    add(
        "transitions",
        _ramp(trans_share, 0.04, 0.2),
        "Formal transition openers",
        f"{len(openers)} of {n_sent} sentences open with a signpost word.",
        [f"{w} ×{c}" if c > 1 else w for w, c in Counter(openers).most_common(8)],
    )

    # Formulaic constructions.
    formula_hits: Counter[str] = Counter()
    for rx, label in FORMULA_RE:
        n = len(rx.findall(prose))
        if n:
            formula_hits[label] += n
    add(
        "formulas",
        _ramp(sum(formula_hits.values()) / n_sent, 0.0, 0.12),
        "Formulaic constructions",
        f"{sum(formula_hits.values())} template-style sentence patterns.",
        list(formula_hits),
    )

    # Lists of three.
    tricolons = TRICOLON_RE.findall(prose)
    add(
        "tricolons",
        _ramp(len(tricolons) / n_sent, 0.08, 0.35),
        "Reflexive lists of three",
        f"{len(tricolons)} 'X, Y, and Z' groupings in {n_sent} sentences.",
        [t.strip() for t in tricolons[:4]],
    )

    # Contractions.
    contracted = len(CONTRACTED_RE.findall(prose))
    uncontracted = len(UNCONTRACTED_RE.findall(prose))
    if formal or uncontracted + contracted < 3:
        signals["contractions"] = 0.0
    else:
        add(
            "contractions",
            _ramp(uncontracted / (uncontracted + contracted), 0.5, 0.95),
            "No contractions",
            f"{uncontracted} spelled-out forms (do not, it is) against {contracted} contractions.",
        )

    # Em dashes.
    dashes = len(re.findall(r"—|\s–\s|\s--\s", prose))
    add(
        "em_dashes",
        _ramp(dashes * per_100, 0.2, 1.0),
        "Heavy em-dash use",
        f"{dashes} em dashes ({dashes * per_100:.1f} per 100 words).",
    )

    # Paragraph symmetry.
    if len(paragraphs) >= 3:
        plens = [len(WORD_RE.findall(p)) for p in paragraphs]
        pcv = statistics.pstdev(plens) / statistics.fmean(plens)
        add(
            "paragraphs",
            _ramp(pcv, 0.4, 0.12),
            "Evenly sized paragraphs",
            f"{len(paragraphs)} paragraphs of nearly equal length (variation {pcv:.2f}).",
        )
    else:
        signals["paragraphs"] = 0.0

    # Repeated sentence openers.
    firsts = Counter(WORD_RE.findall(s)[0].lower() for s in sentences)
    top_word, top_n = firsts.most_common(1)[0]
    if n_sent >= 6:
        add(
            "openers",
            _ramp(top_n / n_sent, 0.2, 0.4),
            "Repeated sentence openers",
            f"{top_n} of {n_sent} sentences start with “{top_word.capitalize()}”.",
        )
    else:
        signals["openers"] = 0.0

    total_weight = sum(w for k, w in WEIGHTS.items() if not (formal and k == "contractions"))
    raw = sum(WEIGHTS[k] * v for k, v in signals.items()) / total_weight
    # Several strong signals together are more damning than their average suggests.
    strong = sum(1 for v in signals.values() if v >= 0.66)
    score = round(min(100.0, raw * 100 * (1 + 0.18 * strong)))

    verdict = (
        "Reads as machine-written" if score >= 55
        else "Mixed signals" if score >= 28
        else "Reads as human-written"
    )
    findings.sort(key=lambda f: -signals[f["id"]] * WEIGHTS[f["id"]])
    return {
        "score": score,
        "verdict": verdict,
        "reliable": n_words >= 120,
        "words": n_words,
        "sentences": n_sent,
        "signals": signals,
        "findings": findings,
    }
