# Humanizer

Finds the patterns that make text read as machine-written, then rewrites it the way a person would say it.

## Setup

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env   # then configure one provider in .env (see Options)
python server.py
```

Open http://127.0.0.1:8000.

## How it works

1. **Analyze** (`humanizer/analyzer.py`, local, no API call). Scores the text 0-100 on nine signals: sentence-length variation, stock vocabulary, formal transition openers, template constructions, lists of three, contractions, em dashes, paragraph symmetry and repeated sentence openers.
2. **Rewrite** (`humanizer/rewriter.py`). Sends the text and the list of findings to the model (Kimi on Ollama Cloud, or Claude) with instructions to rebuild sentences rather than swap synonyms, keeping every fact, number, name and quote.
3. **Re-check and revise.** The rewrite is analyzed again. If the score is still above 18, the remaining findings go back for a revision pass, up to three passes. The lowest-scoring draft is returned.

**Analyze only** runs step 1 and needs no API key.

## Options

- **Tone**: match the original, casual, professional, or academic (academic does not use contractions and is not penalized for it).
- **Rewrite depth**: light touch, balanced, or deep rewrite (restructures paragraphs).
- **Provider**: the rewrite model can come from Hugging Face (`HF_TOKEN`, `HF_MODEL`), Ollama on this machine or Ollama Cloud (`OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_API_KEY`), or Anthropic (`ANTHROPIC_API_KEY`, `HUMANIZER_MODEL`). See `.env.example`. If more than one is configured, set `HUMANIZER_PROVIDER` to `huggingface`, `ollama` or `anthropic`; otherwise the first configured in that order is used.

### Local models

With `OLLAMA_HOST=http://127.0.0.1:11434` the rewrite runs on a model on this machine: free, no key, but slower and weaker than a large cloud model. Local models get a shorter rule-list prompt with an explicit banned-word list, and their reasoning phase is turned off (on CPU it can run for an hour). On a 16 GB laptop without a large GPU, `gemma4:e4b` takes about 1 to 2 minutes per pass for a 150-word text.

## Limits

- Small local models drift: they can shift the tone toward casual and loosen the meaning of a sentence. Compare the rewrite against the original.

- The pattern score is a local estimate. It is not a GPTZero or Turnitin result, and those detectors use their own models and change them often, so a low score here does not guarantee a pass there. Check important text with the detector that matters to you.
- Read the rewrite before using it. The rewriter is told to preserve facts, but you are responsible for what you submit.
- Text under about 40 words is too short to score; under 120 words the score is rough.
- Input limit is 60,000 characters per request.
