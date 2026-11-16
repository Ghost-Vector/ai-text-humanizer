"""Humanizer web app. Run:  python server.py   then open http://127.0.0.1:8000"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).parent


def _load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_env()  # before importing humanizer, which reads HUMANIZER_MODEL at import time

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from humanizer import HumanizerError, analyze, humanize  # noqa: E402

MAX_CHARS = 60_000
app = FastAPI(title="Humanizer")


class AnalyzeRequest(BaseModel):
    text: str = Field(max_length=MAX_CHARS)
    tone: str = "match"


class HumanizeRequest(AnalyzeRequest):
    intensity: str = "balanced"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.post("/api/analyze")
def api_analyze(req: AnalyzeRequest) -> dict:
    return analyze(req.text, formal=req.tone == "academic")


@app.post("/api/humanize")
async def api_humanize(req: HumanizeRequest) -> StreamingResponse:
    if not req.text.strip():
        raise HTTPException(400, "Text is empty.")

    async def events():
        try:
            async for event in humanize(req.text, req.tone, req.intensity):
                yield json.dumps(event) + "\n"
        except HumanizerError as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
