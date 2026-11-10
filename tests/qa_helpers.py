"""Controlled offline passages and clients; no evaluation answer key is read."""

from __future__ import annotations

import json
from types import SimpleNamespace

from config import load_settings
from src.models import RetrievedChunk


def settings_for(tmp_path):
    return load_settings(project_root=tmp_path, environ={})


def passage(number=1, *, text=None, document_type="רשומה טכנית", period="נתון טכני", section="סעיף בדיקת תוכנה"):
    return RetrievedChunk(
        chunk_id=f"ALT-{number:02d}-controlled-passage",
        document_id=f"ALT-{number:02d}",
        source_file=f"ALT-{number:02d}_בדיקה.md",
        title=f"כותרת מקור טכני {number:02d}",
        section=section,
        text=text or "המסמך הטכני מציין כי מספר הרשומות בבדיקת התוכנה הוא שבע.",
        similarity=0.8,
        document_type=document_type,
        period=period,
        topics="בדיקת תוכנה בלבד",
    )


def response_for(chunk, *, text="מספר הרשומות בבדיקת התוכנה הוא שבע.", status="answered", quote=None):
    return {
        "status": status,
        "claims": [{"text": text, "evidence": [{"chunk_id": chunk.chunk_id, "quote": quote or chunk.text}]}],
    }


class StaticRetriever:
    def __init__(self, chunks=(), *, error=None):
        self.chunks = tuple(chunks)
        self.queries = []
        self.error = error

    def retrieve(self, question):
        self.queries.append(question)
        if self.error is not None:
            raise self.error
        return self.chunks


class ControlledLLM:
    def __init__(self, response=None, *, error=None):
        self.response = {"status": "no_answer", "claims": []} if response is None else response
        self.calls = []
        self.error = error

    def chat(self, messages):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        content = json.dumps(self.response, ensure_ascii=False) if isinstance(self.response, dict) else self.response
        return SimpleNamespace(message=SimpleNamespace(content=content))
