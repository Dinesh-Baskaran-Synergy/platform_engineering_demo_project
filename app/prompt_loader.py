"""Load versioned system prompts from ``app/prompts/<name>.v<N>.md``.

The version string (``name@vN#hash``) is recorded with every eval result so a
score change can be traced back to a prompt change.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"


@dataclass(frozen=True)
class Prompt:
    """A loaded prompt template and its traceable version id."""

    name: str
    text: str
    version: str


@lru_cache(maxsize=None)
def load_prompt(name: str, version: int | None = None) -> Prompt:
    """Load prompt ``name``; latest version unless ``version`` or ``PROMPT_VERSION_<NAME>`` is set."""
    env_v = os.environ.get(f"PROMPT_VERSION_{name.upper()}")
    if version is None and env_v:
        version = int(env_v)
    candidates = {}
    for p in PROMPT_DIR.glob(f"{name}.v*.md"):
        m = re.fullmatch(rf"{re.escape(name)}\.v(\d+)\.md", p.name)
        if m:
            candidates[int(m.group(1))] = p
    if not candidates:
        raise FileNotFoundError(f"No prompt files for '{name}' in {PROMPT_DIR}")
    v = version if version is not None else max(candidates)
    if v not in candidates:
        raise FileNotFoundError(f"Prompt '{name}' has no version {v}")
    text = candidates[v].read_text(encoding="utf-8").strip()
    digest = hashlib.sha256(text.encode()).hexdigest()[:8]
    return Prompt(name=name, text=text, version=f"{name}@v{v}#{digest}")
