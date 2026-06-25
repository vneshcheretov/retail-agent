"""Personas (report tone).

Persona instructions live in ``config/personas.yaml`` and are read on every turn,
so a non-developer can change the report tone without a redeploy. Per-user
personalization lives separately in ``services/memory.py``.
"""

from __future__ import annotations

import re

import yaml

from src.settings import settings


def _load_yaml() -> dict:
    return yaml.safe_load(settings.personas_path.read_text(encoding="utf-8"))


# Persona fields rendered into the report prompt, in order.
_FIELDS = (("tone_of_voice", "Tone of voice"), ("manners", "Manners"), ("length", "Length"))


def load_persona() -> str:
    """Return the active persona as a prompt block, read fresh from disk."""
    data = _load_yaml()
    persona = data.get("personas", {}).get(data.get("active"))
    if not persona:
        return "Write a clear, concise analyst report."
    lines = [f"{title}: {persona[key].strip()}" for key, title in _FIELDS if persona.get(key)]
    if lines:
        return "\n".join(lines)
    return persona.get("instructions", "Write a clear, concise analyst report.").strip()


def list_personas() -> dict[str, str]:
    """Return {name: label} for all personas, marking the active one."""
    data = _load_yaml()
    active = data.get("active")
    return {
        name: f"{p.get('label', name)}{' (active)' if name == active else ''}"
        for name, p in data.get("personas", {}).items()
    }


def set_active_persona(name: str) -> bool:
    """Switch the active persona. Returns False if the name is unknown.

    Only the ``active:`` line is rewritten, so the file's comments and formatting
    (which guide the non-developer editing it) are kept intact.
    """
    if name not in _load_yaml().get("personas", {}):
        return False
    text = settings.personas_path.read_text(encoding="utf-8")
    new_text, count = re.subn(r"(?m)^active:.*$", f"active: {name}", text)
    if count == 0:
        new_text = f"active: {name}\n{text}"
    settings.personas_path.write_text(new_text, encoding="utf-8")
    return True


