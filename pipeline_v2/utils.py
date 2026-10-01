"""
utils.py — Logging, config loading, and retry helpers.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from rich.logging import RichHandler


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_file: str = "pipeline.log", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("formalrx")
    logger.setLevel(level)

    # Console handler (rich)
    if not any(isinstance(h, RichHandler) for h in logger.handlers):
        console = RichHandler(rich_tracebacks=True, markup=True)
        console.setLevel(level)
        logger.addHandler(console)

    # File handler
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    return logger


logger = setup_logging()


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load config.yaml and apply any environment variable overrides.

    Env var format: FORMALRX__<SECTION>__<KEY>  (double underscore separator)
    Examples:
      FORMALRX__ACTIVE_PROVIDER=openai
      FORMALRX__PROVIDERS__CLAUDE__API_KEY=sk-xxx
      FORMALRX__GENERATION__TEMPERATURE=0.0
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    # Apply environment overrides (basic flattening)
    for key, value in os.environ.items():
        if not key.startswith("FORMALRX__"):
            continue
        parts = key[len("FORMALRX__"):].lower().split("__")
        node = config
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        # Try to cast numeric / bool
        final_key = parts[-1]
        if value.lower() in ("true", "false"):
            node[final_key] = value.lower() == "true"
        else:
            try:
                node[final_key] = int(value)
            except ValueError:
                try:
                    node[final_key] = float(value)
                except ValueError:
                    node[final_key] = value

    return config


def get_provider_config(config: dict, provider_override: str | None = None) -> dict:
    """
    Return a provider's configuration merged with generation defaults.
    Also resolves api_key from environment if set as:
      OPENAI_API_KEY / ANTHROPIC_API_KEY / AZURE_OPENAI_API_KEY

    Args:
        config: full config dict
        provider_override: if set, use this provider name instead of active_provider
    """
    provider_name = provider_override or config.get("active_provider", "claude")
    providers = config.get("providers", {})

    if provider_name not in providers:
        raise ValueError(f"Provider '{provider_name}' not found in config.yaml providers section.")

    prov = dict(providers[provider_name])  # copy

    # Override api_key from env if set
    env_key_map = {

        "gemini": "GEMINI_API_KEY", 
        "openai": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
    }
    env_var = env_key_map.get(provider_name)
    if env_var and os.environ.get(env_var):
        prov["api_key"] = os.environ[env_var]

    prov["provider_name"] = provider_name
    return prov


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

import json
from pathlib import Path


def load_checkpoint(checkpoint_file: str) -> set[str]:
    """Return set of idx strings already processed."""
    path = Path(checkpoint_file)
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    done.add(obj.get("idx", ""))
                except json.JSONDecodeError:
                    pass
    logger.info(f"Checkpoint loaded: {len(done)} samples already processed")
    return done


def append_checkpoint(checkpoint_file: str, prediction_dict: dict) -> None:
    """Append a single prediction to the checkpoint file."""
    with open(checkpoint_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(prediction_dict, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Async rate limiter
# ─────────────────────────────────────────────────────────────────────────────

class AsyncRateLimiter:
    """
    Simple token-bucket rate limiter.
    Useful for endpoints that impose requests-per-minute (RPM) limits.
    """

    def __init__(self, max_requests_per_minute: int = 60):
        self._interval = 60.0 / max_requests_per_minute
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = asyncio.get_event_loop().time()
