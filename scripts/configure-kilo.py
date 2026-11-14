"""Register the local RTX 4090 llama-server endpoint in ~/.config/kilo/kilo.json(c)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def strip_jsonc(text: str) -> str:
    """Strip // and /* */ comments and trailing commas without altering string literals."""
    token_re = re.compile(
        r'"(?:\\.|[^"\\])*"'  # double-quoted JSON string
        r"|//[^\r\n]*"  # line comment
        r"|/\*[\s\S]*?\*/",  # block comment
    )
    no_comments = token_re.sub(
        lambda m: m.group(0) if m.group(0).startswith('"') else "",
        text,
    )
    str_or_comma = re.compile(r'"(?:\\.|[^"\\])*"|,\s*(?=[}\]])')
    return str_or_comma.sub(
        lambda m: m.group(0) if m.group(0).startswith('"') else "",
        no_comments,
    )


def build_provider_block(
    *,
    port: int = 10088,
    alias: str = "qwen3.8-27b-coder",
    display_name: str = "Qwen 3.8 27B Coder (208K MTP)",
    api_key: str = "local-dev-key",
    context_limit: int = 212992,
    output_limit: int = 65536,
) -> dict[str, Any]:
    return {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Local GPU (llama.cpp)",
        "options": {
            "baseURL": f"http://127.0.0.1:{port}/v1",
            "apiKey": api_key,
            "timeout": 600000,
            "kiloOpenAICompatibleToolCallContent": "empty-string",
        },
        "models": {
            alias: {
                "name": display_name,
                "tool_call": True,
                "reasoning": True,
                "temperature": True,
                "limit": {
                    "context": context_limit,
                    "output": output_limit,
                },
                "variants": {
                    "low": {
                        "chat_template_kwargs": {
                            "reasoning_effort": "low",
                            "preserve_thinking": True,
                        }
                    },
                    "medium": {
                        "chat_template_kwargs": {
                            "reasoning_effort": "medium",
                            "preserve_thinking": True,
                        }
                    },
                    "xhigh": {
                        "chat_template_kwargs": {
                            "reasoning_effort": "xhigh",
                            "preserve_thinking": True,
                        }
                    },
                },
            }
        },
    }


def merge_config(
    raw_text: str,
    *,
    provider_id: str = "local-llama",
    provider_block: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cleaned = strip_jsonc(raw_text).strip()
    cfg: dict[str, Any] = (
        json.loads(cleaned)
        if cleaned
        else {"$schema": "https://app.kilo.ai/config.json"}
    )
    providers = cfg.get("provider")
    if not isinstance(providers, dict):
        providers = {}
        cfg["provider"] = providers
    providers[provider_id] = provider_block or build_provider_block()

    enabled = cfg.get("enabled_providers")
    if isinstance(enabled, list) and provider_id not in enabled:
        enabled.append(provider_id)

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider-id",
        default="local-llama",
        help="Provider ID in kilo.json (default: local-llama)",
    )
    parser.add_argument(
        "--alias",
        default="qwen3.8-27b-coder",
        help="Model --alias passed to llama-server",
    )
    parser.add_argument(
        "--name",
        default="Qwen 3.8 27B Coder (208K MTP)",
        help="Display name in Kilo model picker",
    )
    parser.add_argument(
        "--port", type=int, default=10088, help="llama-server port (default: 10088)"
    )
    parser.add_argument(
        "--api-key",
        default="local-dev-key",
        help="llama-server API key (default: local-dev-key)",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=212992,
        help="Per-slot context window (default: 212992)",
    )
    parser.add_argument(
        "--output",
        type=int,
        default=65536,
        help="Max output tokens including <think> (default: 65536)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path.home() / ".config" / "kilo",
        help="Kilo config directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print merged JSON without writing to disk",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    block = build_provider_block(
        port=args.port,
        alias=args.alias,
        display_name=args.name,
        api_key=args.api_key,
        context_limit=args.context,
        output_limit=args.output,
    )

    candidates = [args.config_dir / "kilo.json", args.config_dir / "kilo.jsonc"]
    cfg_path = next(
        (p for p in candidates if p.exists()), args.config_dir / "kilo.json"
    )
    raw = cfg_path.read_text(encoding="utf-8-sig") if cfg_path.exists() else ""
    merged = merge_config(raw, provider_id=args.provider_id, provider_block=block)
    rendered = json.dumps(merged, indent=2) + "\n"

    if args.dry_run:
        print(rendered, end="")
        return

    args.config_dir.mkdir(parents=True, exist_ok=True)
    if cfg_path.exists():
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        bak_path = cfg_path.with_name(f"{cfg_path.name}.bak-{stamp}")
        shutil.copy2(cfg_path, bak_path)
        print(f"Backup saved: {bak_path}")

    cfg_path.write_text(rendered, encoding="utf-8")
    print(f"Updated: {cfg_path}")

    kilo_bin = shutil.which("kilo")
    if kilo_bin:
        subprocess.run([kilo_bin, "models", args.provider_id], check=False)


if __name__ == "__main__":
    main()
