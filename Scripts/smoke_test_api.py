"""Phase 06 smoke test: verify RealAPIClient connects to the model server.

Run with:
    ~/.conda/envs/ruchi_bench/python.exe Scripts/smoke_test_api.py

Expects .env in the project root with API_BASE, API_MODEL, API_KEY.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code" / "src"))

from ruchi_bench.inference.client import (  # noqa: E402
    APIClientError,
    APIParseError,
    APITimeoutError,
    RealAPIClient,
)

# Load .env from project root
_env_path = ROOT / ".env"
load_dotenv(_env_path)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _redact(s: str, keep: int = 4) -> str:
    if len(s) <= keep * 2:
        return "***"
    return s[:keep] + "***" + s[-keep:]


def smoke_test() -> bool:
    api_base = os.environ.get("API_BASE", "").rstrip("/")
    api_key = os.environ.get("API_KEY", "")
    model_name = os.environ.get("API_MODEL", "unknown")

    logger.info("=" * 60)
    logger.info("Phase 06 — Real API Smoke Test")
    logger.info("=" * 60)
    logger.info("API_BASE:   %s", api_base)
    logger.info("API_MODEL:  %s", model_name)
    logger.info("API_KEY:    %s", _redact(api_key))
    logger.info("-" * 60)

    if not api_base or not api_key:
        logger.error("API_BASE or API_KEY not set in .env")
        return False

    client = RealAPIClient(
        api_base=f"{api_base}/v1" if not api_base.endswith("/v1") else api_base,
        api_key=api_key,
        model_name=model_name,
        timeout=30.0,
        max_retries=3,
    )

    # Pair / sentence-pair task (XNLI / PAWS-X style)
    pair_content = (
        "判断以下两句话的关系：\n"
        "第一句：今天天气很好\n"
        "第二句：今天阳光明媚\n"
        "选项：A. 蕴含 B. 矛盾 C. 中立\n"
        "答案："
    )

    # Single-text sentiment (ASAP style)
    sentiment_content = (
        "判断情感极性，只输出 positive 或 negative："
        "这家餐厅服务很好，菜品也不错。"
    )

    # MRC (C3 style)
    mrc_content = (
        "根据以下文章回答问题。\n"
        "文章：小明去学校上课，老师教语文和数学。\n"
        "问题：小明在学校学什么？\n"
        "选项：A. 语文和数学 B. 英语 C. 科学"
    )

    tests: list[tuple[str, list[dict[str, str]], int]] = [
        ("sentence-pair", [{"role": "user", "content": pair_content}], 8),
        ("sentiment", [{"role": "user", "content": sentiment_content}], 8),
        ("mrc", [{"role": "user", "content": mrc_content}], 16),
    ]

    all_passed = True
    for name, messages, max_tokens in tests:
        logger.info("[%s] sending request ...", name)
        t0 = time.monotonic()
        try:
            resp = client.call(
                messages=messages,
                sample_id=f"smoke-{name}",
                temperature=0.0,
                max_tokens=max_tokens,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            logger.info(
                "[%s] PASS  latency=%.1fms  model=%s  tokens=%d+%d  content=%r",
                name,
                elapsed_ms,
                resp.model,
                resp.prompt_tokens,
                resp.completion_tokens,
                resp.content[:80] if resp.content else "(empty)",
            )
        except APITimeoutError as exc:
            logger.error("[%s] TIMEOUT: %s", name, exc)
            all_passed = False
        except APIParseError as exc:
            logger.error("[%s] PARSE ERROR: %s", name, exc)
            all_passed = False
        except APIClientError as exc:
            logger.error("[%s] ERROR: %s", name, exc)
            all_passed = False

    client.close()
    logger.info("=" * 60)
    if all_passed:
        logger.info("SMOKE TEST: ALL PASSED")
    else:
        logger.error("SMOKE TEST: FAILED")
    logger.info("=" * 60)
    return all_passed


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Run a connectivity smoke test against the configured API."
    ).parse_args()
    ok = smoke_test()
    sys.exit(0 if ok else 1)
