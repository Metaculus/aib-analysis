"""Structured LLM calls through OpenRouter, cached on disk so reruns cost nothing and every call can be audited."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeVar

from forecasting_tools import GeneralLlm, MonetaryCostManager
from pydantic import BaseModel

from aib_analysis.bot_maker_advice import method

logger = logging.getLogger(__name__)

ReplyType = TypeVar("ReplyType", bound=BaseModel)


class StructuredLlm(Protocol):
    async def ask(self, stage: str, model: str, prompt: str, reply_type: type[ReplyType]) -> ReplyType: ...

    def usage_note(self) -> str: ...


class CachedOpenRouterLlm:
    def __init__(self, cache_folder: Path, cost_manager: MonetaryCostManager | None = None) -> None:
        self.cache_folder = cache_folder
        self.cost_manager = cost_manager
        self.new_calls = 0
        self.cached_calls = 0
        self._semaphore = asyncio.Semaphore(method.MAX_CONCURRENT_CALLS)

    async def ask(self, stage: str, model: str, prompt: str, reply_type: type[ReplyType]) -> ReplyType:
        digest = hashlib.sha256(f"{model}\n{reply_type.__name__}\n{prompt}".encode()).hexdigest()[:24]
        cache_path = self.cache_folder / stage / f"{digest}.json"
        if cache_path.exists():
            self.cached_calls += 1
            return reply_type.model_validate(json.loads(cache_path.read_text())["reply"])
        if "OPENROUTER_API_KEY" not in os.environ:
            raise RuntimeError("OPENROUTER_API_KEY is not set; run this through envh (see README)")
        async with self._semaphore:
            llm = GeneralLlm(model=model, temperature=0, timeout=method.LLM_TIMEOUT_SECONDS)
            reply = await llm.invoke_and_return_verified_type(prompt, reply_type)
        self.new_calls += 1
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"stage": stage, "model": model, "prompt": prompt, "reply": reply.model_dump()}, indent=2))
        return reply

    def usage_note(self) -> str:
        cost = f", ${self.cost_manager.current_usage:.2f} spent so far" if self.cost_manager is not None else ""
        return f"{self.new_calls} new calls, {self.cached_calls} from cache{cost}"


async def ask_until_valid(
    llm: StructuredLlm,
    stage: str,
    model: str,
    prompt: str,
    reply_type: type[ReplyType],
    find_problems: Callable[[ReplyType], list[str]],
) -> tuple[ReplyType, list[str]]:
    """Ask again with the problems listed, up to MAX_FORMAT_RETRIES times; return the last reply and its remaining problems."""
    current_prompt = prompt
    for attempt in range(method.MAX_FORMAT_RETRIES + 1):
        reply = await llm.ask(stage, model, current_prompt, reply_type)
        problems = find_problems(reply)
        if not problems:
            return reply, []
        logger.warning(f"{stage} ({method.model_label(model)}) attempt {attempt + 1}: {'; '.join(problems)}")
        current_prompt = (
            f"{prompt}\n\nYour previous reply had these problems. Reply again with all of them fixed:\n"
            + "\n".join(f"- {problem}" for problem in problems)
        )
    return reply, problems
