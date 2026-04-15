from __future__ import annotations

from typing import Any

__all__ = ["run_fetch_cycle"]


async def run_fetch_cycle(*args: Any, **kwargs: Any) -> None:
	from src.data.fetcher import run_fetch_cycle as _run_fetch_cycle

	await _run_fetch_cycle(*args, **kwargs)
