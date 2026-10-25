"""Select fresh A-share hot stocks for scheduled analysis without editing .env."""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def uses_hot_stock_pool(config: Any, args: Any, stock_codes: Optional[List[str]] = None) -> bool:
    """Explicit stocks, portfolio runs and market-only runs take precedence."""
    return (
        stock_codes is None
        and not getattr(args, "stocks", None)
        and not getattr(args, "portfolio", None)
        and not getattr(args, "market_review", False)
        and getattr(config, "schedule_stock_source", "watchlist") == "hot"
    )


def select_hot_stock_codes(count: int, *, manager: Any = None) -> List[str]:
    """Fetch a new ranking, normalize exchange codes and require a complete pool.

    The existing data-provider fallback chain owns source selection. Failure or
    an incomplete ranking raises before any stock analysis/notification starts.
    The saved watchlist is never used as a substitute or overwritten.
    """
    from data_provider import DataFetcherManager
    from data_provider.base import normalize_stock_code
    from src.services.stock_list_parser import ParseStatus, parse_analysis_target

    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 100:
        raise ValueError("SCHEDULE_HOT_STOCK_COUNT 必须是 1 到 100 之间的整数")
    manager = manager if manager is not None else DataFetcherManager()
    # Leave room for duplicate/malformed entries without changing ranking order.
    rows = manager.get_hot_stocks(n=min(100, count * 2))
    codes: List[str] = []
    sources: List[str] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        raw_code = str(row.get("code") or "").strip()
        target = parse_analysis_target(raw_code)
        if target.asset_type == ParseStatus.INDEX:
            continue
        code = normalize_stock_code(raw_code)
        if len(code) != 6 or not code.isdigit() or code in codes:
            continue
        codes.append(code)
        source = str(row.get("source") or "").strip()
        if source and source not in sources:
            sources.append(source)
        if len(codes) == count:
            break
    if len(codes) != count:
        raise RuntimeError(
            f"热门股榜单不足或不可用：有效股票 {len(codes)}/{count}，本次停止分析。"
        )
    logger.info("定时热门股股票池已刷新：来源=%s，股票=%s", ",".join(sources), ",".join(codes))
    return codes
