"""Dynamic scheduler pools, real occurrence claims and stock normalization."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()
import main  # noqa: E402
from src.services.scheduled_analysis_claim import ScheduledAnalysisSkipped, claim_scheduled_analysis  # noqa: E402
from src.services.scheduled_stock_selection import select_hot_stock_codes, uses_hot_stock_pool  # noqa: E402


class Ranking:
    def __init__(self, *codes):
        self.rows = [{"code": code, "source": "fixture"} for code in codes]

    def get_hot_stocks(self, n):
        return self.rows[:n]


def config(database):
    return SimpleNamespace(
        database_path=str(database), stock_list=["600519"],
        schedule_stock_source="hot", schedule_hot_stock_count=10,
        trading_day_check_enabled=True,
    )


def test_hot_pool_normalizes_deduplicates_and_preserves_ranking_order():
    manager = Ranking("SH600519", "600519.SH", "sz000001", "BJ920748", "HK00700", "invalid")
    assert select_hot_stock_codes(3, manager=manager) == ["600519", "000001", "920748"]
    assert select_hot_stock_codes(1, manager=Ranking("SH000001", "SZ000001")) == ["000001"]


def test_incomplete_hot_pool_fails_instead_of_using_a_previous_watchlist():
    with pytest.raises(RuntimeError, match="1/3"):
        select_hot_stock_codes(3, manager=Ranking("SH600519", "SH600519", "HK00700"))
    with pytest.raises(RuntimeError, match="0/10"):
        select_hot_stock_codes(10, manager=Ranking())


@pytest.mark.parametrize("count", [0, 101, True, "10"])
def test_invalid_hot_count_does_not_query_a_provider(count):
    with pytest.raises(ValueError):
        select_hot_stock_codes(count, manager=object())


def test_explicit_stocks_portfolio_and_market_only_take_precedence():
    cfg = SimpleNamespace(schedule_stock_source="hot")
    assert uses_hot_stock_pool(cfg, SimpleNamespace())
    assert not uses_hot_stock_pool(cfg, SimpleNamespace(), [])
    for options in [{"stocks": "600519"}, {"portfolio": "futu"}, {"market_review": True}]:
        assert not uses_hot_stock_pool(cfg, SimpleNamespace(**options))


def test_two_daily_slots_refresh_the_pool_and_claim_before_fetching():
    with tempfile.TemporaryDirectory() as directory:
        cfg = config(Path(directory) / "analysis.db")
        due = datetime(2026, 10, 14, 12)
        noon = [f"002{i:03d}" for i in range(10)]
        evening = [f"600{i:03d}" for i in range(10)]
        reported = []
        args = SimpleNamespace(_scheduled_for=due, _stock_selection_callback=reported.append)
        with patch("src.core.trading_calendar.get_open_markets_today", return_value=["cn"]), \
             patch("src.services.scheduled_stock_selection.select_hot_stock_codes", side_effect=[noon, evening]) as fetch, \
             patch("main.run_full_analysis", return_value=True) as analyze:
            assert main.run_scheduled_analysis(cfg, args)
            with pytest.raises(ScheduledAnalysisSkipped):
                main.run_scheduled_analysis(cfg, args)
            args._scheduled_for = due + timedelta(hours=6)
            assert main.run_scheduled_analysis(cfg, args)
        assert fetch.call_count == 2
        assert [call.args[2] for call in analyze.call_args_list] == [noon, evening]
        assert all(call.kwargs["refresh_watchlist"] is False for call in analyze.call_args_list)
        assert reported == [noon, evening]
        assert cfg.stock_list == ["600519"]


def test_closed_a_share_market_skips_fetch_analysis_and_notification_path():
    cfg = config("unused.db")
    with patch("src.core.trading_calendar.get_open_markets_today", return_value=["us"]), \
         patch("src.services.scheduled_stock_selection.select_hot_stock_codes") as fetch, \
         patch("main.run_full_analysis") as analyze:
        assert main.run_scheduled_analysis(cfg, SimpleNamespace())
    fetch.assert_not_called()
    analyze.assert_not_called()


def test_provider_failure_propagates_without_analyzing_or_repeating_a_slot():
    with tempfile.TemporaryDirectory() as directory:
        cfg = config(Path(directory) / "analysis.db")
        args = SimpleNamespace(_scheduled_for=datetime(2026, 10, 14, 12), force_run=True)
        with patch("src.services.scheduled_stock_selection.select_hot_stock_codes", side_effect=RuntimeError("ranking failed")) as fetch, \
             patch("main.run_full_analysis") as analyze:
            with pytest.raises(RuntimeError, match="ranking failed"):
                main.run_scheduled_analysis(cfg, args)
            with pytest.raises(ScheduledAnalysisSkipped):
                main.run_scheduled_analysis(cfg, args)
        fetch.assert_called_once()
        analyze.assert_not_called()


def test_hot_claims_use_policy_not_watchlist_and_explicit_stocks_remain_distinct():
    with tempfile.TemporaryDirectory() as directory:
        cfg = config(Path(directory) / "analysis.db")
        due, args = datetime(2026, 10, 14, 12), SimpleNamespace()
        assert claim_scheduled_analysis(cfg, args, due)
        cfg.stock_list = ["000001"]
        assert not claim_scheduled_analysis(cfg, args, due)
        cfg.schedule_hot_stock_count = 5
        assert claim_scheduled_analysis(cfg, args, due)
        assert claim_scheduled_analysis(cfg, args, due, stock_codes=["600519"])
        assert claim_scheduled_analysis(cfg, args, due, stock_codes=["000001"])


def test_explicit_scheduled_stocks_bypass_the_hot_pool():
    cfg = config("unused.db")
    with patch("src.services.scheduled_stock_selection.select_hot_stock_codes") as fetch, \
         patch("main.run_full_analysis", return_value=True) as analyze:
        assert main.run_scheduled_analysis(cfg, SimpleNamespace(), ["000001"])
    fetch.assert_not_called()
    assert analyze.call_args.args[2] == ["000001"]
