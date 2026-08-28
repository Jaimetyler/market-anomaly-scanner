import pytest

from scanner.corporate_actions import (
    CorporateActionEvent,
    check_corporate_action_risk,
    classify_split,
    find_nearby_actions,
    parse_split_event,
    split_adjustment_factor,
)


def test_classify_forward_split():
    assert (
        classify_split(
            split_from=1,
            split_to=4,
        )
        == "FORWARD_SPLIT"
    )


def test_classify_reverse_split():
    assert (
        classify_split(
            split_from=20,
            split_to=1,
        )
        == "REVERSE_SPLIT"
    )


def test_classify_neutral_split():
    assert (
        classify_split(
            split_from=1,
            split_to=1,
        )
        == "NEUTRAL_SPLIT"
    )


def test_classify_split_rejects_invalid_values():
    with pytest.raises(ValueError):
        classify_split(
            split_from=0,
            split_to=1,
        )

    with pytest.raises(ValueError):
        classify_split(
            split_from=1,
            split_to=0,
        )

    with pytest.raises(ValueError):
        classify_split(
            split_from=-1,
            split_to=1,
        )


def test_split_adjustment_factor():
    assert (
        split_adjustment_factor(
            split_from=1,
            split_to=4,
        )
        == pytest.approx(4.0)
    )

    assert (
        split_adjustment_factor(
            split_from=20,
            split_to=1,
        )
        == pytest.approx(0.05)
    )


def test_parse_reverse_split_event():
    event = parse_split_event(
        {
            "ticker": "TEST",
            "execution_date": "2026-05-01",
            "split_from": 20,
            "split_to": 1,
        }
    )

    assert event.ticker == "TEST"
    assert event.event_type == "REVERSE_SPLIT"

    assert (
        event.execution_date.isoformat()
        == "2026-05-01"
    )

    assert event.split_from == pytest.approx(20.0)
    assert event.split_to == pytest.approx(1.0)


def test_find_nearby_actions():
    actions = [
        CorporateActionEvent(
            event_type="REVERSE_SPLIT",
            execution_date=parse_split_event(
                {
                    "execution_date": "2026-05-01",
                    "split_from": 20,
                    "split_to": 1,
                }
            ).execution_date,
            split_from=20,
            split_to=1,
            ticker="TEST",
        ),
        CorporateActionEvent(
            event_type="FORWARD_SPLIT",
            execution_date=parse_split_event(
                {
                    "execution_date": "2026-06-01",
                    "split_from": 1,
                    "split_to": 2,
                }
            ).execution_date,
            split_from=1,
            split_to=2,
            ticker="TEST",
        ),
    ]

    nearby = find_nearby_actions(
        signal_date="2026-05-03",
        actions=actions,
        lookback_days=10,
        lookforward_days=3,
    )

    assert len(nearby) == 1

    assert (
        nearby[0].event_type
        == "REVERSE_SPLIT"
    )


def test_reverse_split_is_excluded():
    event = parse_split_event(
        {
            "ticker": "TEST",
            "execution_date": "2026-05-01",
            "split_from": 20,
            "split_to": 1,
        }
    )

    result = check_corporate_action_risk(
        signal_date="2026-05-02",
        actions=[event],
    )

    assert result.flagged is True
    assert result.exclude_from_research is True

    assert (
        "RECENT_REVERSE_SPLIT"
        in result.flags
    )


def test_forward_split_flagged_but_not_excluded():
    event = parse_split_event(
        {
            "ticker": "TEST",
            "execution_date": "2026-05-01",
            "split_from": 1,
            "split_to": 4,
        }
    )

    result = check_corporate_action_risk(
        signal_date="2026-05-02",
        actions=[event],
    )

    assert result.flagged is True

    assert (
        result.exclude_from_research
        is False
    )

    assert (
        "RECENT_FORWARD_SPLIT"
        in result.flags
    )


def test_extreme_move_is_flagged_without_known_action():
    result = check_corporate_action_risk(
        signal_date="2026-05-02",
        actions=[],
        return_1d=125,
    )

    assert result.flagged is True

    assert (
        "EXTREME_MOVE_NEEDS_CA_CHECK"
        in result.flags
    )

    assert (
        result.exclude_from_research
        is False
    )


def test_extreme_five_day_move_is_flagged():
    result = check_corporate_action_risk(
        signal_date="2026-05-02",
        actions=[],
        return_5d=350,
    )

    assert result.flagged is True

    assert (
        "EXTREME_MOVE_NEEDS_CA_CHECK"
        in result.flags
    )


def test_extreme_twenty_day_move_is_flagged():
    result = check_corporate_action_risk(
        signal_date="2026-05-02",
        actions=[],
        return_20d=600,
    )

    assert result.flagged is True

    assert (
        "EXTREME_MOVE_NEEDS_CA_CHECK"
        in result.flags
    )


def test_normal_move_with_no_actions_is_clean():
    result = check_corporate_action_risk(
        signal_date="2026-05-02",
        actions=[],
        return_1d=15,
        return_5d=50,
        return_20d=100,
    )

    assert result.flagged is False

    assert (
        result.exclude_from_research
        is False
    )

    assert result.flags == []
    assert result.reasons == []
    assert result.nearby_actions == []


def test_parse_massive_forward_split():
    event = parse_split_event(
        {
            "ticker": "TEST",
            "execution_date": "2026-05-01",
            "adjustment_type": "forward_split",
            "split_from": 1,
            "split_to": 5,
            "historical_adjustment_factor": 0.2,
        }
    )

    assert event.event_type == "FORWARD_SPLIT"

    assert (
        event.historical_adjustment_factor
        == pytest.approx(0.2)
    )


def test_parse_massive_reverse_split():
    event = parse_split_event(
        {
            "ticker": "TEST",
            "execution_date": "2026-05-01",
            "adjustment_type": "reverse_split",
            "split_from": 20,
            "split_to": 1,
            "historical_adjustment_factor": 20.0,
        }
    )

    assert event.event_type == "REVERSE_SPLIT"

    assert (
        event.historical_adjustment_factor
        == pytest.approx(20.0)
    )


def test_parse_stock_dividend():
    event = parse_split_event(
        {
            "ticker": "TEST",
            "execution_date": "2026-05-01",
            "adjustment_type": "stock_dividend",
            "split_from": 1,
            "split_to": 1.1,
            "historical_adjustment_factor": 0.909091,
        }
    )

    assert (
        event.event_type
        == "STOCK_DIVIDEND"
    )

    assert (
        event.historical_adjustment_factor
        == pytest.approx(0.909091)
    )


def test_stock_dividend_is_flagged_not_excluded():
    event = parse_split_event(
        {
            "ticker": "TEST",
            "execution_date": "2026-05-01",
            "adjustment_type": "stock_dividend",
            "split_from": 1,
            "split_to": 1.1,
        }
    )

    result = check_corporate_action_risk(
        signal_date="2026-05-02",
        actions=[event],
    )

    assert result.flagged is True

    assert (
        result.exclude_from_research
        is False
    )

    assert (
        "RECENT_STOCK_DIVIDEND"
        in result.flags
    )