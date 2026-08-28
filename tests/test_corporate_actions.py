import pytest

from scanner.corporate_actions import (
    CorporateActionEvent,
    check_corporate_action_risk,
    classify_split,
    find_nearby_actions,
    parse_split_event,
    split_adjustment_factor,
)


def test_forward_split_classification():
    assert (
        classify_split(
            split_from=1,
            split_to=5,
        )
        == "FORWARD_SPLIT"
    )


def test_reverse_split_classification():
    assert (
        classify_split(
            split_from=10,
            split_to=1,
        )
        == "REVERSE_SPLIT"
    )


def test_neutral_split_classification():
    assert (
        classify_split(
            split_from=1,
            split_to=1,
        )
        == "NEUTRAL_SPLIT"
    )


def test_invalid_split_values():
    with pytest.raises(ValueError):
        classify_split(
            split_from=0,
            split_to=5,
        )


def test_split_adjustment_factor():
    assert (
        split_adjustment_factor(
            split_from=1,
            split_to=5,
        )
        == pytest.approx(5.0)
    )

    assert (
        split_adjustment_factor(
            split_from=10,
            split_to=1,
        )
        == pytest.approx(0.1)
    )


def test_parse_reverse_split():
    raw = {
        "ticker": "TEST",
        "execution_date": "2026-01-15",
        "split_from": 20,
        "split_to": 1,
    }

    event = parse_split_event(raw)

    assert (
        event.event_type
        == "REVERSE_SPLIT"
    )

    assert (
        event.execution_date.isoformat()
        == "2026-01-15"
    )

    assert event.ticker == "TEST"


def test_find_nearby_actions():
    actions = [
        CorporateActionEvent(
            event_type="REVERSE_SPLIT",
            execution_date=(
                __import__(
                    "datetime"
                ).date(
                    2026,
                    1,
                    10,
                )
            ),
        ),
        CorporateActionEvent(
            event_type="FORWARD_SPLIT",
            execution_date=(
                __import__(
                    "datetime"
                ).date(
                    2025,
                    10,
                    1,
                )
            ),
        ),
    ]

    nearby = find_nearby_actions(
        signal_date="2026-01-15",
        actions=actions,
        lookback_days=10,
        lookforward_days=3,
    )

    assert len(nearby) == 1

    assert (
        nearby[0].event_type
        == "REVERSE_SPLIT"
    )


def test_reverse_split_excluded_from_clean_research():
    action = parse_split_event(
        {
            "ticker": "TEST",
            "execution_date": "2026-01-10",
            "split_from": 20,
            "split_to": 1,
        }
    )

    result = (
        check_corporate_action_risk(
            signal_date="2026-01-15",
            actions=[action],
            return_5d=500,
        )
    )

    assert result.flagged is True

    assert (
        result.exclude_from_research
        is True
    )

    assert (
        "RECENT_REVERSE_SPLIT"
        in result.flags
    )

    assert (
        "EXTREME_MOVE_NEEDS_CA_CHECK"
        in result.flags
    )


def test_forward_split_flagged_but_not_excluded():
    action = parse_split_event(
        {
            "ticker": "TEST",
            "execution_date": "2026-01-14",
            "split_from": 1,
            "split_to": 4,
        }
    )

    result = (
        check_corporate_action_risk(
            signal_date="2026-01-15",
            actions=[action],
            return_5d=40,
        )
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


def test_extreme_move_flagged_without_known_action():
    result = (
        check_corporate_action_risk(
            signal_date="2026-01-15",
            actions=[],
            return_1d=150,
        )
    )

    assert result.flagged is True

    assert (
        result.exclude_from_research
        is False
    )

    assert (
        "EXTREME_MOVE_NEEDS_CA_CHECK"
        in result.flags
    )


def test_normal_move_without_action_is_clean():
    result = (
        check_corporate_action_risk(
            signal_date="2026-01-15",
            actions=[],
            return_1d=15,
            return_5d=40,
            return_20d=80,
        )
    )

    assert result.flagged is False

    assert (
        result.exclude_from_research
        is False
    )

    assert result.flags == []