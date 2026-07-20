import pytest

from mmml.md.temperature import parse_temperature_schedule


def test_constant_and_linear_temperature_schedules():
    constant = parse_temperature_schedule("298")
    assert constant.temperature_at(500, 1000) == pytest.approx(298)

    ramp = parse_temperature_schedule("200->400")
    assert ramp.temperature_at(0, 1000) == pytest.approx(200)
    assert ramp.temperature_at(500, 1000) == pytest.approx(300)
    assert ramp.temperature_at(1000, 1000) == pytest.approx(400)


def test_staged_temperature_schedule_boundary_and_round_trip_values():
    schedule = parse_temperature_schedule("200->300:0.25,300:0.5,300->200:0.25")
    assert [schedule.temperature_at(step, 1000) for step in (0, 250, 750, 1000)] == [
        200, 300, 300, 200
    ]


def test_temperature_schedule_validation():
    with pytest.raises(ValueError, match="sum to 1"):
        parse_temperature_schedule("200:0.2,300:0.2")
    with pytest.raises(ValueError, match="either all"):
        parse_temperature_schedule("200:0.5,300")
    with pytest.raises(ValueError, match="positive"):
        parse_temperature_schedule("0")
