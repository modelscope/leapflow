"""Tests for the shared check() convention in leapspace.app_space.utils."""

from leapspace.app_space.utils import check


def test_check_pass_line(capsys):
    assert check("reply-sent", True, "found") is True
    assert capsys.readouterr().out == "PASS reply-sent: found\n"


def test_check_fail_line_without_detail(capsys):
    assert check("badge-cleared", False) is False
    assert capsys.readouterr().out == "FAIL badge-cleared: \n"


def test_check_accumulates_with_and():
    ok = True
    ok &= check("a", True)
    ok &= check("b", False)
    ok &= check("c", True)
    assert ok is False
