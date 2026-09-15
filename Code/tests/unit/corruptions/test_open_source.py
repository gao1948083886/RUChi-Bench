"""Tests for safety filters around third-party augmentation output."""

from ruchi_bench.corruptions.open_source import (
    _has_disallowed_added_chars,
    _sanitize_added_chars,
)


def test_noise_filter_rejects_emoji_and_format_characters() -> None:
    assert _has_disallowed_added_chars("中文文本", "中文👫文本")
    assert _has_disallowed_added_chars("中文文本", "中文\ufeff文本")
    assert _has_disallowed_added_chars("中文文本", "中文\u2029文本")


def test_noise_filter_accepts_normal_text_and_punctuation() -> None:
    assert not _has_disallowed_added_chars("中文文本", "中文A1，文本")


def test_noise_sanitizer_keeps_normal_insertions_and_drops_emoji() -> None:
    assert _sanitize_added_chars("中文文本", "中文A👫1，文本") == "中文A1，文本"
