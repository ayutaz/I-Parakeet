import pytest

from iparakeet.eval.rtf import RTFMeter


def test_rtf_is_total_processing_time_over_total_audio():
    meter = RTFMeter()
    meter.add(audio_seconds=10.0, processing_seconds=0.5)
    meter.add(audio_seconds=30.0, processing_seconds=1.5)
    assert meter.rtf == pytest.approx(2.0 / 40.0)


def test_padding_ratio_counts_extra_processed_audio():
    meter = RTFMeter()
    meter.add(audio_seconds=4.0, processing_seconds=0.1, padded_seconds=5.0)
    meter.add(audio_seconds=6.0, processing_seconds=0.1, padded_seconds=7.0)
    assert meter.padding_ratio == pytest.approx(2.0 / 10.0)


def test_empty_meter_reports_zero():
    meter = RTFMeter()
    assert meter.rtf == 0.0
    assert meter.padding_ratio == 0.0
