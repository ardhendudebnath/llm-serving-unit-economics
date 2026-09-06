from __future__ import annotations

from bench.gpu import Monitor, Sample, decode_throttle_reasons, throttling_warning


def _sample(*, util: float, clock: float, clock_max: float = 3090.0,
            reasons: str = "0x0000000000000000", vram: float = 4000.0,
            temp: float = 60.0) -> Sample:
    return Sample(
        at=0.0,
        memory_used_mib=vram,
        memory_total_mib=12227.0,
        utilisation_pct=util,
        temperature_c=temp,
        sm_clock_mhz=clock,
        sm_clock_max_mhz=clock_max,
        power_w=80.0,
        throttle_reasons=reasons,
    )


# ------------------------------------------------------------- bitmask ------

def test_throttle_reasons_are_a_hex_bitmask_not_a_string():
    # nvidia-smi reports 0x0000000000000000, not "Not Active". Assuming the
    # latter made every healthy sample look like it carried an unknown reason.
    assert decode_throttle_reasons("0x0000000000000000") == ([], False)


def test_idle_and_clock_settings_are_not_performance_problems():
    names, limiting = decode_throttle_reasons("0x0000000000000001")
    assert names == ["GpuIdle"]
    assert not limiting


def test_thermal_and_power_bits_do_count():
    for raw, name in (
        ("0x0000000000000004", "SwPowerCap"),
        ("0x0000000000000020", "SwThermalSlowdown"),
        ("0x0000000000000040", "HwThermalSlowdown"),
        ("0x0000000000000080", "HwPowerBrakeSlowdown"),
    ):
        names, limiting = decode_throttle_reasons(raw)
        assert name in names and limiting


def test_multiple_bits_decode_together():
    names, limiting = decode_throttle_reasons("0x0000000000000044")
    assert names == ["SwPowerCap", "HwThermalSlowdown"]
    assert limiting


def test_unparseable_bitmask_is_not_read_as_throttling():
    assert decode_throttle_reasons("[N/A]") == ([], False)
    assert decode_throttle_reasons("") == ([], False)


# ------------------------------------------------------------ throttling ----

def test_an_idle_gpu_is_not_throttled():
    # The regression that real hardware exposed: an idle card sits at 180 MHz
    # against a 3090 MHz maximum -- a clock ratio of 0.06 -- purely to save
    # power. Reading that as throttling would flag every quiet moment as a
    # broken measurement.
    idle = _sample(util=0.0, clock=180.0)
    assert idle.clock_ratio < 0.1
    assert not idle.busy
    assert not idle.throttled


def test_a_busy_gpu_with_dropped_clocks_is_throttled():
    assert _sample(util=98.0, clock=1800.0).throttled


def test_a_busy_gpu_at_full_clocks_is_not_throttled():
    assert not _sample(util=98.0, clock=3000.0).throttled


def test_hardware_reasons_win_even_when_clocks_look_fine():
    # The card's own reason bits are authoritative: it can report a thermal
    # slowdown in the same instant the clock sample happens to read high.
    hot = _sample(util=95.0, clock=3050.0, reasons="0x0000000000000040")
    assert hot.throttled


# -------------------------------------------------------------- summary -----

def test_summary_judges_throttling_over_busy_samples_only():
    mon = Monitor()
    # Eight idle samples between load, two busy ones -- both throttled.
    mon.samples = [_sample(util=0.0, clock=180.0) for _ in range(8)]
    mon.samples += [_sample(util=99.0, clock=1500.0) for _ in range(2)]
    summary = mon.summary()

    assert summary["busy_samples"] == 2
    # Averaging the idle gaps in would have diluted a real thermal problem to
    # 20% and hidden it.
    assert summary["throttled_fraction"] == 1.0


def test_utilisation_is_averaged_over_every_sample_including_idle():
    # This is the number the cost model needs. Counting only busy samples
    # would inflate it, which is exactly the dishonesty it exists to prevent.
    mon = Monitor()
    mon.samples = [_sample(util=0.0, clock=180.0) for _ in range(9)]
    mon.samples += [_sample(util=100.0, clock=3000.0)]
    assert mon.summary()["mean_utilisation_pct"] == 10.0


def test_peak_vram_is_the_maximum_not_the_last_reading():
    mon = Monitor()
    mon.samples = [
        _sample(util=90.0, clock=3000.0, vram=v) for v in (2000.0, 9800.0, 3000.0)
    ]
    assert mon.summary()["peak_vram_mib"] == 9800.0


def test_no_busy_samples_reports_unknown_rather_than_clean():
    mon = Monitor()
    mon.samples = [_sample(util=0.0, clock=180.0)]
    assert mon.summary()["throttled_fraction"] is None


def test_summary_is_none_when_nothing_could_be_sampled():
    # So a report can say "not measured" instead of printing zeros that look
    # like a reading.
    assert Monitor().summary() is None


# -------------------------------------------------------------- warning -----

def test_warning_is_silent_on_a_healthy_point():
    mon = Monitor()
    mon.samples = [_sample(util=95.0, clock=3000.0) for _ in range(10)]
    assert throttling_warning(mon.summary()) == ""


def test_warning_names_the_hardware_reason():
    mon = Monitor()
    mon.samples = [
        _sample(util=95.0, clock=1500.0, reasons="0x0000000000000040", temp=87.0)
        for _ in range(10)
    ]
    warning = throttling_warning(mon.summary())
    assert "HwThermalSlowdown" in warning
    assert "87C" in warning
    # The consequence matters more than the fact: a throttled point understates
    # what the same software does on a card that holds its clocks.
    assert "floor" in warning


def test_warning_handles_an_unmeasured_point():
    assert throttling_warning(None) == ""
