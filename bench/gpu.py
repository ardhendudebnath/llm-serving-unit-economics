"""GPU telemetry sampled alongside a load sweep.

Three of the plan's requirements land here, and one hazard specific to this
project's hardware.

**Peak VRAM**, because the quantisation ladder is partly a memory story and
re-running a sweep to recover a number nobody recorded is money set on fire.

**Utilisation**, because it is the figure that decides whether a cost number is
honest. A per-request cost computed on a card running at 12 % is dominated by
idle time, and someone will ask.

**Clocks and temperature**, because the measurements for this project run on a
*laptop* GPU. A laptop card under sustained load throttles: it will hold boost
clocks for the first part of a sweep and then drop, so a 120-second run at high
arrival rate can measure the cooling system rather than the server. Throttling
is therefore detected and reported per load point, not discovered afterwards in
a chart that slopes the wrong way.

Nothing here is inferred. Every field comes from `nvidia-smi`, and when it is
absent the sample is recorded as unavailable rather than filled with a plausible
default -- the same rule the cost model applies to prices.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Self

#: Queried fields, in the order nvidia-smi is asked for them.
_FIELDS = (
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "temperature.gpu",
    "clocks.current.sm",
    "clocks.max.sm",
    "power.draw",
    "clocks_throttle_reasons.active",
)

#: How often to sample. Fast enough to catch a thermal drop partway through a
#: 120s load point, slow enough that the subprocess cost is irrelevant.
SAMPLE_INTERVAL_S = 2.0

#: SM clock below this fraction of maximum counts as throttled -- but only
#: while the GPU is actually working. Not 1.0: boost clocks fluctuate normally,
#: and a threshold at maximum would report every healthy sweep as throttled.
THROTTLE_RATIO = 0.85

#: Below this utilisation the clock ratio means nothing. An idle GPU downclocks
#: hard for power saving -- this card sits at 180 MHz against a 3090 MHz
#: maximum, a ratio of 0.06 -- and calling that "throttled" would flag every
#: idle moment as a measurement problem. Found by sampling an idle card, which
#: is exactly why this is measured rather than assumed.
BUSY_UTILISATION_PCT = 50.0

#: nvidia-smi reports throttle reasons as a hex bitmask, not a readable string.
#: These are NVML's clocksThrottleReasons bits.
_THROTTLE_BITS: tuple[tuple[int, str, bool], ...] = (
    # (bit, name, counts as a performance problem)
    (0x0001, "GpuIdle", False),
    (0x0002, "ApplicationsClocksSetting", False),
    (0x0004, "SwPowerCap", True),
    (0x0008, "HwSlowdown", True),
    (0x0010, "SyncBoost", False),
    (0x0020, "SwThermalSlowdown", True),
    (0x0040, "HwThermalSlowdown", True),
    (0x0080, "HwPowerBrakeSlowdown", True),
    (0x0100, "DisplayClockSetting", False),
)


def decode_throttle_reasons(raw: str) -> tuple[list[str], bool]:
    """Turn the hex bitmask into names, and say whether any limits performance.

    Idle and applications-clock settings are reported but do not count: the
    first is normal for an unused card and the second is a deliberate cap.
    """
    try:
        bits = int(raw, 16) if raw.lower().startswith("0x") else int(raw)
    except (ValueError, AttributeError):
        return ([], False)

    names = [name for bit, name, _ in _THROTTLE_BITS if bits & bit]
    limiting = any(bits & bit for bit, _, counts in _THROTTLE_BITS if counts)
    return (names, limiting)


@dataclass(slots=True)
class Sample:
    at: float
    memory_used_mib: float
    memory_total_mib: float
    utilisation_pct: float
    temperature_c: float
    sm_clock_mhz: float
    sm_clock_max_mhz: float
    power_w: float
    throttle_reasons: str = ""

    @property
    def clock_ratio(self) -> float:
        if self.sm_clock_max_mhz <= 0:
            return 1.0
        return self.sm_clock_mhz / self.sm_clock_max_mhz

    @property
    def busy(self) -> bool:
        """Whether the card was working when this sample was taken."""
        return self.utilisation_pct >= BUSY_UTILISATION_PCT

    @property
    def throttled(self) -> bool:
        """True only when performance was actually being limited.

        The hardware's own reason bits are authoritative and are trusted
        whenever they are set. Falling back to the clock ratio requires the GPU
        to be busy first -- an idle card downclocks to a small fraction of its
        maximum, and treating that as throttling would flag every quiet moment
        as a broken measurement.
        """
        _, limiting = decode_throttle_reasons(self.throttle_reasons)
        if limiting:
            return True
        return self.busy and self.clock_ratio < THROTTLE_RATIO


def available() -> bool:
    return shutil.which("nvidia-smi") is not None


def sample_once() -> Sample | None:
    """One reading, or None when nvidia-smi is unavailable or unparseable."""
    if not available():
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(_FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None

    # First GPU only. This project is explicitly single-GPU; a multi-GPU host
    # would need per-device attribution and the report does not claim to do it.
    line = out.splitlines()[0] if out else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != len(_FIELDS):
        return None

    def _num(raw: str) -> float:
        try:
            return float(raw)
        except ValueError:
            # "[N/A]" on cards that do not expose a field -- notably power on
            # many laptop GPUs. Recorded as absent, not as zero.
            return float("nan")

    return Sample(
        at=time.time(),
        memory_used_mib=_num(parts[0]),
        memory_total_mib=_num(parts[1]),
        utilisation_pct=_num(parts[2]),
        temperature_c=_num(parts[3]),
        sm_clock_mhz=_num(parts[4]),
        sm_clock_max_mhz=_num(parts[5]),
        power_w=_num(parts[6]),
        throttle_reasons=parts[7],
    )


@dataclass
class Monitor:
    """Samples the GPU on a background thread for the length of a load point.

    Used as a context manager so a sweep cannot forget to stop it:

        with Monitor() as mon:
            ...run the load point...
        row = mon.summary()
    """

    interval_s: float = SAMPLE_INTERVAL_S
    samples: list[Sample] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        if not available():
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 2)

    def _run(self) -> None:
        while not self._stop.is_set():
            if (s := sample_once()) is not None:
                self.samples.append(s)
            self._stop.wait(self.interval_s)

    def summary(self) -> dict | None:
        """What the sweep records beside the latency numbers.

        None when nothing could be sampled, so the report can say "not
        measured" rather than printing zeros that look like a reading.
        """
        if not self.samples:
            return None

        used = [s.memory_used_mib for s in self.samples]
        util = [s.utilisation_pct for s in self.samples]
        temp = [s.temperature_c for s in self.samples]

        # Throttling is judged over the samples where the card was working.
        # Averaging in the idle gaps between load points would dilute a real
        # thermal problem into invisibility.
        busy = [s for s in self.samples if s.busy]
        throttled = [s for s in busy if s.throttled]
        ratios = [s.clock_ratio for s in busy] or [s.clock_ratio for s in self.samples]

        reasons: set[str] = set()
        for s in self.samples:
            names, limiting = decode_throttle_reasons(s.throttle_reasons)
            if limiting:
                reasons.update(names)

        return {
            "samples": len(self.samples),
            "busy_samples": len(busy),
            "peak_vram_mib": round(max(used), 1),
            "total_vram_mib": round(self.samples[0].memory_total_mib, 1),
            # Over every sample, including idle: this is the utilisation figure
            # the cost model needs, and inflating it by counting only busy
            # moments would be exactly the dishonesty it exists to prevent.
            "mean_utilisation_pct": round(sum(util) / len(util), 1),
            "peak_temperature_c": round(max(temp), 1),
            "mean_sm_clock_ratio_busy": round(sum(ratios) / len(ratios), 3),
            # The number that says whether this load point can be trusted.
            # Zero busy samples means the load never engaged the GPU, which is
            # reported as unknown rather than as a clean run.
            "throttled_fraction": (
                round(len(throttled) / len(busy), 3) if busy else None
            ),
            "throttle_reasons": sorted(reasons),
        }


def throttling_warning(summary: dict | None) -> str:
    """A sentence for the sweep log, or empty when the point looks sound.

    Deliberately loud. A throttled load point is not a slightly noisy one -- it
    measured the cooling system, and its throughput number understates what the
    same software does on a card that can hold its clocks.
    """
    if not summary:
        return ""
    fraction = summary.get("throttled_fraction")
    if fraction is None or fraction < 0.1:
        return ""
    reasons = ", ".join(summary.get("throttle_reasons") or []) or "clock drop"
    return (
        f"GPU throttled for {fraction:.0%} of the busy samples at this point "
        f"({reasons}; mean SM clock {summary['mean_sm_clock_ratio_busy']:.0%} "
        f"of max, peak {summary['peak_temperature_c']:.0f}C) -- throughput "
        "here is a floor, not the card's sustained rate"
    )
