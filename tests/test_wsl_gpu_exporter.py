"""The WSL stand-in for dcgm-exporter: the series the rules read, and no invented zeros."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "wsl_gpu_exporter", ROOT / "deploy" / "observability" / "wsl-gpu-exporter.py"
)
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)

ROW = ["0", "87", "10840", "1387", "112.45", "2520"]


def test_serves_the_three_dcgm_series_the_rules_and_dashboard_read():
    text = exporter.render([ROW])
    assert 'DCGM_FI_DEV_GPU_UTIL{gpu="0",source="nvidia-smi"} 87.0' in text
    assert 'DCGM_FI_DEV_FB_USED{gpu="0",source="nvidia-smi"} 10840.0' in text
    assert 'DCGM_FI_DEV_FB_FREE{gpu="0",source="nvidia-smi"} 1387.0' in text


def test_every_series_is_typed_and_labelled_as_a_stand_in():
    text = exporter.render([ROW])
    for name in exporter.SERIES:
        assert f"# TYPE {name} gauge" in text
    samples = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    assert samples and all('source="nvidia-smi"' in ln for ln in samples)


def test_a_field_nvidia_smi_cannot_report_is_omitted_not_zeroed():
    # A laptop GPU can answer [N/A] for power. Reporting 0 W would be a
    # measurement nobody took.
    text = exporter.render([["0", "87", "10840", "1387", "[N/A]", "2520"]])
    assert "DCGM_FI_DEV_POWER_USAGE{" not in text
    assert "DCGM_FI_DEV_GPU_UTIL{" in text
