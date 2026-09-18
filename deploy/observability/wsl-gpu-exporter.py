"""A DCGM-shaped GPU exporter for WSL2, where DCGM itself does not run.

    python3 wsl-gpu-exporter.py --port 9400

The dashboard and alert rules read DCGM series (DCGM_FI_DEV_GPU_UTIL,
DCGM_FI_DEV_FB_USED, DCGM_FI_DEV_FB_FREE), because on a real GPU node
dcgm-exporter is the standard source. Under WSL2 DCGM cannot run: the GPU is
reached through /dev/dxg, not the device nodes DCGM needs. nvidia-smi does
work, so this serves those names from nvidia-smi on every scrape, in the units
DCGM uses (percent, MiB), plus power draw and SM clock for context.

It is a stand-in, and says so. Every series carries source="nvidia-smi", so a
query can tell it from real DCGM. Nothing is derived or defaulted: a failed
nvidia-smi call returns HTTP 503 rather than zeros, because a zero
utilisation would fire the idle-GPU cost alert for a card that may be busy.
"""

from __future__ import annotations

import argparse
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

#: nvidia-smi fields, in query order. Column 0 is the GPU index.
FIELDS = ("index", "utilization.gpu", "memory.used", "memory.free",
          "power.draw", "clocks.sm")

#: DCGM series name -> (help text, column in FIELDS).
SERIES = {
    "DCGM_FI_DEV_GPU_UTIL": ("GPU utilization (in %).", 1),
    "DCGM_FI_DEV_FB_USED": ("Framebuffer memory used (in MiB).", 2),
    "DCGM_FI_DEV_FB_FREE": ("Framebuffer memory free (in MiB).", 3),
    "DCGM_FI_DEV_POWER_USAGE": ("Power draw (in W).", 4),
    "DCGM_FI_DEV_SM_CLOCK": ("SM clock frequency (in MHz).", 5),
}

UNAVAILABLE = {"[N/A]", "N/A", "[Not Supported]", ""}


def read() -> list[list[str]]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(FIELDS)}",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=True,
    ).stdout
    return [[cell.strip() for cell in line.split(",")]
            for line in out.splitlines() if line.strip()]


def render(rows: list[list[str]]) -> str:
    """Prometheus text format. A field nvidia-smi cannot report is omitted."""
    lines: list[str] = []
    for name, (help_text, column) in SERIES.items():
        lines += [f"# HELP {name} {help_text}", f"# TYPE {name} gauge"]
        for row in rows:
            value = row[column]
            if value in UNAVAILABLE:
                continue
            lines.append(f'{name}{{gpu="{row[0]}",source="nvidia-smi"}} {float(value)}')
    return "\n".join(lines) + "\n"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/metrics":
            self.send_error(404)
            return
        try:
            body = render(read()).encode()
        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
            self.send_error(503, f"nvidia-smi failed: {exc}")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass  # one line per scrape every 10 s is noise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=9400)
    args = ap.parse_args()
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
