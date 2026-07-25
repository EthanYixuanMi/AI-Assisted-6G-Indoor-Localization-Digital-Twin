"""Optionally capture the real Streamlit UI using Playwright.

This helper is deliberately separate from the browser-free core exporters.
Missing Playwright/browser support therefore never prevents the experiment
pipeline or static dashboard exports from succeeding.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = (
    "normal",
    "high_noise",
    "strong_blockage",
    "anchor_failure",
    "domain_shift",
)


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _wait_until_ready(url: str, process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Streamlit stopped before becoming ready (exit {process.returncode})."
            )
        try:
            with urlopen(url, timeout=1.0) as response:
                if response.status < 500:
                    return
        except (URLError, TimeoutError):
            time.sleep(0.3)
    raise RuntimeError(f"Timed out after {timeout:.0f}s waiting for {url}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture the rendered Streamlit dashboard with Playwright."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "report_assets" / "screenshots" / "streamlit",
    )
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument(
        "--scenario",
        choices=(*SCENARIOS, "all"),
        default="all",
    )
    return parser


def _check_optional_dependencies() -> None:
    missing = [
        name
        for name in ("streamlit", "playwright")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            "Optional browser capture is unavailable because "
            + ", ".join(missing)
            + " is not installed. The required browser-free command remains "
            "`python scripts/export_dashboard_snapshot.py`. To enable this "
            "helper, install Playwright and its Chromium browser."
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )
    try:
        _check_optional_dependencies()
        from playwright.sync_api import sync_playwright

        port = args.port or _available_port()
        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "app.py"),
            "--server.headless=true",
            "--server.address=127.0.0.1",
            f"--server.port={port}",
            "--browser.gatherUsageStats=false",
        ]
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        base_url = f"http://127.0.0.1:{port}"
        try:
            _wait_until_ready(base_url, process, args.timeout)
            output_dir = args.output_dir.resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            scenarios = (
                SCENARIOS if args.scenario == "all" else (args.scenario,)
            )
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(viewport={"width": 1800, "height": 1100})
                for scenario in scenarios:
                    page.goto(
                        f"{base_url}/?scenario={scenario}&capture=1",
                        wait_until="networkidle",
                        timeout=int(args.timeout * 1000),
                    )
                    page.get_by_text(
                        "AI-Assisted 6G Indoor Localization Digital Twin",
                        exact=False,
                    ).first.wait_for(timeout=int(args.timeout * 1000))
                    output = output_dir / f"streamlit_{scenario}.png"
                    page.screenshot(path=str(output), full_page=True)
                    logging.info("Captured %s", output)
                browser.close()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    except Exception as exc:
        logging.error("Streamlit browser capture failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
