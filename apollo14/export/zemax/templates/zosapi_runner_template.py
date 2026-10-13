"""Template for the generated ``build_and_run.py`` bundle entry point.

The wrapper coordinates the generated builder and sweep and verifies that
their HTML outputs use the expected layout version.
"""

BUILD_AND_RUN_SCRIPT = r'''"""Build the model and run the FOV sweep from the bundle directory.

Run this from the bundle folder itself:

    python build_and_run.py

It locates the generated scripts under ``scripts/`` beside this entry point,
reads ``prescription.json`` from the bundle root, and writes results to ``out/``.
"""

import json
import subprocess
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

BUNDLE_DIRECTORY = Path(__file__).resolve().parent
REPORTING_DIRECTORY = BUNDLE_DIRECTORY / "scripts" / "reporting"
sys.path.insert(0, str(REPORTING_DIRECTORY))

from html_plotting import REPORT_LAYOUT_VERSION
from position_space_analysis import detector_directory_name


class _Tee:
    """Mirror text writes to multiple streams."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


@contextmanager
def _tee_console(log_path):
    """Mirror stdout/stderr to a log file under the bundle's out directory."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        with redirect_stdout(_Tee(sys.stdout, handle)):
            with redirect_stderr(_Tee(sys.stderr, handle)):
                yield


def _run_stage(command, *, cwd):
    """Run one stage and relay its output to the console and wrapper log.

    Each generated stage owns its dedicated log file. Relaying stdout here
    also includes that output in ``build_and_run.log`` without two processes
    writing the same stage log concurrently.
    """
    print(f"Running: {' '.join(command)}")
    start = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        return_code = process.wait()
    finally:
        if process.stdout is not None:
            process.stdout.close()
    elapsed = time.perf_counter() - start
    print(f"Finished in {elapsed:.1f}s with exit code {return_code}")
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _expected_html_reports(out_dir, prescription_path):
    """Return every HTML file that the current prescription should generate."""
    with Path(prescription_path).open(encoding="utf-8") as handle:
        prescription = json.load(handle)
    reports = [out_dir / "position_space_index.html"]
    for optical_object in prescription.get("objects", []):
        if optical_object.get("type") != "detector_rectangle":
            continue
        directory = detector_directory_name(optical_object)
        reports.append(out_dir / directory / "position_space.html")
    for specification in prescription.get("analyses", {}).get(
            "pupil_reports", []):
        report_directory = out_dir / "reports" / specification["report_name"]
        reports.append(report_directory / "index.html")
        if specification.get("export_angular_response", False):
            reports.append(report_directory / "angular_response.html")
    coating_report = prescription.get("analyses", {}).get("coating_report")
    if coating_report:
        reports.append(
            out_dir / "reports" / coating_report["report_name"] / "index.html")
    return list(dict.fromkeys(reports))


def _verify_html_report_layout(out_dir, prescription_path):
    """Reject missing or stale HTML instead of leaving old output unnoticed."""
    reports = _expected_html_reports(out_dir, prescription_path)
    missing = [path for path in reports if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Expected HTML reports were not generated: "
            + ", ".join(str(path) for path in missing))

    marker = f'data-apollo14-report-layout="{REPORT_LAYOUT_VERSION}"'
    stale = [
        path for path in reports
        if marker not in path.read_text(encoding="utf-8", errors="replace")
    ]
    if stale:
        raise RuntimeError(
            f"HTML report layout {REPORT_LAYOUT_VERSION!r} was not written to: "
            + ", ".join(str(path) for path in stale)
            + ". Copy build_and_run.py and the complete scripts directory from "
              "the same export bundle "
              "and run build_and_run.py again.")
    print(
        f"Verified HTML report layout {REPORT_LAYOUT_VERSION}: "
        f"{len(reports)} report(s).")


def main():
    bundle_dir = BUNDLE_DIRECTORY
    build_script = bundle_dir / "scripts" / "build_zemax_model.py"
    sweep_script = bundle_dir / "scripts" / "run_fov_sweep.py"
    prescription = bundle_dir / "prescription.json"
    model = bundle_dir / "apollo14_combiner.zmx"
    out_dir = bundle_dir / "out"
    logs_dir = out_dir / "logs"
    wrapper_log = logs_dir / "build_and_run.log"
    build_log = logs_dir / "build_zemax_model.log"
    sweep_log = logs_dir / "run_fov_sweep.log"

    with _tee_console(wrapper_log):
        print(f"Bundle directory: {bundle_dir}")
        print(f"Expected HTML report layout: {REPORT_LAYOUT_VERSION}")
        print(f"  build log : {build_log}")
        print(f"  sweep log : {sweep_log}")
        out_dir.mkdir(parents=True, exist_ok=True)

        _run_stage(
            [sys.executable, "-u", str(build_script), str(prescription),
             str(model)],
            cwd=bundle_dir)

        _run_stage(
            [sys.executable, "-u", str(sweep_script), str(model),
             str(prescription),
             str(out_dir)],
            cwd=bundle_dir)

        _verify_html_report_layout(out_dir, prescription)
        print("Build-and-run completed.")


if __name__ == "__main__":
    main()
'''
