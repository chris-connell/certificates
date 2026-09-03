#!/usr/bin/env python3
"""Run every exact certificate for the Ledger defects paper and report.

Each certificate is executed in a separate interpreter, from this directory,
and is required both to exit with status 0 and to print its characteristic
success line.  The second requirement guards against a future edit that
swallows an assertion without failing the process.

Usage:

    python3 run_all.py                 run everything
    python3 run_all.py --list          show the certificates and exit
    python3 run_all.py -v              stream each certificate's output
    python3 run_all.py --only family   run a subset by name fragment
    python3 run_all.py --skip-missing-deps
                                       report unmet dependencies as SKIP
                                       rather than stopping

Exit status is 0 only when every selected certificate passed.  Uses the
standard library only, so it runs even when the optional dependency of
completion_exact_certificate.py is absent.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Certificate:
    script: str
    summary: str
    expect: tuple[str, ...]
    requires: tuple[str, ...] = field(default=())


# Order matters: harmonic_obstruction_exact imports the family certificate as
# a module, so the family certificate is listed (and run) first.
CERTIFICATES = (
    Certificate(
        script="completion_family_exact_certificate.py",
        summary="uniform first-jet certificate over Z[s,c]/(s^2+c^2-1)",
        expect=("uniform certificate over Z[s,c]/(s^2+c^2-1): PASS",),
    ),
    Certificate(
        script="exact_completion_check.py",
        summary="polarization inverse for the SO(2)-equivalent representative",
        expect=("EXACT CERTIFICATE PASSED",),
    ),
    Certificate(
        script="completion_exact_certificate.py",
        summary="rational rank computation at (s,c)=(3/5,4/5)",
        expect=(
            "exact tensor and polynomial certificate: PASS",
            "verified over Q:",
        ),
        requires=("numpy",),
    ),
    Certificate(
        script="harmonic_obstruction_exact.py",
        summary="scalar audit of N_A, C_A, F_A and the directional violation",
        expect=("exact harmonic-obstruction audit: PASS",),
    ),
)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


def missing_dependencies(certificate: Certificate) -> list[str]:
    return [name for name in certificate.requires
            if importlib.util.find_spec(name) is None]


def run(certificate: Certificate, verbose: bool, timeout: float | None):
    """Execute one certificate.  Returns (status, elapsed, output, detail)."""
    path = HERE / certificate.script
    if not path.is_file():
        return FAIL, 0.0, "", f"{certificate.script} not found in {HERE}"

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=HERE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return FAIL, time.perf_counter() - started, "", f"timed out after {timeout}s"
    elapsed = time.perf_counter() - started
    output = completed.stdout or ""

    if verbose:
        print(output.rstrip())

    if completed.returncode != 0:
        return FAIL, elapsed, output, f"exit status {completed.returncode}"

    for line in certificate.expect:
        if line not in output:
            return FAIL, elapsed, output, f"missing expected output: {line!r}"

    return PASS, elapsed, output, ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the exact certificates and report pass or fail.",
    )
    parser.add_argument("--list", action="store_true",
                        help="list the certificates and exit")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="stream each certificate's output as it runs")
    parser.add_argument("--only", action="append", metavar="FRAGMENT",
                        help="run only certificates whose filename contains "
                             "FRAGMENT; repeatable")
    parser.add_argument("--skip-missing-deps", action="store_true",
                        help="report a certificate whose dependencies are "
                             "absent as SKIP instead of failing")
    parser.add_argument("--timeout", type=float, default=None, metavar="SECONDS",
                        help="per-certificate timeout (default: none)")
    args = parser.parse_args()

    selected = CERTIFICATES
    if args.only:
        fragments = [fragment.lower() for fragment in args.only]
        selected = tuple(
            certificate for certificate in CERTIFICATES
            if any(fragment in certificate.script.lower() for fragment in fragments)
        )
        if not selected:
            print(f"no certificate matches {args.only}", file=sys.stderr)
            return 2

    if args.list:
        width = max(len(c.script) for c in CERTIFICATES)
        for certificate in CERTIFICATES:
            extra = (f"  [requires {', '.join(certificate.requires)}]"
                     if certificate.requires else "")
            print(f"{certificate.script:<{width}}  {certificate.summary}{extra}")
        return 0

    if args.verbose and args.only is None:
        print(f"Python {sys.version.split()[0]} at {sys.executable}\n")

    results = []
    for certificate in selected:
        missing = missing_dependencies(certificate)
        if missing:
            names = ", ".join(missing)
            if args.skip_missing_deps:
                print(f"{SKIP}  {certificate.script}  ({names} not installed)")
                results.append((certificate, SKIP, 0.0, "", names))
                continue
            print(f"{FAIL}  {certificate.script}  ({names} not installed)",
                  file=sys.stderr)
            print(f"\nInstall it with:  {sys.executable} -m pip install "
                  f"-r requirements.txt", file=sys.stderr)
            print("Or re-run with --skip-missing-deps to continue without it.",
                  file=sys.stderr)
            return 1

        if args.verbose:
            print(f"--- {certificate.script}")
        status, elapsed, output, detail = run(certificate, args.verbose,
                                              args.timeout)
        results.append((certificate, status, elapsed, output, detail))
        note = f"  ({detail})" if detail else ""
        if args.verbose:
            print(f"--- {status} in {elapsed:.1f}s{note}\n")
        else:
            print(f"{status}  {certificate.script}  {elapsed:5.1f}s{note}")
            if status == FAIL and output:
                print("\n".join("      " + line
                                for line in output.rstrip().splitlines()))

    passed = sum(1 for _, status, *_ in results if status == PASS)
    failed = sum(1 for _, status, *_ in results if status == FAIL)
    skipped = sum(1 for _, status, *_ in results if status == SKIP)
    total_time = sum(elapsed for _, _, elapsed, *_ in results)

    parts = [f"{passed} passed"]
    if failed:
        parts.append(f"{failed} failed")
    if skipped:
        parts.append(f"{skipped} skipped")
    print(f"\n{', '.join(parts)} in {total_time:.1f}s")

    if failed or skipped:
        return 1
    print("All exact certificates verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
