"""Run every Part 1 script, in order.

Each script is standalone and can still be run on its own; this just runs the
whole part with one command and reports what happened.
"""

from __future__ import annotations

import runpy
import time
from pathlib import Path

PART_DIR = Path(__file__).resolve().parent / "part1"
SCRIPTS = (
    "characterize_plant.py",
    "characterize_camera.py",
    "analyze_fundamental_limits.py",
)


def main() -> int:
    print(f"Part 1: running {len(SCRIPTS)} scripts\n")
    failures: list[str] = []
    durations: dict[str, float] = {}

    for name in SCRIPTS:
        print("=" * 72)
        print(f"  {name}")
        print("=" * 72, flush=True)
        started = time.perf_counter()
        try:
            runpy.run_path(str(PART_DIR / name), run_name="__main__")
        except Exception as error:
            # Caught deliberately: one broken script should not hide the
            # results of the others, so the run continues and every failure is
            # collected for the summary below.
            failures.append(name)
            print(f"\n  FAILED: {type(error).__name__}: {error}", flush=True)
        durations[name] = time.perf_counter() - started
        print(f"\n  ({durations[name]:.1f} s)\n", flush=True)

    print("=" * 72)
    print("  Summary")
    print("=" * 72)
    for name in SCRIPTS:
        status = "failed" if name in failures else "ok"
        print(f"  {name:<34} {status:>8} {durations[name]:8.1f} s")
    print(f"\n  total {sum(durations.values()):.1f} s")
    if failures:
        print(f"  {len(failures)} script(s) failed: {', '.join(failures)}")
        return 1
    print("  All scripts ran. Plots are in outputs/part1/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
