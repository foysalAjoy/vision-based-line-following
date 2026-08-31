import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORLDS = ROOT / "worlds"

WINDOWS_GUESSES = [
    r"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe",
    r"C:\Program Files\Webots\webots.exe",
    r"C:\Users\Public\Webots\msys64\mingw64\bin\webots.exe",
]
MAC_GUESSES = ["/Applications/Webots.app/Contents/MacOS/webots"]
LINUX_GUESSES = ["/usr/local/webots/webots", "/snap/bin/webots"]


def find_webots(explicit):
    if explicit:
        p = Path(explicit)
        if p.exists():
            return str(p)
        sys.exit(f"ERROR: --webots path does not exist: {p}")

    found = shutil.which("webots")
    if found:
        return found

    env = os.environ.get("WEBOTS_HOME")
    if env:
        for rel in ("msys64/mingw64/bin/webots.exe", "webots", "webots.exe"):
            p = Path(env) / rel
            if p.exists():
                return str(p)

    system = platform.system()
    guesses = (WINDOWS_GUESSES if system == "Windows"
               else MAC_GUESSES if system == "Darwin" else LINUX_GUESSES)
    for g in guesses:
        if Path(g).exists():
            return g

    sys.exit("ERROR: could not locate the Webots executable.\n"
             "Pass it explicitly, e.g.\n"
             '  python scripts/run_all_experiments.py --webots '
             '"C:/Program Files/Webots/msys64/mingw64/bin/webots.exe"')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3,
                    help="runs per world (3 gives a usable mean +/- SD)")
    ap.add_argument("--only", default="all",
                    choices=["all", "baseline", "robust"],
                    help="restrict the sweep to one condition family")
    ap.add_argument("--webots", default=None, help="path to the webots binary")
    ap.add_argument("--gui", action="store_true",
                    help="show the Webots window instead of running headless")
    ap.add_argument("--timeout", type=int, default=600,
                    help="seconds before a stuck run is killed")
    args = ap.parse_args()

    webots = find_webots(args.webots)

    worlds = sorted(WORLDS.glob("exp_*.wbt"))
    if args.only == "baseline":
        worlds = [w for w in worlds if "baseline" in w.stem]
    elif args.only == "robust":
        worlds = [w for w in worlds if "robust" in w.stem]

    if not worlds:
        sys.exit("No exp_*.wbt worlds found.\n"
                 "Run: python scripts/make_experiment_worlds.py")

    total = len(worlds) * args.repeats
    print("=" * 62)
    print(" BATCH EXPERIMENT RUNNER")
    print("=" * 62)
    print(f" webots  : {webots}")
    print(f" worlds  : {len(worlds)}")
    print(f" repeats : {args.repeats}")
    print(f" total   : {total} runs")
    print(f" mode    : {'GUI' if args.gui else 'headless (fast)'}")
    print("=" * 62)

    done = 0
    failed = []
    t_start = time.time()

    for rep in range(1, args.repeats + 1):
        for world in worlds:
            done += 1
            cmd = [webots, "--batch", "--stdout", "--stderr"]
            if args.gui:
                cmd += ["--mode=fast"]
            else:
                cmd += ["--mode=fast", "--no-rendering", "--minimize"]
            cmd.append(str(world))

            print(f"\n[{done}/{total}] rep {rep}  {world.name}")
            try:
                r = subprocess.run(cmd, timeout=args.timeout)
                if r.returncode not in (0, None):
                    print(f"    returncode {r.returncode}")
            except subprocess.TimeoutExpired:
                print(f"    TIMEOUT after {args.timeout}s - killed")
                failed.append(f"{world.name} (rep {rep})")

    mins = (time.time() - t_start) / 60.0
    print("\n" + "=" * 62)
    print(f" finished {done} runs in {mins:.1f} min")
    if failed:
        print(" timed out:")
        for f in failed:
            print(f"   {f}")
    print(" Next: python scripts/make_dissertation_outputs.py")
    print("=" * 62)


if __name__ == "__main__":
    main()
