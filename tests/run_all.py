"""Run every tests/test_*.py in its own process and summarise.

    .venv\\Scripts\\python.exe tests\\run_all.py            (Windows)
    python tests/run_all.py                                (any OS, with the project's environment active)
    python tests/run_all.py test_cli test_samples          (only these)
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
wanted = set(a.removesuffix(".py") for a in sys.argv[1:])
tests = [p for p in sorted(HERE.glob("test_*.py")) if not wanted or p.stem in wanted]
results = []
for p in tests:
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(p)], cwd=HERE, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    checks = [line.strip() for line in out.splitlines() if line.strip().startswith(("PASS", "FAIL", "NOTE"))]
    results.append((p.stem, proc.returncode, time.time() - t0, checks, out))
    print(f"{'PASSED' if proc.returncode == 0 else 'FAILED'}  {p.stem}  ({time.time() - t0:.0f}s, "
          f"{sum(c.startswith('PASS') for c in checks)} checks passed)")
    for c in checks:
        if not c.startswith("PASS"):
            print("     ", c)
    if proc.returncode != 0 and not any(c.startswith("FAIL") for c in checks):
        print("      (crashed) " + "\n      ".join(out.strip().splitlines()[-8:]))
failed = [r[0] for r in results if r[1] != 0]
print(f"\n{len(results) - len(failed)} of {len(results)} test files passed" + (f"; failed: {failed}" if failed else ""))
sys.exit(1 if failed else 0)
