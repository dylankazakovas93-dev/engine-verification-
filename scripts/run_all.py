#!/usr/bin/env python
from __future__ import annotations
import subprocess, sys
cmd=[sys.executable,"-m","pytest","-q"]
print("+"," ".join(cmd))
r=subprocess.run(cmd)
raise SystemExit(r.returncode)
