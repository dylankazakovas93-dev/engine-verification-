#!/usr/bin/env python
from __future__ import annotations
import argparse, re, shutil
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--root",default=None,help="repository root; useful for scaffold validation")
    ap.add_argument("--dry-run",action="store_true")
    a=ap.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_]*",a.name):
        raise SystemExit("strategy name must match [a-z][a-z0-9_]*")
    root=Path(a.root).resolve() if a.root else Path(__file__).resolve().parents[1]
    dest=root/"strategies"/a.name
    if dest.exists():
        raise SystemExit(f"{dest} already exists")
    if a.dry_run:
        print(dest)
        print("DRY RUN: would copy the frozen-spec-first strategy scaffold")
        return
    shutil.copytree(root/"templates"/"strategy",dest)
    (dest/"tests").mkdir()
    (dest/"tests"/".gitkeep").touch()
    print(dest)
    print("STOP GATE: freeze SPECIFICATION.md, number REQUIREMENTS.md, and review ORACLE.md before implementing candidate code.")
if __name__=="__main__": main()
