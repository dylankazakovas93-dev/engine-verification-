#!/usr/bin/env python
from __future__ import annotations
import argparse, shutil
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("name")
    a=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    dest=root/"strategies"/a.name
    if dest.exists():
        raise SystemExit(f"{dest} already exists")
    shutil.copytree(root/"templates"/"strategy",dest)
    print(dest)
    print("Next: freeze SPECIFICATION.md, number REQUIREMENTS.md, then write ORACLE.md before candidate code.")
if __name__=="__main__": main()
