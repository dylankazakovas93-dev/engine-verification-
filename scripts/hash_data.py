#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifier.data import sha256_file

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out",default=None)
    a=ap.parse_args()
    p=Path(a.data)
    digest,size=sha256_file(p)
    result={"path":str(p.resolve()),"bytes":size,"sha256":digest}
    text=json.dumps(result,indent=2)
    print(text)
    if a.out: Path(a.out).write_text(text+"\n",encoding="utf-8")
if __name__=="__main__": main()
