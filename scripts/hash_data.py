#!/usr/bin/env python
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha256(path: Path, chunk=8*1024*1024):
    h=hashlib.sha256()
    size=0
    with path.open("rb") as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b); size += len(b)
    return h.hexdigest(), size

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--out",default=None)
    a=ap.parse_args()
    p=Path(a.data)
    digest,size=sha256(p)
    result={"path":str(p.resolve()),"bytes":size,"sha256":digest}
    text=json.dumps(result,indent=2)
    print(text)
    if a.out: Path(a.out).write_text(text+"\n",encoding="utf-8")
if __name__=="__main__": main()
