#!/usr/bin/env python3
"""Download the public Kaggle archive with TLS verification; verify train.csv."""
import argparse
import hashlib
import io
from pathlib import Path
import ssl
import urllib.request
import zipfile
import certifi

URL='https://www.kaggle.com/api/v1/datasets/download/parisrohan/credit-score-classification'
EXPECTED='d2ebcc056a64c48710b1aeb96777155835d372d7ad202529f64666011d214da0'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('data/train.csv'))
    parser.add_argument('--archive',type=Path,help='Use a previously downloaded Kaggle ZIP')
    args=parser.parse_args()
    if args.output.exists():
        if hashlib.sha256(args.output.read_bytes()).hexdigest()==EXPECTED:
            print(f'Already verified: {args.output}');return
        raise FileExistsError('Existing data does not match; choose a new output path')
    if args.archive:
        archive=args.archive.read_bytes()
    else:
        context=ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(URL,timeout=120,context=context) as response:
            archive=response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        raw=z.read('train.csv')
    digest=hashlib.sha256(raw).hexdigest()
    if digest!=EXPECTED:
        raise ValueError(f'Source changed: expected {EXPECTED}, received {digest}; do not silently use a new dataset')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(raw)
    print(f'Verified {len(raw)} bytes at {args.output}; SHA256 {digest}')

if __name__=='__main__':main()
