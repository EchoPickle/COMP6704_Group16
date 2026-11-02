#!/usr/bin/env python3
"""Obtain the pinned reference archive locally and verify its SHA-256."""
import argparse
import hashlib
import io
from pathlib import Path
import ssl
import urllib.request
import zipfile

REFERENCE_COMMIT = "5624d97b6771cc3814a2063d151db861051976c6"
URL = (
    "https://raw.githubusercontent.com/Cindy0725/"
    f"COMP6704-Group-24-Convex-Optimization/{REFERENCE_COMMIT}/data.zip"
)
EXPECTED_SHA256 = "a9c84b961b361a7f41fe80ef85d6b01d90068616e48da7ab134bd2a99e58dcac"
REQUIRED_FILES = {"train_data_s.csv", "train_data_m.csv", "train_data_l.csv", "test_data.csv"}


def verify_archive(raw):
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Archive mismatch: expected {EXPECTED_SHA256}, received {digest}")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if not REQUIRED_FILES.issubset(archive.namelist()):
            raise ValueError("Reference archive is missing a required CSV file")
        if archive.testzip() is not None:
            raise ValueError("Reference archive failed its CRC check")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an existing local reference ZIP instead of downloading")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data/data.zip")
    args = parser.parse_args()
    if args.output.exists():
        verify_archive(args.output.read_bytes())
        print(f"Already verified: {args.output}")
        return
    if args.archive:
        raw = args.archive.read_bytes()
    else:
        request = urllib.request.Request(URL, headers={"User-Agent": "COMP6704-Group16-LBFGS"})
        with urllib.request.urlopen(request, timeout=120, context=ssl.create_default_context()) as response:
            raw = response.read()
    verify_archive(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(f"Verified archive: {args.output}; SHA-256 {EXPECTED_SHA256}")


if __name__ == "__main__":
    main()
