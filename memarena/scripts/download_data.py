#!/usr/bin/env python3
"""Download public benchmark files into ./data (which is git-ignored).

    python scripts/download_data.py longmemeval-oracle longmemeval-s locomo

Sources:
  LongMemEval (cleaned, 2025-09): https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned  (MIT)
  LoCoMo:                         https://github.com/snap-research/locomo
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request

FILES = {
    "longmemeval-oracle": ("longmemeval", "longmemeval_oracle.json",
                           "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json"),
    "longmemeval-s": ("longmemeval", "longmemeval_s_cleaned.json",
                      "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"),
    "longmemeval-m": ("longmemeval", "longmemeval_m_cleaned.json",
                      "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json"),
    "locomo": ("locomo", "locomo10.json",
               "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", choices=sorted(FILES))
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    args = ap.parse_args()
    for name in args.names:
        sub, filename, url = FILES[name]
        target_dir = os.path.join(args.dir, sub)
        os.makedirs(target_dir, exist_ok=True)
        target = os.path.join(target_dir, filename)
        if os.path.exists(target) and os.path.getsize(target) > 0:
            print(f"{name}: already present at {target}")
            continue
        print(f"{name}: downloading {url}", flush=True)
        tmp = target + ".part"
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 20)
        os.replace(tmp, target)
        print(f"{name}: {os.path.getsize(target) / 1e6:.1f} MB -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
