"""多线程分块下载 CIFAR-10（PyTorch 官方源在国内只有十几 KB/s）。

把下载结果保存为 torchvision 期望的 cifar-10-python.tar.gz，
之后 torchvision.datasets.CIFAR10(download=True) 会自动复用，无需重新下载。

用法：python scripts/download_cifar10.py --out data
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

URLS = [
    # HuggingFace 镜像站上的同一份数据（官方源在国外，速度只有十几 KB/s）
    "https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet",
    "https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet",
]

PARQUET_DIR = "parquet"
THREADS = 8


def head_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        return int(response.headers["Content-Length"])


def download_range(url: str, path: Path, start: int, end: int, part: int) -> None:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    with open(f"{path}.part{part}", "wb") as f:
        f.write(data)


def download(url: str, path: Path, threads: int = THREADS) -> None:
    total = head_size(url)
    print(f"{path.name}: {total / 1e6:.1f} MB")

    chunk = total // threads + 1
    jobs = []
    for i in range(threads):
        start = i * chunk
        end = min(total - 1, start + chunk - 1)
        if start > end:
            break
        jobs.append((start, end, i))

    start_time = time.time()
    pool = []
    for start, end, i in jobs:
        t = threading.Thread(target=download_range, args=(url, path, start, end, i))
        t.start()
        pool.append((t, start, end, i))
    for t, *_ in pool:
        t.join()

    with open(path, "wb") as out:
        for _, start, end, i in pool:
            with open(f"{path}.part{i}", "rb") as part:
                out.write(part.read())
            os.remove(f"{path}.part{i}")

    elapsed = time.time() - start_time
    print(f"  完成 {os.path.getsize(path) / 1e6:.1f} MB，用时 {elapsed:.0f}s，"
          f"平均 {total / 1e3 / elapsed:.0f} KB/s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data")
    parser.add_argument("--threads", type=int, default=THREADS)
    args = parser.parse_args()

    target_dir = Path(args.out) / PARQUET_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    for url in URLS:
        path = target_dir / url.rsplit("/", 1)[-1]
        if path.exists():
            print(f"已存在，跳过：{path}")
            continue
        download(url, path, args.threads)

    print("\n下载完成，接下来运行：python scripts/prepare_cifar10.py")


if __name__ == "__main__":
    main()
