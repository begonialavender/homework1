"""把 HuggingFace 镜像下载的 parquet 转成 torchvision 期望的 CIFAR-10 二进制格式。

转换后 data/cifar-10-batches-py/ 里就是官方 data_batch_1..5、test_batch 与
batches.meta 的结构，torchvision.datasets.CIFAR10 可以直接读取，无需再联网。

用法：python scripts/prepare_cifar10.py --data data
"""

from __future__ import annotations

import argparse
import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_parquet(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """返回 (data[N,3072] uint8, labels[N])。"""
    frame = pd.read_parquet(path)
    image_column = next(
        c for c in frame.columns
        if frame[c].dtype == object and isinstance(frame[c].iloc[0], dict)
    )
    label_column = next(c for c in frame.columns if "label" in c.lower())

    data = np.empty((len(frame), 3072), dtype=np.uint8)
    for i, item in enumerate(frame[image_column]):
        image = Image.open(io.BytesIO(item["bytes"])).convert("RGB").resize((32, 32))
        data[i] = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).reshape(-1)

    return data, frame[label_column].to_numpy(dtype=np.int64)


def write_batch(path: Path, data: np.ndarray, labels: np.ndarray, name: str) -> None:
    payload = {
        "batch_label": name,
        "labels": labels.tolist(),
        "data": data,
        "filenames": [f"{name}_{i:05d}.png" for i in range(len(labels))],
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  写入 {path.name}（{len(labels)} 张）")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    args = parser.parse_args()

    data_dir = Path(args.data)
    parquet_dir = data_dir / "parquet"
    out_dir = data_dir / "cifar-10-batches-py"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("转换训练集…")
    train_data, train_labels = load_parquet(parquet_dir / "train-00000-of-00001.parquet")
    for i in range(5):
        sl = slice(i * 10000, (i + 1) * 10000)
        write_batch(out_dir / f"data_batch_{i + 1}", train_data[sl], train_labels[sl],
                    f"training batch {i + 1} of 5")

    print("转换测试集…")
    test_data, test_labels = load_parquet(parquet_dir / "test-00000-of-00001.parquet")
    write_batch(out_dir / "test_batch", test_data, test_labels, "testing batch 1 of 1")

    with open(out_dir / "batches.meta", "wb") as f:
        pickle.dump(
            {"num_cases_per_batch": 10000, "label_names": CLASSES, "num_vis": 3072}, f
        )
    print(f"完成：{out_dir}")


if __name__ == "__main__":
    main()
