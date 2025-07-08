import os
import pyarrow as pa
import lance
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from torchvision.datasets import ImageFolder


def create_lance_from_classification_dataset(root_path="data/places365", output_path="data/places365.lance", dataset_name="PLACES365", fragment_size=1024, batch_size=1024, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
    ])

    if dataset_name == "CIFAR100":
        dataset = datasets.CIFAR100(root=root_path, train=True, download=True)
    elif dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(root=root_path, train=True, download=True)
    elif dataset_name == "PLACES365":
        dataset = ImageFolder(root=root_path, transform=transform)
    else:
        raise ValueError("Unsupported dataset")

    def record_batch_generator():
        images, labels = [], []
        for idx, (img, label) in enumerate(tqdm(dataset)):
            arr = np.array(img)
            buf = arr.tobytes()
            images.append(buf)
            labels.append(label)
            if (idx + 1) % batch_size == 0:
                yield pa.record_batch({
                    "image": pa.array(images, type=pa.binary()),
                    "label": pa.array(labels, type=pa.int64())
                })
                images, labels = [], []
        if images:
            yield pa.record_batch({
                "image": pa.array(images, type=pa.binary()),
                "label": pa.array(labels, type=pa.int64())
            })

    schema = pa.schema([
        ("image", pa.binary()),
        ("label", pa.int64())
    ])

    lance.write_dataset(
        record_batch_generator(),
        schema=schema,
        uri=output_path,
        mode="overwrite",
        max_rows_per_file=fragment_size
    )

    print(f"Lance dataset written to {output_path} with fragment size {fragment_size}")


if __name__ == "__main__":
    create_lance_from_classification_dataset()