import os
import pyarrow as pa
import lance
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.datasets import Food101


def create_lance_from_classification_dataset(root_path="data/food101", 
                                             output_path="data/food101.lance", 
                                             dataset_name="FOOD101", 
                                             fragment_size=10000, 
                                             batch_size=10000, 
                                             image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
    ])

    if dataset_name == "CIFAR100":
        dataset = datasets.CIFAR100(root=root_path, train=True, download=True)
    elif dataset_name == "CIFAR10":
        dataset = datasets.CIFAR10(root=root_path, train=True, download=True)
    elif dataset_name == "PLACES365":
        dataset = ImageFolder(root=root_path, transform=transform)
    elif dataset_name == "FOOD101":
        dataset = Food101(root=root_path, split="train", download=True)
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
                yield pa.record_batch(
                    [pa.array(images, type=pa.binary()), pa.array(labels, type=pa.int64())],
                    names=["image", "label"]
                )
                images, labels = [], []
        if images:
            yield pa.record_batch(
                [pa.array(images, type=pa.binary()), pa.array(labels, type=pa.int64())],
                names=["image", "label"]
            )

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

# Usage:
# For Food101, run with:
#   create_lance_from_classification_dataset(dataset_name="FOOD101", root_path="data/food101", output_path="data/food101.lance")

if __name__ == "__main__":
    create_lance_from_classification_dataset()